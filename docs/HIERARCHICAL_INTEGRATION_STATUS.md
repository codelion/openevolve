# Hierarchical Evolution Integration Status

**Last Updated**: 2025-11-17

## Summary

The hierarchical abstraction layer system has been implemented but requires additional integration work to fully connect with the worker process execution model.

## ✅ Completed

### 1. Core Hierarchical Components
- ✅ Five-layer abstraction system (`layers.py`)
- ✅ Evolutionary Memory Graph (`emg.py`)
- ✅ Context compilation (`context.py`)
- ✅ Tiered model selection (`model_tiers.py`)
- ✅ Layer transitions (`transitions.py`)
- ✅ Insight extraction (`insights.py`)
- ✅ Orchestrator (`orchestrator.py`)

### 2. Configuration Support
- ✅ `HierarchicalConfig` in `config.py`
- ✅ Tier model configuration (tier0-tier3)
- ✅ Layer transition triggers
- ✅ EMG and insight settings

### 3. Multi-API Support
- ✅ Per-model `api_base` configuration
- ✅ Automatic API key detection based on URL
  - `moonshot.cn` → `KIMI_API_KEY`
  - `bigmodel.cn`/`z.ai` → `GLM_API_KEY`
  - `openai.com` → `OPENAI_API_KEY`
- ✅ Applied to all tier models in `load_config()`

### 4. Controller Integration
- ✅ Controller initializes `HierarchicalOrchestrator`
- ✅ Orchestrator has access to database and config
- ✅ Orchestrator creates tiered model selector
- ✅ Method `get_ensemble_for_iteration()` implemented

### 5. Bug Fixes
- ✅ Fixed circular import (`NodeType`, `TransitionTriggers`)
- ✅ Fixed evaluator return format (now returns `EvaluationResult`)
- ✅ Added problem-specific `system_message` to all configs

## ❌ Not Completed - Critical Gap

### Worker Process Integration

**Problem**: The hierarchical orchestrator's tiered ensembles are not used in worker processes.

**Current Behavior**:
1. Controller creates `HierarchicalOrchestrator`
2. Orchestrator can determine which tier/ensemble to use for each iteration
3. **But** worker processes always use `config.llm.models` (the base models)
4. Tiered models (tier0-tier3) are never actually used for LLM generation

**Code Flow**:
```
Controller (has orchestrator)
  ↓
ProcessParallelController (no orchestrator access)
  ↓
_submit_iteration(iteration, ...)
  ↓
_run_iteration_worker(iteration, db_snapshot, ...)
  ↓
_lazy_init_worker_components()
  ↓
_worker_llm_ensemble = LLMEnsemble(config.llm.models)  ← Always uses base models!
```

**Evidence**:
- `openevolve/process_parallel.py:105` - Worker creates ensemble from `config.llm.models`
- `openevolve/process_parallel.py:189` - Worker uses `_worker_llm_ensemble` for generation
- `openevolve/hierarchy/orchestrator.py:175` - `get_ensemble_for_iteration()` exists but is never called
- `grep -r "get_ensemble_for_iteration"` - Only found in orchestrator.py, not called anywhere

**Impact**:
- Hierarchical evolution appears to run (no errors)
- But all iterations use the same models
- Cost optimization is not realized
- Strategic reasoning at higher layers doesn't happen
- The system degenerates to standard evolution with overhead

## 🔧 Required Fix

### Solution Design

To properly integrate tiered models, we need to:

**Option 1: Pass Tier Models Per Iteration** (Recommended)
1. Modify `ProcessParallelController.__init__()` to accept `hierarchical_orchestrator`
2. In `_submit_iteration()`:
   ```python
   # Get models for this iteration
   if self.hierarchical_orchestrator:
       tier_ensemble = self.hierarchical_orchestrator.get_ensemble_for_iteration(iteration)
       tier_models = [model.to_dict() for model in tier_ensemble.models]
       db_snapshot["tier_models"] = tier_models
   ```
3. In `_run_iteration_worker()`:
   ```python
   # Use tier models if provided
   if "tier_models" in db_snapshot and db_snapshot["tier_models"]:
       models = [LLMModelConfig(**m) for m in db_snapshot["tier_models"]]
   else:
       models = _worker_config.llm.models

   _worker_llm_ensemble = LLMEnsemble(models)
   ```

**Option 2: Recreate Orchestrator in Worker**
- Pass hierarchical config to worker
- Worker creates its own orchestrator
- Worker calls `get_ensemble_for_iteration(iteration)`
- More overhead but cleaner separation

**Option 3: Pre-compute Tier Assignments**
- At evolution start, pre-compute which tier for each iteration
- Pass tier assignment table to workers
- Workers look up their tier and use corresponding models
- Inflexible but simple

### Implementation Steps

1. **Modify ProcessParallelController**:
   - Add `hierarchical_orchestrator` parameter to `__init__()`
   - Store as `self.hierarchical_orchestrator`

2. **Modify controller.py initialization**:
   ```python
   self.parallel_controller = ProcessParallelController(
       self.config,
       self.evaluation_file,
       self.database,
       self.evolution_tracer,
       file_suffix=self.config.file_suffix,
       hierarchical_orchestrator=self.hierarchical_orchestrator,  # ← Add this
   )
   ```

3. **Modify _submit_iteration**:
   - Call `orchestrator.get_ensemble_for_iteration(iteration)`
   - Extract model configs from ensemble
   - Add to `db_snapshot["tier_models"]`

4. **Modify _lazy_init_worker_components**:
   - Check for `tier_models` in snapshot (passed as global)
   - If present, use those models
   - Otherwise fall back to `config.llm.models`

5. **Update _run_iteration_worker signature**:
   - Current: `_run_iteration_worker(iteration, db_snapshot, parent_id, inspiration_ids)`
   - Option A: Add `tier_models` parameter
   - Option B: Put in `db_snapshot` (cleaner, already passing dict)

### Testing

After implementation, verify:
1. Log which models are used for each iteration
2. Confirm tier progression (Tier 0 → Tier 1 → Tier 2 → Tier 3)
3. Check API calls go to correct endpoints
4. Verify cost is reduced vs using premium models for all

## 📋 Temporary Workaround

Until proper integration is complete, users can manually set the `llm.models` to be the Tier 0 models they want to use throughout evolution. This provides:
- ✅ Consistent model usage
- ✅ API key detection works
- ✅ Cost predictability
- ❌ No hierarchical reasoning benefits
- ❌ No automatic tier escalation

## 📝 Documentation Needed

Once fixed, update:
- `/docs/MULTI_API_SETUP.md` - Remove any caveats about tier integration
- `/examples/*/README.md` - Add actual vs expected tier progression logs
- Add troubleshooting section for verifying tier usage
- Create example log output showing tier transitions

## 🔍 Verification Commands

Check if hierarchical tiers are working:

```bash
# Run with verbose logging
python run_hierarchical.py --config config_multi_api.yaml --iterations 10 2>&1 | grep -i "tier\|ensemble\|model"

# Should see logs like:
# "Using code_details ensemble in normal phase (iteration 1)"
# "Using implementation_patterns ensemble in normal phase (iteration 7)"

# Check API calls
python run_hierarchical.py --config config_multi_api.yaml --iterations 5 2>&1 | grep "HTTP Request: POST"

# Should see different API endpoints:
# POST https://open.bigmodel.cn/...  (GLM for Tier 0-2)
# POST https://api.moonshot.cn/...   (KIMI for Tier 3)
```

## 🎯 Priority

**High Priority** - This is the core value proposition of hierarchical evolution. Without it, the system is just standard evolution with extra overhead.

## 📞 Contact

For questions or to contribute the fix, please open an issue or PR on GitHub.
