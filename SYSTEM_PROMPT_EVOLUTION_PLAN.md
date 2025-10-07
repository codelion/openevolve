# Dynamic System Prompt Evolution Implementation Plan

## Analysis of Current Architecture

The system prompt currently flows like this:
1. **Config file** (`config.yaml`) → `PromptConfig.system_message` (line 193 in [config.py](openevolve/config.py#L193))
2. **Controller init** → Updates LLM models with system_message (line 504 in [config.py](openevolve/config.py#L504))
3. **PromptSampler.build_prompt()** → Uses `self.config.system_message` (line 105 in [prompt/sampler.py](openevolve/prompt/sampler.py#L105))
4. **Worker processes** → Receive serialized config, reconstruct PromptSampler (line 108 in [process_parallel.py](process_parallel.py#L108))

## What Needs to Change

### 1. **Make System Prompt Dynamic** ✅
**Current:** System prompt is static from config file
**Needed:** System prompt can be updated during evolution

**Changes:**
- **[config.py](openevolve/config.py)**: Add `current_system_message` field to track the active prompt (separate from original config)
- **[controller.py](openevolve/controller.py)**: Add method `update_system_prompt(new_prompt: str)` to Controller
- **[prompt/sampler.py](openevolve/prompt/sampler.py)**: Already supports overrides via `system_template_override` (line 29) - leverage this!
- **[database.py](openevolve/database.py)**: Store system prompt history as metadata in checkpoints

### 2. **System Prompt Rewriting Process** 🆕
**Strategy:** Periodic meta-evolution loop that analyzes recent iterations and rewrites the system prompt

**New Component: `SystemPromptRewriter`** (new file: `openevolve/system_prompt_rewriter.py`)

```python
class SystemPromptRewriter:
    def __init__(self, config, llm_ensemble):
        self.interval = config.get('system_prompt_rewrite_interval', 100)  # every N iterations
        self.num_examples = config.get('system_prompt_examples_count', 20)  # programs to analyze

    async def should_rewrite(self, iteration: int) -> bool:
        """Check if we should rewrite the system prompt"""
        return iteration > 0 and iteration % self.interval == 0

    async def rewrite_prompt(
        self,
        current_prompt: str,
        recent_programs: List[Program],
        database: ProgramDatabase
    ) -> str:
        """Analyze recent evolution and generate improved system prompt"""
        # 1. Collect notable examples (best performers, diverse solutions, failures)
        # 2. Compress/summarize the evolution trajectory
        # 3. Construct meta-prompt asking LLM to improve system prompt
        # 4. Return new system prompt
```

**Key methods:**
- `collect_notable_examples()`: Get top performers, diverse programs, common failures
- `compress_evolution_trajectory()`: Summarize what worked/didn't work
- `build_meta_prompt()`: Create prompt for LLM to rewrite system prompt
- `validate_new_prompt()`: Ensure new prompt is valid and different

### 3. **Integration into Evolution Loop** 🔄
**Location:** [process_parallel.py](openevolve/process_parallel.py) `ProcessParallelController.run_evolution()`

**Modification flow:**
```python
async def run_evolution(self, start_iter, max_iter, target_score, checkpoint_callback):
    for iteration in range(start_iter, max_iter):
        # Check if we should rewrite system prompt
        if await self.prompt_rewriter.should_rewrite(iteration):
            # Collect recent programs from database
            recent_programs = self.database.get_recent_programs(
                count=self.prompt_rewriter.num_examples
            )

            # Generate new system prompt
            new_prompt = await self.prompt_rewriter.rewrite_prompt(
                current_prompt=self.config.prompt.system_message,
                recent_programs=recent_programs,
                database=self.database
            )

            # Update system prompt for all workers
            self.update_system_prompt(new_prompt)

            logger.info(f"🔄 Updated system prompt at iteration {iteration}")

        # Continue normal evolution...
```

### 4. **Worker Synchronization** ⚙️
**Challenge:** Workers run in separate processes with their own config copies

**Solutions:**
- **Option A (Simple):** Workers already recreate components - just update config in main process, workers get new prompt on next iteration
- **Option B (Robust):** Pass `current_system_message` as parameter to `_run_iteration_worker()` alongside db_snapshot
- **Recommended:** Option B - modify worker function signature:
  ```python
  def _run_iteration_worker(
      iteration: int,
      db_snapshot: Dict,
      parent_id: str,
      inspiration_ids: List[str],
      system_message_override: Optional[str] = None  # NEW
  )
  ```

### 5. **Configuration Schema Updates** 📋
**File:** [config.py](openevolve/config.py)

Add new config section:
```python
@dataclass
class SystemPromptEvolutionConfig:
    enabled: bool = False
    rewrite_interval: int = 100  # iterations between rewrites
    num_examples: int = 20  # programs to include in meta-prompt
    meta_llm_model: str = None  # use different/better model for meta-evolution
    min_improvement_threshold: float = 0.05  # only update if programs improve
    keep_history: bool = True  # track all system prompt versions
```

### 6. **Checkpointing & Resume** 💾
**Files:** [controller.py](openevolve/controller.py) `_save_checkpoint()` / `_load_checkpoint()`

**Add to checkpoint:**
- Current system prompt text
- System prompt evolution history
- Metrics before/after each rewrite

```python
# In _save_checkpoint():
checkpoint_data["system_prompt_history"] = [
    {
        "iteration": iter,
        "prompt": prompt,
        "avg_score_before": score_before,
        "avg_score_after": score_after
    }
    for iter, prompt, score_before, score_after in self.prompt_history
]
```

### 7. **Meta-Prompt Design** 📝
**New file:** `openevolve/prompt/templates/meta_system_prompt.txt`

Example template:
```
You are a meta-optimizer improving prompts for code evolution systems.

Current system prompt:
---
{current_system_prompt}
---

Recent evolution results ({num_programs} programs from last {interval} iterations):

TOP PERFORMERS (what worked):
{top_programs_summary}

DIVERSE SOLUTIONS (alternative approaches):
{diverse_programs_summary}

COMMON ISSUES (what didn't work):
{failure_patterns}

EVOLUTION STATISTICS:
- Average score improvement: {avg_improvement}
- Best performer score: {best_score}
- Common mutation types: {mutation_types}

Your task: Rewrite the system prompt to guide the LLM toward more effective code mutations.
Focus on patterns that succeeded and avoid patterns that failed.

Output only the new system prompt, nothing else.
```

## Implementation Sequence

1. **Phase 1: Core Infrastructure** (2-3 hours)
   - Add `SystemPromptRewriter` class
   - Update config schema
   - Add checkpoint support for prompt history

2. **Phase 2: Integration** (2-3 hours)
   - Integrate into `ProcessParallelController`
   - Update worker synchronization
   - Test with simple interval-based rewriting

3. **Phase 3: Intelligence** (3-4 hours)
   - Implement smart example selection (best/diverse/failures)
   - Build compression logic for evolution trajectory
   - Design effective meta-prompts
   - Add validation and safety checks

4. **Phase 4: Testing & Refinement** (2-3 hours)
   - Test on blocksworld or simple example
   - Tune rewrite interval and example count
   - Verify checkpoint/resume works correctly

## Key Design Decisions

**Q: How often to rewrite?**
A: Start with 50-100 iterations. Too frequent = instability, too rare = missed opportunities.

**Q: How many examples to include?**
A: 10-20 programs. Balance between context richness and prompt length/cost.

**Q: Which LLM for meta-evolution?**
A: Use same or better model (e.g., if evolving with gpt-4, use gpt-4 or o1 for meta-prompt).

**Q: How to validate new prompts?**
A: Ensure minimum length, check for placeholders, optionally run a few test iterations before committing.

## Files to Create/Modify

**New files:**
- `openevolve/system_prompt_rewriter.py` - Main rewriter logic
- `openevolve/prompt/templates/meta_system_prompt.txt` - Meta-prompt template

**Modified files:**
- `openevolve/config.py` - Add SystemPromptEvolutionConfig
- `openevolve/controller.py` - Add update_system_prompt(), checkpoint prompt history
- `openevolve/process_parallel.py` - Integrate rewriting into evolution loop
- `openevolve/database.py` - Add get_recent_programs() method
- `openevolve/prompt/sampler.py` - Minor updates if needed (already supports overrides!)

## Comparison with Your Initial Strategy

Your initial idea was solid! Here's how this plan builds on it:

✅ **Your idea:** Run for 10-100 iterations, collect outputs, condense into prompt, rewrite system prompt
✅ **This plan:** Implements exactly that with specific architecture decisions

**Enhancements added:**
1. **Worker synchronization** - handles the multi-process architecture
2. **Checkpoint integration** - can resume with evolved prompts
3. **Smart example selection** - not just recent, but top/diverse/failures
4. **Validation layer** - ensures new prompts are actually improvements
5. **Configuration schema** - makes it tunable without code changes

## Alternative: Simpler Prototype

If you want to prototype quickly, you could:

1. **Skip worker synchronization** - just update `self.config.prompt.system_message` in controller
2. **Skip checkpointing** - keep prompt history in memory only
3. **Simple example collection** - just take last N programs sorted by score
4. **Hardcode interval** - rewrite every 50 iterations
5. **Use existing LLM ensemble** - no separate meta-LLM

This would be ~200 lines of code in a single new file that integrates into the controller.

## Expected Impact

Based on AlphaEvolve paper results:
- **Initial runs:** System prompt likely suboptimal for specific task
- **After 1-2 rewrites:** Prompt adapts to task characteristics, sees 10-20% improvement
- **After 3-4 rewrites:** Diminishing returns, but maintains diversity

This is particularly powerful for:
- Long evolution runs (500+ iterations)
- Complex tasks where initial prompt misses key insights
- Multi-objective optimization where prompt needs to balance goals

## Risk Mitigation

1. **Prompt degradation:** Keep history, allow rollback if scores drop
2. **Instability:** Require minimum interval between rewrites (50+ iterations)
3. **Cost:** Meta-prompts can be expensive - make interval configurable
4. **Validation:** Test new prompt on small sample before committing

---

# Evaluation Methodology

## Overview: Leveraging OpenEvolve's Built-in Data Collection

OpenEvolve already has **excellent infrastructure** for measuring evolution performance! You can leverage:

1. **Evolution Tracing** ([evolution_trace.py](openevolve/evolution_trace.py))
   - Logs every iteration with parent/child metrics
   - Tracks improvement deltas automatically
   - Supports JSONL, JSON, and HDF5 formats
   - Already calculates statistics (improvement rate, best/worst changes)

2. **Checkpoint System** ([controller.py](openevolve/controller.py))
   - Saves full database state every N iterations
   - Includes `best_program_info.json` with metrics
   - Can extract evolution traces post-hoc from checkpoints

3. **Per-Program Artifacts** ([database.py](openevolve/database.py))
   - Stores evaluation details (errors, timing, etc.)
   - Optional prompt logging (`database.log_prompts: true`)
   - Artifacts can include evaluation feedback

4. **Database Statistics**
   - Tracks best program across all islands
   - MAP-Elites coverage metrics
   - Island-specific performance

**Bottom line:** Most of the data you need is already being collected or can be enabled with simple config changes!

---

## Core Metrics to Collect

### 1. Evolution Performance Metrics (PRIMARY)

These directly answer: "Does meta-prompting help find better solutions faster?"

**Metrics:**
- **Final best score** - `combined_score` at end of run
- **Convergence speed** - Iterations to reach score thresholds (0.8, 0.9, 0.95)
- **Best score at checkpoints** - Score at iterations 10, 20, 50, 100
- **Improvement rate** - Percentage of iterations that improve best score
- **Average improvement per iteration** - Mean delta in `combined_score`

**Where stored:**
```
output_dir/
├── evolution_trace.jsonl          # Real-time iteration logs
├── checkpoints/
│   └── checkpoint_N/
│       └── best_program_info.json # Best at checkpoint N
└── best/
    └── best_program_info.json     # Final best program
```

**How to extract:**
```python
# Load evolution trace
import json
traces = []
with open("evolution_trace.jsonl") as f:
    for line in f:
        traces.append(json.loads(line))

# Get final best score
final_score = max(t['child_metrics']['combined_score'] for t in traces)

# Find iterations to threshold
for threshold in [0.8, 0.9, 0.95]:
    iters = [t['iteration'] for t in traces
             if t['child_metrics']['combined_score'] >= threshold]
    first_iter = min(iters) if iters else None
```

---

### 2. System Prompt Evolution History (NEW - YOU'LL ADD THIS)

This answers: "How did the system prompt change and did those changes correlate with improvements?"

**Metrics:**
- **Number of rewrites** - How many times prompt was updated
- **Rewrite timing** - Which iterations triggered rewrites
- **Before/after scores** - Avg score 10 iterations before vs after each rewrite
- **Prompt content** - Actual prompt text for qualitative analysis
- **Trigger reasons** - Why rewrite happened (scheduled interval, convergence detection, etc.)

**Data structure to save:**
```json
{
  "system_prompt_history": [
    {
      "iteration": 50,
      "timestamp": 1234567890.0,
      "old_prompt": "You are an expert programmer...",
      "new_prompt": "You are a world-class algorithm designer...",
      "avg_score_before": 0.65,
      "avg_score_after": 0.72,
      "improvement": 0.07,
      "num_programs_analyzed": 15,
      "trigger_reason": "scheduled_interval",
      "meta_prompt_used": "meta_system_prompt",
      "notable_changes": [
        "Added focus on edge cases",
        "Emphasized efficiency over readability"
      ]
    },
    {
      "iteration": 100,
      "timestamp": 1234567990.0,
      "old_prompt": "You are a world-class algorithm designer...",
      "new_prompt": "You are an expert in optimization...",
      "avg_score_before": 0.72,
      "avg_score_after": 0.78,
      "improvement": 0.06,
      "num_programs_analyzed": 15,
      "trigger_reason": "scheduled_interval"
    }
  ],
  "summary": {
    "total_rewrites": 2,
    "total_improvement": 0.13,
    "avg_improvement_per_rewrite": 0.065
  }
}
```

**Save locations:**
- Each checkpoint: `checkpoints/checkpoint_N/system_prompt_history.json`
- Final output: `output_dir/system_prompt_evolution.json`

---

### 3. Search Efficiency Metrics (SECONDARY)

These answer: "Did meta-prompting improve exploration of the solution space?"

**Metrics:**
- **MAP-Elites coverage** - Percentage of feature grid cells filled
- **Island diversity** - Distribution of programs across islands
- **Generation depth** - Average/max generations of successful programs
- **Evaluation time** - Time per iteration (LLM + evaluation)
- **Code diversity** - Edit distance between top programs

**Where stored:**
- Database: `island_feature_maps` (coverage calculation)
- Evolution trace: `generation` field (lineage depth)
- Logs: Timing information from controller

**How to extract:**
```python
# MAP-Elites coverage from checkpoint
import pickle
with open("checkpoint_100/database.pkl", "rb") as f:
    db = pickle.load(f)

total_cells = db.feature_bins ** len(db.config.feature_dimensions)
filled_cells = sum(len(island_map) for island_map in db.island_feature_maps)
coverage = filled_cells / (total_cells * db.config.num_islands)

# Generation depth from traces
avg_generation = sum(t['generation'] for t in traces) / len(traces)
max_generation = max(t['generation'] for t in traces)
```

---

### 4. Qualitative Analysis Data (TERTIARY)

For understanding *why* meta-prompting worked (or didn't).

**Data to collect:**
- **Best program code** at each prompt rewrite checkpoint
- **LLM responses** that led to breakthroughs (high improvement deltas)
- **Failure patterns** - Common errors from artifacts
- **Mutation strategies** - What kinds of changes were successful

**Where stored:**
```
checkpoints/
└── checkpoint_N/
    ├── best_program.py                      # Code snapshot
    ├── system_prompt_history.json           # Prompt at this point
    └── programs/
        └── {program_id}.json                # Full program with prompts/artifacts
```

**How to extract:**
```python
# Get best program code at each rewrite point
rewrite_iterations = [50, 100, 150]
for iter in rewrite_iterations:
    with open(f"checkpoints/checkpoint_{iter}/best_program.py") as f:
        code = f.read()
    # Analyze code structure, complexity, etc.

# Find breakthrough moments (large improvements)
breakthroughs = [t for t in traces
                 if t['improvement_delta'].get('combined_score', 0) > 0.05]
```

---

## Experimental Design

### Baseline Condition (Control)

**Goal:** Measure performance WITHOUT meta-prompting

**Configuration:**
```yaml
# Add/modify in config.yaml for baseline runs
system_prompt_evolution:
  enabled: false  # Disable meta-prompting

evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: false
  include_prompts: true

database:
  log_prompts: true  # Save prompts for analysis
```

**Run parameters:**
- **Examples:** function_minimization, blocksworld, llm_prompt_optimization
- **Iterations:** 100-200 per run (enough to see convergence)
- **Replicates:** 3-5 runs per example (for statistical power)
- **Random seeds:** Use different seeds for each replicate

**Expected runtime:**
- function_minimization: ~5-10 min per run → 30-50 min total
- blocksworld: ~20-30 min per run → 1.5-2.5 hours total
- llm_prompt_optimization: ~1-2 hours per run → 5-10 hours total

---

### Treatment Condition (Meta-Prompting)

**Goal:** Measure performance WITH meta-prompting enabled

**Configuration:**
```yaml
# Meta-prompting enabled configuration
system_prompt_evolution:
  enabled: true
  rewrite_interval: 50      # Rewrite every 50 iterations
  num_examples: 15          # Use 15 programs for meta-prompt
  min_improvement_threshold: 0.0  # Always try rewrite (no filtering)
  keep_history: true
  meta_llm_model: null      # Use same model as evolution

evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: false
  include_prompts: true

database:
  log_prompts: true
```

**Run parameters:**
- Same examples, iterations, and replicates as baseline
- Use SAME random seeds as baseline for paired comparison (if possible)

**Additional data:**
- System prompt evolution history (saved automatically)
- Timing of rewrites
- Before/after rewrite metrics

---

### Which Examples to Test

**Priority 1 (MUST TEST):**
1. **function_minimization**
   - Fast iterations (~30 sec each)
   - Clear optimization objective
   - Easy to interpret results
   - Good for debugging

**Priority 2 (SHOULD TEST):**
2. **blocksworld**
   - Medium complexity
   - Interesting search space
   - Your recent work - good for demo

**Priority 3 (NICE TO HAVE):**
3. **llm_prompt_optimization**
   - Ironic meta-meta-optimization!
   - Longer runtime but compelling results

**Skip unless time permits:**
- attention_optimization (hardware-specific)
- rust_adaptive_sort (language complexity)
- web_scraper_optillm (external dependencies)

---

## Data Storage Structure

Organize all experimental runs in a dedicated directory:

```
evaluation_results/
├── baseline/                           # Control condition
│   ├── function_minimization/
│   │   ├── run_1_seed_42/
│   │   │   ├── evolution_trace.jsonl
│   │   │   ├── checkpoints/
│   │   │   │   ├── checkpoint_50/
│   │   │   │   ├── checkpoint_100/
│   │   │   │   └── ...
│   │   │   ├── best/
│   │   │   │   └── best_program_info.json
│   │   │   └── logs/
│   │   ├── run_2_seed_123/
│   │   ├── run_3_seed_456/
│   │   └── summary.json                # Aggregate metrics
│   ├── blocksworld/
│   │   ├── run_1_seed_42/
│   │   └── ...
│   └── llm_prompt_optimization/
│
├── meta_prompt/                        # Treatment condition
│   ├── function_minimization/
│   │   ├── run_1_seed_42/
│   │   │   ├── evolution_trace.jsonl
│   │   │   ├── system_prompt_evolution.json  # NEW!
│   │   │   ├── checkpoints/
│   │   │   │   ├── checkpoint_50/
│   │   │   │   │   ├── system_prompt_history.json
│   │   │   │   │   └── ...
│   │   │   ├── best/
│   │   │   └── logs/
│   │   ├── run_2_seed_123/
│   │   └── run_3_seed_456/
│   ├── blocksworld/
│   └── llm_prompt_optimization/
│
├── analysis/                           # Analysis outputs
│   ├── comparison_results.json         # Statistical tests
│   ├── plots/
│   │   ├── function_minimization_comparison.png
│   │   ├── blocksworld_learning_curves.png
│   │   └── system_prompt_impact.png
│   ├── statistical_tests.csv
│   └── report.md                       # Summary
│
└── README.md                           # Experiment documentation
```

---

## Analysis Framework

### Python Script: `scripts/evaluate_meta_prompting.py`

```python
"""
Evaluate the impact of system prompt evolution on OpenEvolve performance
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any

def load_evolution_trace(trace_path: Path) -> pd.DataFrame:
    """Load evolution trace from JSONL file"""
    traces = []
    with open(trace_path, 'r') as f:
        for line in f:
            traces.append(json.loads(line))

    df = pd.DataFrame(traces)

    # Extract metrics from nested dicts
    if 'child_metrics' in df.columns:
        df['combined_score'] = df['child_metrics'].apply(
            lambda x: x.get('combined_score', 0) if isinstance(x, dict) else 0
        )

    if 'improvement_delta' in df.columns:
        df['score_improvement'] = df['improvement_delta'].apply(
            lambda x: x.get('combined_score', 0) if isinstance(x, dict) else 0
        )

    # Calculate cumulative best
    df['best_score_so_far'] = df['combined_score'].cummax()

    return df

def extract_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract key metrics from a single run directory"""
    metrics = {
        'run_path': str(run_dir),
        'example_name': run_dir.parent.name,
        'condition': run_dir.parent.parent.name,  # baseline or meta_prompt
    }

    # Load evolution trace
    trace_path = run_dir / "evolution_trace.jsonl"
    if not trace_path.exists():
        print(f"Warning: No evolution trace found in {run_dir}")
        return None

    df = load_evolution_trace(trace_path)

    # Core performance metrics
    metrics['final_best_score'] = df['best_score_so_far'].iloc[-1]
    metrics['total_iterations'] = len(df)
    metrics['improvement_rate'] = (df['score_improvement'] > 0).mean()
    metrics['avg_improvement_per_iter'] = df['score_improvement'].mean()

    # Convergence metrics
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        reached = df[df['best_score_so_far'] >= threshold]
        if len(reached) > 0:
            metrics[f'iters_to_{int(threshold*100)}pct'] = reached['iteration'].iloc[0]
        else:
            metrics[f'iters_to_{int(threshold*100)}pct'] = None

    # Load final best program info
    best_info_path = run_dir / "best" / "best_program_info.json"
    if best_info_path.exists():
        with open(best_info_path) as f:
            best_info = json.load(f)
            metrics['final_iteration'] = best_info.get('current_iteration', len(df))
            metrics['best_generation'] = best_info.get('generation', 0)

    # Load system prompt evolution history (if meta-prompting)
    prompt_history_path = run_dir / "system_prompt_evolution.json"
    if prompt_history_path.exists():
        with open(prompt_history_path) as f:
            prompt_history = json.load(f)
            history = prompt_history.get('system_prompt_history', [])
            metrics['num_prompt_rewrites'] = len(history)
            metrics['total_prompt_improvement'] = sum(
                h.get('improvement', 0) for h in history
            )
    else:
        metrics['num_prompt_rewrites'] = 0
        metrics['total_prompt_improvement'] = 0

    # Store full trace for plotting
    metrics['trace_df'] = df

    return metrics

def compare_conditions(
    baseline_runs: List[Dict],
    treatment_runs: List[Dict]
) -> Dict[str, Any]:
    """Statistical comparison between baseline and treatment conditions"""

    results = {}

    # Extract scores
    baseline_scores = [r['final_best_score'] for r in baseline_runs]
    treatment_scores = [r['final_best_score'] for r in treatment_runs]

    # T-test
    t_stat, p_value = stats.ttest_ind(treatment_scores, baseline_scores)

    # Effect size (Cohen's d)
    mean_diff = np.mean(treatment_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt(
        (np.var(baseline_scores) + np.var(treatment_scores)) / 2
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Basic statistics
    results['baseline'] = {
        'mean': np.mean(baseline_scores),
        'std': np.std(baseline_scores),
        'min': np.min(baseline_scores),
        'max': np.max(baseline_scores),
        'n': len(baseline_scores)
    }

    results['treatment'] = {
        'mean': np.mean(treatment_scores),
        'std': np.std(treatment_scores),
        'min': np.min(treatment_scores),
        'max': np.max(treatment_scores),
        'n': len(treatment_scores)
    }

    # Comparison
    results['comparison'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_improvement': mean_diff,
        'pct_improvement': (mean_diff / results['baseline']['mean'] * 100)
                          if results['baseline']['mean'] > 0 else 0,
        'significant': p_value < 0.05
    }

    # Convergence speed comparison
    for threshold in [70, 80, 90, 95]:
        key = f'iters_to_{threshold}pct'
        baseline_iters = [r[key] for r in baseline_runs if r[key] is not None]
        treatment_iters = [r[key] for r in treatment_runs if r[key] is not None]

        if baseline_iters and treatment_iters:
            speedup = (np.mean(baseline_iters) - np.mean(treatment_iters))
            results['comparison'][f'speedup_{threshold}pct'] = speedup

    return results

def plot_learning_curves(
    baseline_runs: List[Dict],
    treatment_runs: List[Dict],
    example_name: str,
    output_path: Path
):
    """Plot evolution curves comparing baseline vs treatment"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Baseline runs
    ax = axes[0]
    for i, run in enumerate(baseline_runs):
        df = run['trace_df']
        ax.plot(df['iteration'], df['best_score_so_far'],
               alpha=0.5, label=f"Run {i+1}", linewidth=2)

    ax.set_title(f"{example_name} - Baseline (No Meta-Prompting)",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Score", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Treatment runs
    ax = axes[1]
    for i, run in enumerate(treatment_runs):
        df = run['trace_df']
        ax.plot(df['iteration'], df['best_score_so_far'],
               alpha=0.5, label=f"Run {i+1}", linewidth=2)

        # Mark prompt rewrite points
        if run['num_prompt_rewrites'] > 0:
            # Load prompt history to get exact iterations
            # For now, estimate based on interval (50)
            rewrite_interval = 50
            for j in range(1, run['num_prompt_rewrites'] + 1):
                rewrite_iter = j * rewrite_interval
                if rewrite_iter <= df['iteration'].max():
                    ax.axvline(rewrite_iter, color='red',
                              linestyle='--', alpha=0.4, linewidth=1)

    # Add one red line to legend
    ax.axvline(-1, color='red', linestyle='--',
              alpha=0.4, label='Prompt Rewrite', linewidth=2)

    ax.set_title(f"{example_name} - Meta-Prompting Enabled",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def generate_report(results: Dict[str, Any], output_path: Path):
    """Generate markdown report of evaluation results"""

    report = ["# Meta-Prompting Evaluation Report\n"]

    for example_name, comparison in results.items():
        report.append(f"## {example_name}\n")

        baseline = comparison['baseline']
        treatment = comparison['treatment']
        comp = comparison['comparison']

        report.append("### Performance Summary\n")
        report.append(f"- **Baseline Mean Score:** {baseline['mean']:.4f} ± {baseline['std']:.4f}")
        report.append(f"- **Treatment Mean Score:** {treatment['mean']:.4f} ± {treatment['std']:.4f}")
        report.append(f"- **Improvement:** {comp['mean_improvement']:.4f} ({comp['pct_improvement']:.1f}%)")
        report.append(f"- **Statistical Significance:** {'Yes ✅' if comp['significant'] else 'No ❌'} (p={comp['p_value']:.4f})")
        report.append(f"- **Effect Size (Cohen's d):** {comp['cohens_d']:.3f}\n")

        if any(f'speedup_{t}pct' in comp for t in [70, 80, 90, 95]):
            report.append("### Convergence Speed\n")
            for threshold in [70, 80, 90, 95]:
                key = f'speedup_{threshold}pct'
                if key in comp:
                    speedup = comp[key]
                    report.append(f"- **To {threshold}% optimal:** {speedup:.1f} iterations faster")
            report.append("\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Generated report: {output_path}")

def main():
    """Main evaluation pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate meta-prompting impact')
    parser.add_argument('--baseline', type=Path, required=True,
                       help='Path to baseline results directory')
    parser.add_argument('--treatment', type=Path, required=True,
                       help='Path to treatment results directory')
    parser.add_argument('--output', type=Path, default=Path('analysis'),
                       help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(exist_ok=True)
    plots_dir = args.output / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Find all example directories
    baseline_examples = [d for d in args.baseline.iterdir() if d.is_dir()]
    treatment_examples = [d for d in args.treatment.iterdir() if d.is_dir()]

    example_names = set([e.name for e in baseline_examples + treatment_examples])

    all_results = {}

    for example_name in example_names:
        print(f"\n{'='*60}")
        print(f"Analyzing: {example_name}")
        print('='*60)

        # Load baseline runs
        baseline_dir = args.baseline / example_name
        baseline_runs = []
        if baseline_dir.exists():
            for run_dir in sorted(baseline_dir.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith('run_'):
                    metrics = extract_run_metrics(run_dir)
                    if metrics:
                        baseline_runs.append(metrics)

        # Load treatment runs
        treatment_dir = args.treatment / example_name
        treatment_runs = []
        if treatment_dir.exists():
            for run_dir in sorted(treatment_dir.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith('run_'):
                    metrics = extract_run_metrics(run_dir)
                    if metrics:
                        treatment_runs.append(metrics)

        if not baseline_runs or not treatment_runs:
            print(f"⚠️  Insufficient data for {example_name}")
            continue

        print(f"Loaded {len(baseline_runs)} baseline runs, {len(treatment_runs)} treatment runs")

        # Compare conditions
        comparison = compare_conditions(baseline_runs, treatment_runs)
        all_results[example_name] = comparison

        # Print summary
        comp = comparison['comparison']
        print(f"\n📊 Results:")
        print(f"   Baseline:  {comparison['baseline']['mean']:.4f} ± {comparison['baseline']['std']:.4f}")
        print(f"   Treatment: {comparison['treatment']['mean']:.4f} ± {comparison['treatment']['std']:.4f}")
        print(f"   Improvement: {comp['pct_improvement']:+.1f}% (p={comp['p_value']:.4f})")

        # Plot learning curves
        plot_path = plots_dir / f"{example_name}_comparison.png"
        plot_learning_curves(baseline_runs, treatment_runs, example_name, plot_path)

    # Save results
    results_path = args.output / 'comparison_results.json'
    # Remove trace_df before saving (not JSON serializable)
    for example, data in all_results.items():
        for cond in ['baseline_runs', 'treatment_runs']:
            if cond in data:
                for run in data[cond]:
                    run.pop('trace_df', None)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Saved results to {results_path}")

    # Generate report
    report_path = args.output / 'report.md'
    generate_report(all_results, report_path)

    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
```

---

## Quick Start: Running Experiments

### Step 1: Enable Evolution Tracing

For ALL examples being tested, add to their `config.yaml`:

```yaml
evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: false
  include_prompts: true

database:
  log_prompts: true
```

### Step 2: Run Baseline Experiments

```bash
# Create baseline results directory
mkdir -p evaluation_results/baseline

# Example: function_minimization (3 runs with different seeds)
for seed in 42 123 456; do
    python openevolve-run.py \
        examples/function_minimization/initial_program.py \
        examples/function_minimization/evaluator.py \
        --config examples/function_minimization/config.yaml \
        --iterations 100 \
        --output-dir evaluation_results/baseline/function_minimization/run_${seed}_seed_${seed}
done

# Example: blocksworld (3 runs)
for seed in 42 123 456; do
    python openevolve-run.py \
        examples/blocksworld/blocksworld_planner.py \
        examples/blocksworld/blocksworld_evaluator.py \
        --config examples/blocksworld/config.yaml \
        --iterations 100 \
        --output-dir evaluation_results/baseline/blocksworld/run_${seed}_seed_${seed}
done
```

### Step 3: Create Meta-Prompting Config

Create `examples/function_minimization/config_meta.yaml`:

```yaml
# Inherit from base config, add meta-prompting
<<: *base_config  # Or copy entire base config

system_prompt_evolution:
  enabled: true
  rewrite_interval: 50
  num_examples: 15
  min_improvement_threshold: 0.0
  keep_history: true

evolution_trace:
  enabled: true
  format: "jsonl"
  include_code: false
  include_prompts: true

database:
  log_prompts: true
```

### Step 4: Run Treatment Experiments

```bash
# Create treatment results directory
mkdir -p evaluation_results/meta_prompt

# Run with meta-prompting (same seeds for paired comparison)
for seed in 42 123 456; do
    python openevolve-run.py \
        examples/function_minimization/initial_program.py \
        examples/function_minimization/evaluator.py \
        --config examples/function_minimization/config_meta.yaml \
        --iterations 100 \
        --output-dir evaluation_results/meta_prompt/function_minimization/run_${seed}_seed_${seed}
done
```

### Step 5: Analyze Results

```bash
python scripts/evaluate_meta_prompting.py \
    --baseline evaluation_results/baseline \
    --treatment evaluation_results/meta_prompt \
    --output analysis/
```

This will generate:
- `analysis/comparison_results.json` - Statistical test results
- `analysis/plots/function_minimization_comparison.png` - Learning curves
- `analysis/report.md` - Summary report for presentation

---

## Key Metrics for Presentation

Focus on these headline numbers:

### 1. Primary Success Metrics

**Performance Improvement:**
```
"Meta-prompting improved final scores by X% on average across N examples"
```

**Convergence Speed:**
```
"Reached 90% optimal score Y iterations faster (Z% speedup)"
```

**Consistency:**
```
"Reduced variance across runs by W%"
```

### 2. Supporting Evidence

**System Prompt Evolution:**
```
"System prompts evolved to emphasize [specific patterns]"
"After K rewrites, prompts adapted from generic → task-specific"
```

**Search Efficiency:**
```
"Explored A% more of the feature space"
"Achieved B% higher MAP-Elites coverage"
```

**Cost-Benefit:**
```
"Added C% more LLM calls but achieved D% better results"
"ROI: E% improvement per additional LLM query"
```

### 3. Qualitative Insights

**Show prompt evolution:**
```
Initial:  "You are an expert programmer..."
Iteration 50: "You are an optimization specialist focusing on edge cases..."
Iteration 100: "You are a performance engineer who..."
```

**Best program comparison:**
- Code quality metrics (complexity, readability)
- Novel algorithmic approaches discovered
- Performance characteristics

**Failure mode analysis:**
```
"Meta-prompting helped most on tasks with [X characteristic]"
"Less effective when [Y condition]"
```

---

## Expected Challenges & Solutions

### Challenge 1: High Variance in Evolution

**Problem:** Evolutionary algorithms are stochastic; single runs may not be representative

**Solutions:**
- Run 3-5 replicates per condition
- Use paired comparison (same seeds for baseline/treatment)
- Report effect sizes, not just p-values
- Use non-parametric tests (Mann-Whitney U) if distributions are skewed

### Challenge 2: Different Examples Respond Differently

**Problem:** Meta-prompting might help some tasks more than others

**Solutions:**
- Test multiple examples with different characteristics
- Report per-example and aggregate results
- Identify task characteristics that predict success
- Don't cherry-pick - show all results

### Challenge 3: Attribution is Unclear

**Problem:** Hard to tell if improvements are from meta-prompting specifically

**Solutions:**
- Ablation study: Test different rewrite intervals (25, 50, 100)
- Compare prompt rewrite timing with score jumps
- Analyze if improvements correlate with rewrites
- Control for total compute (same number of LLM calls)

### Challenge 4: Runtime and Cost

**Problem:** Experiments take time and API calls cost money

**Solutions:**
- Start with fastest example (function_minimization)
- Use cheaper models for development (Gemini Flash)
- Cache LLM responses where possible
- Run treatment and baseline in parallel

---

## Minimal Viable Evaluation (Time-Constrained Version)

If you have limited time before the interview, here's the **minimum** evaluation that would still be convincing:

### 1. Test Single Example: `function_minimization`

**Why:**
- Fastest iterations (~30 seconds each)
- Clear optimization objective
- Easy to visualize and explain
- Low API cost

### 2. Run 3 Baseline + 3 Treatment (100 iterations each)

**Total runtime:** ~2 hours
**Total iterations:** 600
**Cost:** ~$5-10 in API calls (depending on model)

### 3. Report These 3 Metrics

**Metric 1: Final Score Comparison**
```
Baseline:    0.845 ± 0.032
Treatment:   0.912 ± 0.018
Improvement: +7.9% (p=0.031) ✅
```

**Metric 2: Convergence Speed**
```
Iterations to 90% optimal:
Baseline:  78 ± 12 iterations
Treatment: 52 ± 8 iterations
Speedup:   33% faster ✅
```

**Metric 3: Visual Evidence**
- Learning curve plot showing faster convergence
- Mark prompt rewrite points on treatment curve
- Show one example of prompt evolution

### 4. Time Estimate

- Setup: 30 minutes
- Baseline runs: 45 minutes (3 × 15 min)
- Treatment runs: 45 minutes (3 × 15 min)
- Analysis: 30 minutes
- **Total: 2.5 hours**

This would be sufficient to demonstrate:
1. The concept works (positive results)
2. You can measure it rigorously (statistics)
3. You understand the tradeoffs (cost/benefit)

---

## Bonus: Post-Hoc Analysis Tools

OpenEvolve includes utilities for extracting data from existing checkpoints:

### Extract Evolution Traces from Checkpoints

```python
from openevolve.evolution_trace import extract_evolution_trace_from_checkpoint

# If you forgot to enable evolution tracing, you can extract it retroactively!
traces = extract_evolution_trace_from_checkpoint(
    checkpoint_dir="evaluation_results/baseline/function_minimization/run_1/checkpoints/checkpoint_100",
    output_path="analysis/baseline_run1_trace.jsonl",
    format="jsonl",
    include_code=True,
    include_prompts=True
)

print(f"Extracted {len(traces)} evolution traces")
```

### Extract Full Lineage Chains

```python
from openevolve.evolution_trace import extract_full_lineage_traces

# Get complete parent-child chains for all programs
lineages = extract_full_lineage_traces(
    checkpoint_dir="evaluation_results/meta_prompt/function_minimization/run_1/checkpoints/checkpoint_100",
    output_path="analysis/meta_run1_lineages.json",
    format="json"
)

# Each lineage shows the complete evolution path of a program
for lineage in lineages[:3]:  # Show top 3 most evolved programs
    print(f"\nProgram {lineage['final_program_id']}:")
    print(f"  Generations: {lineage['generation_depth']}")
    print(f"  Final score: {lineage['final_metrics']['combined_score']:.4f}")
    print(f"  Evolution steps: {len(lineage['improvement_steps'])}")

    # Show each improvement step
    for step in lineage['improvement_steps']:
        improvement = step['improvement']['combined_score']
        print(f"    Step {step['step']}: {improvement:+.4f}")
```

### Analyze Prompt Impact

```python
# Load system prompt evolution history
with open("evaluation_results/meta_prompt/function_minimization/run_1/system_prompt_evolution.json") as f:
    prompt_data = json.load(f)

for i, rewrite in enumerate(prompt_data['system_prompt_history']):
    print(f"\nRewrite #{i+1} at iteration {rewrite['iteration']}:")
    print(f"  Score before: {rewrite['avg_score_before']:.4f}")
    print(f"  Score after:  {rewrite['avg_score_after']:.4f}")
    print(f"  Improvement:  {rewrite['improvement']:+.4f}")
    print(f"  Prompt diff preview:")

    # Show what changed (simple diff)
    old_words = set(rewrite['old_prompt'].split())
    new_words = set(rewrite['new_prompt'].split())
    added = new_words - old_words
    removed = old_words - new_words

    if added:
        print(f"    Added: {', '.join(list(added)[:10])}")
    if removed:
        print(f"    Removed: {', '.join(list(removed)[:10])}")
```

These tools are already implemented and ready to use - no additional coding required!
