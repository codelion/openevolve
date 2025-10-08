# Global Learnings System - Implementation Summary

## Overview
Successfully implemented a Global Learnings System for OpenEvolve that aggregates insights across all iterations and islands to help the LLM avoid repeated mistakes and learn from successful patterns.

## What Was Implemented

### 1. Configuration (`openevolve/config.py`)
Added `GlobalLearningsConfig` dataclass with comprehensive configuration options:
- **Enable/disable**: `enabled` flag
- **Tracking modes**: `track_failures`, `track_successes`, `track_both`
- **Aggregation**: `window_size`, `max_learnings`
- **Thresholds**: `min_failure_count`, `min_success_count`, `min_improvement_threshold`
- **Error types**: `include_syntax_errors`, `include_runtime_errors`, `include_performance_regressions`
- **Injection points**: `inject_in_system_prompt`, `inject_in_user_prompt`
- **Verbosity levels**: `"minimal"`, `"concise"`, `"detailed"`

### 2. Core Logic (`openevolve/global_learnings.py`)
Created the `GlobalLearnings` class with:

#### Data Structures
- `FailurePattern`: Tracks syntax errors, runtime errors, performance regressions
- `SuccessPattern`: Tracks successful changes with improvement metrics

#### Key Methods
- `update_from_iteration()`: Updates learnings from iteration results
- `_extract_syntax_errors()`: Parses stderr for Python syntax errors
- `_extract_runtime_errors()`: Detects IndexError, KeyError, TypeError, etc.
- `_detect_performance_regressions()`: Identifies >10% metric decreases
- `get_top_failures()`: Returns most frequent failure patterns
- `get_top_successes()`: Returns most impactful success patterns
- `generate_prompt_section()`: Formats learnings for LLM prompt injection
- `save()`/`load()`: Checkpoint persistence

#### Pattern Detection
Uses regex patterns to extract common errors:
```python
# Example patterns
"NameError: name 'temp' is not defined" → "Undefined variable: temp"
"IndexError: list index out of range" → "IndexError: list index out of range"
```

### 3. Prompt Integration (`openevolve/prompt/sampler.py`)
Updated `PromptSampler.build_prompt()`:
- Added `global_learnings` parameter
- Injects learnings into system message when provided
- Learnings appear before the main prompt content

### 4. Controller Integration (`openevolve/controller.py`)
Updated `OpenEvolve` class:
- Initializes `GlobalLearnings` instance
- Passes to `ProcessParallelController`
- Saves/loads learnings in checkpoints

### 5. Parallel Processing (`openevolve/process_parallel.py`)
Updated `ProcessParallelController`:
- Accepts `global_learnings` parameter
- Includes learnings section in database snapshots
- Workers use learnings in prompt generation
- Updates learnings from iteration results
- Creates `Result` objects for compatibility

### 6. Checkpoint Support
Learnings are saved to `checkpoint_N/global_learnings.json`:
```json
{
  "failure_patterns": {
    "syntax:Undefined variable: x": {
      "pattern_type": "syntax",
      "description": "Undefined variable: x",
      "count": 5,
      "first_seen": 10,
      "last_seen": 45
    }
  },
  "success_patterns": {...},
  "iteration_history": [1, 2, 3, ...],
  "last_update_iteration": 50
}
```

## Usage

### Basic Configuration (Failure-Only Mode)
```yaml
global_learnings:
  enabled: true
  track_failures: true
  track_successes: false
  window_size: 50
  max_learnings: 5
  inject_in_system_prompt: true
  verbosity: "concise"
```

### Example Prompt Injection
When enabled, the system message will include:
```
## Evolution Insights (Global Learnings)

### Common Pitfalls:
❌ Undefined variable: temp (seen 5x)
❌ IndexError: list index out of range (seen 3x)
⚠️ score decreased by 15.2% (0.850 → 0.721) (seen 4x)
```

### Both Modes (Failures + Successes)
```yaml
global_learnings:
  enabled: true
  track_both: true  # Overrides individual flags
  track_successes: true
  min_success_count: 3
  min_improvement_threshold: 0.05
  verbosity: "detailed"
```

### Example with Successes
```
## Evolution Insights (Global Learnings)

### Common Pitfalls:
❌ Syntax error: Invalid syntax (seen 4x)

### Successful Patterns:
✅ Vectorized loop using numpy (seen 3x, avg improvement: +12.5%)
✅ Cached intermediate results (seen 4x, avg improvement: +8.3%)
```

## Benefits

### Failure-Only Mode (Recommended)
- Helps LLM avoid repeating mistakes across all islands
- Reduces wasted evaluations on known-bad patterns
- Faster convergence by learning from collective errors
- Lower token usage than success tracking
- Particularly useful for syntax/runtime errors

### Success Tracking Mode
- Highlights patterns that consistently improve performance
- May guide mutations toward successful strategies
- More comprehensive but higher token cost
- Best for longer evolution runs (>500 iterations)

## Architecture Notes

### Cross-Island Learning
- Aggregates learnings from ALL islands (not island-specific)
- Provides global view of what works/doesn't work
- Complements island-local context (top programs, recent attempts)

### Performance Regression Detection
- Compares child vs. parent metrics
- Flags >10% decreases as regressions
- Tracks which changes led to performance loss

### Pattern Grouping
- Normalizes similar errors (e.g., all NameErrors)
- Uses string matching for grouping
- Counts occurrences across iterations

### Update Frequency
- `update_interval` controls when learnings refresh
- Default: every 10 iterations
- Balances freshness vs. stability

## File Changes Summary

1. **openevolve/config.py**: Added `GlobalLearningsConfig` class and integrated into master `Config`
2. **openevolve/global_learnings.py**: New file with complete learnings system
3. **openevolve/prompt/sampler.py**: Added `global_learnings` parameter to `build_prompt()`
4. **openevolve/controller.py**: Initialize, save/load global learnings
5. **openevolve/process_parallel.py**: Pass learnings to workers, update from results
6. **examples/config_with_global_learnings.yaml**: Example configuration file

## Testing Recommendations

1. **Unit Tests**: Test pattern extraction, aggregation, formatting
2. **Integration Test**: Run small evolution (50 iterations) with failures enabled
3. **Checkpoint Test**: Verify save/load preserves learnings state
4. **Prompt Test**: Verify learnings appear in generated prompts

## Future Enhancements

Potential improvements (not implemented):
- LLM-based pattern summarization for more insightful descriptions
- Temporal weighting (recent errors weighted higher)
- Per-island learnings (in addition to global)
- Custom pattern extractors for specific languages
- Learning decay (old patterns fade over time)
- Positive reinforcement signals from fitness improvements

## Configuration Examples

See `examples/config_with_global_learnings.yaml` for a complete working example.

### Minimal Configuration
```yaml
global_learnings:
  enabled: true
```
Uses all defaults (failure-only, window_size=50, max_learnings=5)

### Aggressive Tracking
```yaml
global_learnings:
  enabled: true
  track_both: true
  window_size: 100
  max_learnings: 8
  min_failure_count: 2
  verbosity: "detailed"
```

### Disabled (Default)
```yaml
global_learnings:
  enabled: false
```
No overhead when disabled.
