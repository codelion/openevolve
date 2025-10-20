# Huawei 2017 CDN - Scoring System Guide

## Overview

This document explains the scoring system used to evaluate solutions for the Huawei CodeCraft 2017 CDN optimization problem.

## Scoring Formula

```
combined_score = 0.40 × success_rate + 0.35 × cost_score + 0.25 × time_score
```

### Component Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| **Success Rate** | 40% | Percentage of test cases solved correctly |
| **Cost Score** | 35% | How low the total cost is (primary competition metric) |
| **Time Score** | 25% | How fast the solution executes (CRITICAL!) |

## Detailed Score Calculations

### 1. Success Rate
```python
success_rate = valid_solutions / total_test_cases
```
- Range: [0.0, 1.0]
- Each failed test case reduces this proportionally

### 2. Cost Score
```python
cost_score = 10000.0 / (avg_cost + 1.0)
```
- Normalized using a reference cost of 10,000
- Lower costs yield higher scores
- This is the PRIMARY competition ranking metric

### 3. Time Score (NEW - Exponential Decay)
```python
time_score = exp(-avg_time / 10.0)

# Bonus for very fast solutions (< 1s)
if avg_time < 1.0:
    time_score = min(1.0, time_score × 1.2)
```

**Time Score Table:**

| Avg Time | Time Score | Impact (×0.25) | Assessment |
|----------|------------|----------------|------------|
| 0.5s     | 1.0000     | 0.2500         | Excellent  |
| 1.0s     | 0.9048     | 0.2262         | Excellent  |
| 2.0s     | 0.8187     | 0.2047         | Excellent  |
| 3.0s     | 0.7408     | 0.1852         | Excellent  |
| **5.0s** | **0.6065** | **0.1516**     | **Good**   |
| 7.0s     | 0.4966     | 0.1241         | Good       |
| **10.0s**| **0.3679** | **0.0920**     | Acceptable |
| 15.0s    | 0.2231     | 0.0558         | Acceptable |
| **20.0s**| **0.1353** | **0.0338**     | Poor       |
| 25.0s    | 0.0821     | 0.0205         | Poor       |
| **30.0s**| **0.0498** | **0.0124**     | Very Poor  |

## Key Insights

### Time Performance Impact

1. **Time weight is 25%** - This is NOT negligible!
2. **Target: < 5 seconds per test case** for good scores
3. **Hard limit: 30 seconds** - Solutions timing out get 0 score

### Performance Degradation Examples

Assuming 100% success rate and perfect cost:

```
Time  1s → combined_score = 0.9762
Time  5s → combined_score = 0.9016  (-0.075 vs 1s)
Time 10s → combined_score = 0.8420  (-0.060 vs 5s)
Time 20s → combined_score = 0.7838  (-0.058 vs 10s)
Time 30s → combined_score = 0.7624  (-0.021 vs 20s)
```

**Going from 5s to 10s costs ~6% of total score!**

## Algorithm Optimization Priorities

### 1. First: Ensure Correctness (40%)
- All constraints must be satisfied
- Invalid solutions get 0 score

### 2. Then: Minimize Cost (35%)
- This is the primary competition metric
- Focus on:
  - Better server placement
  - Optimal routing
  - Server type selection

### 3. Always: Keep It Fast (25%)
- **Don't ignore execution time!**
- Complexity matters:
  - O(N²) is acceptable
  - O(N³) may timeout on large cases
- Use efficient data structures
- Avoid redundant computations

## Evolution Strategy Recommendations

### Balance Cost vs. Time Trade-offs

1. **Quick wins first**: Simple, fast heuristics
2. **Iterative refinement**: Add complexity only if time allows
3. **Profiling**: Identify bottlenecks
4. **Early termination**: Stop optimization when improvement is marginal

### Example Trade-off Analysis

```
Option A: Greedy algorithm
  - Cost: 8,000
  - Time: 2s
  - Combined: 0.40×1.0 + 0.35×1.25 + 0.25×0.82 = 1.04

Option B: Complex optimization
  - Cost: 7,500
  - Time: 15s
  - Combined: 0.40×1.0 + 0.35×1.33 + 0.25×0.22 = 0.92

Option A wins! Faster execution compensates for slightly higher cost.
```

## Artifacts Reported

The evaluator now reports:

```json
{
  "metrics": {
    "valid_solutions": 1.0,
    "avg_cost": 7234.5,
    "avg_time": 4.23,
    "cost_score": 1.382,
    "time_score": 0.655,
    "combined_score": 0.928
  },
  "artifacts": {
    "num_test_cases": 5,
    "num_valid": 5,
    "avg_time_per_case": "4.23s",
    "total_time": "21.15s",
    "time_breakdown": {
      "fastest": "3.12s",
      "slowest": "5.87s"
    },
    "scoring_weights": {
      "success_rate": "40%",
      "cost": "35%",
      "time": "25%"
    }
  }
}
```

## Summary

- ✅ **Time is critical**: 25% weight, exponential penalty
- ✅ **Target performance**: < 5 seconds per test case
- ✅ **Cost still matters most**: 35% weight
- ✅ **Must be correct first**: 40% for success rate
- ✅ **Balance is key**: Don't over-optimize one metric at the expense of others

---

**Remember**: In the actual competition, cost is the PRIMARY ranking metric among solutions with the same success rate. Our scoring system reflects this by balancing all three factors appropriately for evolution.
