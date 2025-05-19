# Circle Packing Example (n=26)

This example attempts to replicate one of the specific results from the AlphaEvolve paper (Section B.12): packing 26 circles inside a unit square to maximize the sum of their radii.

## Problem Description

The problem is to pack 26 disjoint circles inside a unit square so as to maximize the sum of their radii. The circles must:
- Lie entirely within the unit square [0,1] × [0,1]
- Not overlap with each other

This is a well-studied problem in computational geometry with applications in various fields including material science, facility location, and computer graphics.

## AlphaEvolve Result

According to the paper, AlphaEvolve improved the state of the art for n=26 from 2.634 to 2.635.

## Running the Example

```bash
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config.yaml --iterations 100
```

## Evaluation Metrics

The evaluator calculates several metrics:
- `sum_radii`: The best sum of radii achieved across all trials
- `avg_sum_radii`: Average sum of radii across successful trials
- `target_ratio`: Ratio of achieved sum to target (2.635)
- `reliability`: Fraction of trials that produced valid solutions
- `avg_time`: Average execution time
- `combined_score`: A weighted combination of the above metrics (main fitness metric)

## Expected Results

A successful run should find a packing arrangement with sum of radii approaching or exceeding the value reported in the AlphaEvolve paper: 2.635 for n=26.

## Visualization

The initial program includes a visualization function that you can use to see the packing arrangement:

```python
# Add this to the end of the best program
if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)
```
