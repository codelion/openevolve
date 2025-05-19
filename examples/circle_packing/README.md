# Circle Packing Example

This example attempts to replicate one of the results from the AlphaEvolve paper (Section B.12): packing circles inside a unit square to maximize the sum of their radii.

## Problem Description

Given a positive integer n, the problem is to pack n disjoint circles inside a unit square so as to maximize the sum of their radii. The circles must:
- Lie entirely within the unit square [0,1] × [0,1]
- Not overlap with each other

This is a well-studied problem in computational geometry with applications in various fields including material science, facility location, and computer graphics.

## AlphaEvolve Results

According to the paper, AlphaEvolve found new constructions improving the state of the art:
- For n = 26, improved from 2.634 to 2.635
- For n = 32, improved from 2.936 to 2.937

## Running the Example

```bash
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config.yaml --iterations 100
```

## Evaluation Metrics

The evaluator calculates several metrics:
- `sum_radii_26`: Sum of radii for n=26
- `sum_radii_32`: Sum of radii for n=32
- `target_ratio_26`: Ratio of achieved sum to target (2.635) for n=26
- `target_ratio_32`: Ratio of achieved sum to target (2.937) for n=32
- `validity`: 1.0 if solutions for both n=26 and n=32 are valid, 0.0 otherwise
- `avg_target_ratio`: Average of target ratios
- `combined_score`: avg_target_ratio * validity (main fitness metric)

## Expected Results

A successful run should find packing arrangements with sums approaching or exceeding the values reported in the AlphaEvolve paper:
- n=26: 2.635
- n=32: 2.937

## Visualization

You can visualize the best solution by adding a visualization function to the best program:

```python
def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha='center', va='center')
    
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()
    
# Example usage:
# centers, radii, sum_radii = run_packing(26)
# visualize(centers, radii)
```
