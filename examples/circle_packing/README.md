# Constructor-Based Circle Packing Example (n=26)

This example attempts to replicate one of the specific results from the AlphaEvolve paper (Section B.12): packing 26 circles inside a unit square to maximize the sum of their radii.

## Problem Description

The problem is to pack 26 disjoint circles inside a unit square so as to maximize the sum of their radii. The circles must:
- Lie entirely within the unit square [0,1] × [0,1]
- Not overlap with each other

According to the paper, AlphaEvolve improved the state of the art for n=26 from 2.634 to 2.635.

## Constructor-Based Approach

Following insights from the AlphaEvolve paper, we use a constructor-based approach rather than a search algorithm:

> "For problems with highly symmetric solutions it is advantageous to evolve constructor functions as these tend to be more concise." - AlphaEvolve paper, Section 2.1

Instead of evolving a search algorithm that tries different configurations, we evolve a function that directly constructs a specific packing arrangement. This approach:

1. Is more deterministic (produces the same output each time)
2. Can leverage geometric knowledge about optimal packings
3. Tends to be more concise and easier to evolve
4. Works well for problems with inherent structure or symmetry

## Running the Example

```bash
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config.yaml --iterations 200
```

## Evolved Constructor Functions

The evolution might discover various pattern-based approaches:

1. **Concentric rings**: Placing circles in concentric rings around a central circle
2. **Hexagonal patterns**: Portions of a hexagonal lattice (theoretically optimal for infinite packings)
3. **Mixed-size arrangements**: Varying circle sizes to better utilize space near the boundaries
4. **Specialized patterns**: Custom arrangements specific to n=26

## Evaluation Metrics

The evaluator calculates several metrics:
- `sum_radii`: The sum of radii achieved by the constructor
- `target_ratio`: Ratio of achieved sum to target (2.635)
- `validity`: Confirms circles don't overlap and stay within bounds
- `combined_score`: A weighted combination of metrics (main fitness metric)

## Visualization

The program includes a visualization function to see the constructed packing:

```python
# Add this to the end of the best program
if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)
```

## What to Expect

The evolution process should discover increasingly better constructor functions, with several possible patterns emerging. Given enough iterations, it should approach or exceed the 2.635 value achieved in the paper.

Different runs may converge to different packing strategies, as multiple near-optimal arrangements are possible for this problem.
