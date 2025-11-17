# Rectilinear Polygon Decomposition with Hierarchical Evolution

This example demonstrates the hierarchical abstraction layer system on the rectilinear polygon decomposition problem.

## Problem Statement

Given a rectilinear polygon (polygon with only right-angle edges), decompose it into rectangles such that:

1. **Primary objective**: Minimize rectangles with sides smaller than a threshold length `l_min`
2. **Secondary objective**: Minimize the total number of rectangles

The algorithm must be:
- **Fast**: Complete within a constrained time limit per polygon
- **Feasible**: Successfully decompose as many test polygons as possible

## Goal

Evolve algorithms that maximize the number of polygons successfully decomposed within the time limit while minimizing constraint violations and rectangle count.

## What Makes This Different?

This example uses the **hierarchical abstraction layer system** which adds:

1. **Five-Layer Evolution** - From concrete code (L1) to abstract principles (L5)
2. **Evolutionary Memory Graph** - Rich context beyond simple similarity
3. **Tiered Models** - Fast models for L1, reasoning models for L4/L5
4. **Automatic Layer Transitions** - Escalates to higher layers when plateaued
5. **Insight Extraction** - Learns patterns from evolution history

## Quick Start

```bash
# Navigate to this directory
cd examples/polygon_decomposition

# Make sure you have set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Install OR-Tools (required for CP-SAT solver)
pip install ortools

# Run hierarchical evolution
python run_hierarchical.py --iterations 300
```

## Configuration

The `config.yaml` file contains the hierarchical evolution settings:

```yaml
hierarchical:
  enabled: true  # Enable hierarchical evolution

  # Model tiers for different layers
  tier0_models:  # Fast for L1 (code details)
    - name: "gpt-4o"
      temperature: 0.8

  tier1_models:  # Standard for L2 (patterns)
    - name: "gpt-4o"
      temperature: 0.7

  tier2_models:  # Strong for L3/L4 (architecture/paradigms)
    - name: "gpt-4o"
      temperature: 0.6

  tier3_models:  # Reasoning for L4/L5 (strategic pivots)
    - name: "gpt-4o"  # Use o3-mini or o1 if available
      temperature: 0.5

  # Layer transition triggers
  l2_plateau_iterations: 5   # Evolve patterns when code plateaus
  l3_plateau_iterations: 15  # Evolve architecture when patterns plateau
  l4_plateau_iterations: 50  # Evolve paradigms when architecture plateaus

  # Insight extraction
  enable_insight_extraction: true
  insight_extraction_interval: 50

  # EMG (Evolutionary Memory Graph)
  emg_enabled: true
```

## Expected Evolution Trajectory

Based on the hierarchical design:

### Phase 1: L1 Code Details (Iterations 1-20)
- **Active Layer**: L1 (code tweaks, hyperparameters)
- **Expected**: Initial improvements in CP-SAT model tuning
- **Models**: Tier 0 (fast, high temperature)
- **Example changes**:
  - Adjusting candidate enumeration heuristics
  - Tweaking adjacency detection thresholds
  - Optimizing lexicographic weights

### Phase 2: L2 Implementation Patterns (Iterations 20-60)
- **Trigger**: L1 plateaus for 5 iterations
- **Active Layer**: L2 (algorithms, data structures)
- **Expected**: Better primitive rectangle generation
- **Models**: Tier 1 (standard)
- **Example changes**:
  - Improved sweep line algorithm variants
  - Better rectangle merging strategies
  - Efficient solid rectangle detection

### Phase 3: L3 Architectural Components (Iterations 60-150)
- **Trigger**: L2 plateaus for 15 iterations
- **Active Layer**: L3 (architectural changes)
- **Expected**: Breakthrough through new components
- **Models**: Tier 2 (strong)
- **Example changes**:
  - Multi-stage decomposition (coarse-to-fine)
  - Hierarchical partitioning strategies
  - Constraint propagation frameworks
  - Graph-based merging algorithms

### Phase 4: L4 Algorithmic Paradigms (Iterations 150-300)
- **Trigger**: L3 plateaus for 50 iterations
- **Active Layer**: L4 (paradigm shifts)
- **Expected**: Fundamental approach changes
- **Models**: Tier 2/3 (strong/reasoning)
- **Example changes**:
  - Switch from construction to optimization
  - Adopt dynamic programming approaches
  - Use graph decomposition techniques
  - Employ geometric transformations

## How Hierarchical Evolution Helps

### Problem with Standard Evolution
Standard evolutionary approaches can get stuck in **incremental improvement bias** - they're good at refining the CP-SAT formulation but struggle with architectural innovations like switching to completely different decomposition paradigms.

### Hierarchical Solution
The hierarchical system enables **strategic pivots**:

1. **L5/L4**: Can change fundamental approach (e.g., from SAT to dynamic programming)
2. **L3**: Can add new architectural components (e.g., hierarchical decomposition)
3. **L2/L1**: Continuously optimize within stable architecture

### Real-World Analogy
Think of solving polygon decomposition like designing a house:
- **L1**: Choosing algorithm parameters, constants (paint colors)
- **L2**: Deciding data structures, subroutines (construction techniques)
- **L3**: Designing major components (room layouts)
- **L4**: Choosing algorithmic approach (architectural style)
- **L5**: Deciding fundamental paradigm (house vs apartment vs castle)

Standard evolution only works at L1. Hierarchical evolution can redesign the whole structure while keeping what works.

## The Initial Program

The initial program uses a **CP-SAT (Constraint Programming - Satisfiability)** approach:

1. **Sweep Line Decomposition**: Partition polygon into primitive rectangles using grid lines
2. **Candidate Generation**: Enumerate all valid merged rectangles via BFS/DFS
3. **CP-SAT Optimization**: Select optimal subset that:
   - Covers each primitive exactly once (partition constraint)
   - Minimizes rectangles with sides < `l_min` (primary)
   - Minimizes total rectangles (secondary)
   - Maximizes coverage (tertiary)

This is a solid baseline, but hierarchical evolution can discover fundamentally different approaches.

## Evaluation Metrics

The evaluator tests each algorithm on multiple randomly generated polygons:

- **Primary Fitness**: Number of polygons successfully decomposed within time limit
- **Secondary**: Average violations (rectangles with sides < `l_min`)
- **Tertiary**: Average number of rectangles

The combined score uses lexicographic ordering:
```
score = feasibility_count * 1000 - violation_penalty * 10 - rectangle_penalty
```

## Polygon Generation

Polygons are generated using a **random rectangle placement** approach:

1. Place random non-overlapping rectangles on a grid
2. Extract the outer boundary of their union
3. Return vertices in counter-clockwise order

Difficulty levels control:
- **Easy**: 3-5 rectangles, small grid (10x10)
- **Medium**: 5-8 rectangles, medium grid (20x20)
- **Hard**: 8-12 rectangles, large grid (30x30)

## Monitoring Progress

The run script outputs hierarchical statistics:

```
📊 Hierarchical Evolution Statistics:
  Current generation: 150

  Layer Status:
    code_details: best=5234.50, attempts=150, success_rate=35%
    implementation_patterns: best=6120.30, attempts=45, success_rate=42%
    architectural_components: best=7890.10, attempts=12, success_rate=58%

  EMG: 487 nodes, 623 edges
  Insights extracted: 3
```

This tells you:
- Which layers are being evolved
- Success rates at each layer
- Size of the Evolutionary Memory Graph
- Number of insights extracted

## Checkpoints

Checkpoints are saved every 50 iterations to `openevolve_output/checkpoints/checkpoint_N/`:

- `database.pkl` - Program database
- `emg.pkl` - Evolutionary Memory Graph
- `layer_transitions.json` - Layer transition history
- `insights.json` - Extracted insights
- `best_program.py` - Best program code

Resume from checkpoint:

```bash
python run_hierarchical.py --checkpoint openevolve_output/checkpoints/checkpoint_100/
```

## Comparing with Standard Evolution

To compare with standard (non-hierarchical) evolution:

1. Set `hierarchical.enabled: false` in `config.yaml`
2. Run: `python run_hierarchical.py --iterations 300`

You should observe:
- **Standard evolution**: Steady incremental progress, likely plateaus at local optimum
- **Hierarchical evolution**: More varied trajectory with architectural breakthroughs

## Expected Results

Based on the hierarchical design:

| Approach | Expected Feasibility | Notes |
|----------|---------------------|-------|
| Initial program | ~5-7/10 polygons | CP-SAT baseline |
| Standard evolution | ~7-8/10 polygons | Parameter tuning |
| Hierarchical (L1-L2) | ~8-9/10 polygons | Better algorithms |
| Hierarchical (L1-L3) | ~9-10/10 polygons | Architectural innovations |

With secondary objectives also improving through evolution.

**Note**: Actual results depend on:
- Model quality (use o3-mini/o1 for Tier 3 for best results)
- Random seed
- Number of iterations
- Evolution trajectory (hierarchical evolution is less deterministic)

## Analysis

After running, analyze the evolution:

```python
# View evolution trace
import json

with open("openevolve_output/checkpoints/checkpoint_300/layer_transitions.json") as f:
    transitions = json.load(f)

# Check when each layer was activated
for event in transitions["layer_transitions"]:
    print(f"Iteration {event['iteration']}: {event['reason']} to {event['layer']}")

# View insights
with open("openevolve_output/checkpoints/checkpoint_300/insights.json") as f:
    insights = json.load(f)

for insight_id, insight in insights.items():
    print(f"{insight['insight_type']}: {insight['content']}")
```

## Troubleshooting

### "Hierarchical evolution is DISABLED"
- Check `config.yaml` has `hierarchical.enabled: true`
- Make sure you're using the correct config file

### OR-Tools not installed
```bash
pip install ortools
```

### Models not using different tiers
- Verify tier models are configured in `config.yaml`
- Check logs for "Using X ensemble in Y phase"

### EMG not growing
- Ensure `emg_enabled: true` in config
- Check that iterations are completing successfully

### No insights extracted
- Set `enable_insight_extraction: true`
- Wait for `insight_extraction_interval` generations (default: 50)

### Evaluation timeouts
- Increase `timeout` in evaluator section of config
- Reduce `num_test_polygons` in evaluator.py for faster iterations

## Next Steps

1. **Try Different Models**: Use o3-mini or o1 for Tier 3 (requires API access)
2. **Adjust Triggers**: Modify `l2_plateau_iterations` etc. for faster/slower transitions
3. **Enable LLM Insights**: Set `use_llm_for_insights: true` for deeper pattern analysis
4. **Increase Iterations**: Run for 500-1000 iterations to see L4/L5 evolution
5. **Harder Polygons**: Change difficulty to "hard" in evaluator.py
6. **More Polygons**: Increase `num_test_polygons` for more robust evaluation

## Technical Details

### CP-SAT Formulation

The initial program formulates polygon decomposition as a constraint satisfaction problem:

**Variables**: Boolean variable `cand_i` for each candidate rectangle

**Constraints**:
```
For each primitive rectangle p:
  sum(cand_i where p ∈ cand_i) = 1  # Exactly-one coverage
```

**Objective** (lexicographic):
```
minimize:
  (num + 1)² * (rectangles with sides < l_min) +
  (num + 1) * (total rectangles) -
  (primitives covered)
```

### Sweep Line Algorithm

1. Extract all unique x and y coordinates from polygon vertices
2. Create grid using these coordinates
3. Check each grid cell to see if its center is inside the polygon
4. Cells inside form the set of primitive rectangles

### BFS/DFS Candidate Enumeration

1. Start from each primitive as a seed
2. Expand by adding adjacent primitives
3. Check if union forms a solid rectangle (area matches bounding box)
4. Continue until all connected solid rectangles are found

## References

- OR-Tools CP-SAT Solver: https://developers.google.com/optimization/cp/cp_solver
- Hierarchical Evolution Design: See `/docs/HIERARCHICAL_EVOLUTION.md`
- OpenEvolve Documentation: See main README

## License

Same as OpenEvolve (MIT License)
