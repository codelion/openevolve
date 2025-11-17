# Circle Packing with Hierarchical Evolution

This example demonstrates the hierarchical abstraction layer system on the classic circle packing problem (n=26 circles in a unit square).

## Goal

Replicate or improve over AlphaEvolve's reported result of **2.635** for the sum of radii.

## What Makes This Different?

This example uses the **hierarchical abstraction layer system** which adds:

1. **Five-Layer Evolution** - From concrete code (L1) to abstract principles (L5)
2. **Evolutionary Memory Graph** - Rich context beyond simple similarity
3. **Tiered Models** - Fast models for L1, reasoning models for L4/L5
4. **Automatic Layer Transitions** - Escalates to higher layers when plateaued
5. **Insight Extraction** - Learns patterns from evolution history

## Quick Start

### Option 1: Using OpenAI GPT-4 (Default)

```bash
# Navigate to this directory
cd examples/circle_packing_hierarchical

# Make sure you have set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run hierarchical evolution
python run_hierarchical.py --iterations 300
```

### Option 2: Using KIMI K2 + GLM-4.6 (Cost-Optimized)

For cost-effective multi-API configuration using KIMI K2 for high reasoning and GLM-4.6 for low reasoning:

```bash
# Set up API keys
export KIMI_API_KEY="your-kimi-api-key"
export GLM_API_KEY="your-glm-api-key"

# Run with multi-API configuration
python run_hierarchical.py --config config_multi_api.yaml --iterations 300
```

**Cost savings**: ~3-5x cheaper than using premium models for all tiers!

See [Multi-API Setup Guide](/docs/MULTI_API_SETUP.md) for detailed configuration.

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
- **Expected**: Incremental improvements from ~2.0 to ~2.2
- **Models**: Tier 0 (fast, high temperature)

### Phase 2: L2 Implementation Patterns (Iterations 20-60)
- **Trigger**: L1 plateaus for 5 iterations
- **Active Layer**: L2 (algorithms, data structures)
- **Expected**: Improvements to ~2.3-2.4 through better packing algorithms
- **Models**: Tier 1 (standard)
- **Example changes**:
  - Switch from simple ring placement to optimized grid
  - Add adaptive radius computation
  - Implement local search refinement

### Phase 3: L3 Architectural Components (Iterations 60-150)
- **Trigger**: L2 plateaus for 15 iterations
- **Active Layer**: L3 (architectural changes)
- **Expected**: Breakthrough to ~2.5-2.6 through new architectural components
- **Models**: Tier 2 (strong)
- **Example changes**:
  - Add multi-stage packing (large circles first, fill gaps)
  - Introduce constraint satisfaction framework
  - Implement hierarchical decomposition

### Phase 4: L4 Algorithmic Paradigms (Iterations 150-300)
- **Trigger**: L3 plateaus for 50 iterations
- **Active Layer**: L4 (paradigm shifts)
- **Expected**: Push towards 2.635+ through paradigm changes
- **Models**: Tier 2/3 (strong/reasoning)
- **Example changes**:
  - Switch from construction to optimization paradigm
  - Adopt continuous relaxation approaches
  - Use symmetry exploitation

## How Hierarchical Evolution Helps

### Problem with Standard Evolution
Standard evolutionary approaches (including base OpenEvolve) can get stuck in **incremental improvement bias** - they're good at refining existing solutions but struggle with architectural innovations.

### Hierarchical Solution
The hierarchical system enables **strategic pivots**:

1. **L5/L4**: Can change fundamental approach without throwing away all knowledge
2. **L3**: Can add new architectural components (e.g., "discretization loss" in matrix multiplication)
3. **L2/L1**: Continuously optimize within stable architecture

### Real-World Analogy
Think of solving circle packing like building a house:
- **L1**: Choosing paint colors, door handles (code details)
- **L2**: Deciding construction techniques (implementation patterns)
- **L3**: Designing room layouts, structures (architecture)
- **L4**: Choosing architectural style (paradigm)
- **L5**: Deciding "house" vs "apartment" vs "castle" (meta-principle)

Standard evolution only works at L1. Hierarchical evolution can redesign the whole structure while keeping what works.

## Monitoring Progress

The run script outputs hierarchical statistics:

```
📊 Hierarchical Evolution Statistics:
  Current generation: 150

  Layer Status:
    code_details: best=0.8500, attempts=150, success_rate=35%
    implementation_patterns: best=0.9100, attempts=45, success_rate=42%
    architectural_components: best=0.9600, attempts=12, success_rate=58%

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
- Standard evolution: Steady incremental progress, likely plateaus around 2.2-2.4
- Hierarchical evolution: More varied trajectory with architectural breakthroughs, targeting 2.635+

## Expected Results

Based on the hierarchical design and AlphaEvolve paper:

| Approach | Expected Sum of Radii | Notes |
|----------|----------------------|-------|
| Initial program | ~1.8-2.0 | Simple ring pattern |
| Standard evolution | ~2.2-2.4 | Incremental improvements |
| Hierarchical (L1-L2) | ~2.3-2.5 | Better algorithms |
| Hierarchical (L1-L3) | ~2.5-2.635 | Architectural innovations |
| AlphaEvolve (paper) | **2.635** | Reported result |

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

### Models not using different tiers
- Verify tier models are configured in `config.yaml`
- Check logs for "Using X ensemble in Y phase"

### EMG not growing
- Ensure `emg_enabled: true` in config
- Check that iterations are completing successfully

### No insights extracted
- Set `enable_insight_extraction: true`
- Wait for `insight_extraction_interval` generations (default: 50)

## Next Steps

1. **Try Different Models**: Use o3-mini or o1 for Tier 3 (requires API access)
2. **Adjust Triggers**: Modify `l2_plateau_iterations` etc. for faster/slower transitions
3. **Enable LLM Insights**: Set `use_llm_for_insights: true` for deeper pattern analysis
4. **Increase Iterations**: Run for 500-1000 iterations to see L4/L5 evolution

## References

- AlphaEvolve Paper: Reports 2.635 for n=26
- Hierarchical Evolution Design: See `/docs/HIERARCHICAL_EVOLUTION.md`
- OpenEvolve Documentation: See main README

## License

Same as OpenEvolve (MIT License)
