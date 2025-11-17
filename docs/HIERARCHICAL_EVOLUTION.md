# Hierarchical Abstraction Layers: Design and Implementation

This document describes the hierarchical abstraction layer system in OpenEvolve, designed to enable breakthrough creativity through multi-level evolution.

## Overview

The hierarchical abstraction layer system extends OpenEvolve with a five-layer abstraction hierarchy where each layer evolves at different frequencies using specialized models. This enables strategic pivots and architectural innovations beyond simple code optimization.

### The Five Layers

1. **L5: Meta-Principles** (Most Abstract)
   - Fundamental philosophical approaches and mathematical frameworks
   - Examples: "Exploit problem symmetry", "Transform continuous to discrete"
   - Evolves rarely (every 500+ generations), uses Tier 3 reasoning models
   - Changes only when completely stuck

2. **L4: Algorithmic Paradigms** (Strategic)
   - Specific mathematical/computational strategies
   - Examples: "Use group theory", "Apply dynamic programming"
   - Evolves when L3 exhausts (every 100+ generations)
   - Uses Tier 2/3 models

3. **L3: Architectural Components** (Tactical)
   - Concrete structures and their relationships
   - Examples: "Permutation arrays + composition table", "State space representation"
   - Evolves when L2 exhausts (every 20+ generations)
   - Uses Tier 2 models

4. **L2: Implementation Patterns** (Operational)
   - Specific algorithms, data structures, code patterns
   - Examples: "Adam optimizer with LR schedule", "LRU cache for memoization"
   - Evolves when L1 plateaus (every 5+ generations)
   - Uses Tier 1 models

5. **L1: Code Details** (Concrete)
   - Actual code, hyperparameters, specific values
   - Examples: `learning_rate=0.023`, exact loop structures
   - Evolves continuously every generation
   - Uses Tier 0/1 models

## Architecture Components

### 1. Layer Abstractions (`openevolve/hierarchy/layers.py`)

Defines the five-layer hierarchy with:
- `Layer` base class for all abstraction layers
- `L1CodeDetails` through `L5MetaPrinciples` specialized classes
- `HierarchicalProgram` linking all layers together

```python
from openevolve.hierarchy import HierarchicalProgram, L1CodeDetails, L2ImplementationPatterns

# Create a hierarchical program
program = HierarchicalProgram(
    id="prog_001",
    l1_code=L1CodeDetails(
        id="l1_001",
        code="def optimize(): ...",
        hyperparameters={"lr": 0.01}
    ),
    l2_patterns=L2ImplementationPatterns(
        id="l2_001",
        patterns=["Adam optimizer", "Gradient clipping"],
        algorithms=["SGD with momentum"]
    )
)
```

### 2. Evolutionary Memory Graph (`openevolve/hierarchy/emg.py`)

A rich graph structure that explicitly encodes relationships between solutions, mutations, failures, insights, and constraints.

**Node Types:**
- `SOLUTION`: A solution at a specific layer
- `MUTATION`: A transition between solutions
- `FAILURE`: A failed attempt with diagnostic info
- `INSIGHT`: Extracted pattern or learning
- `CONSTRAINT`: Active requirement
- `ANALOGY`: Cross-domain analogy

**Edge Types:**
- `PARENT_OF` / `CHILD_OF`: Evolutionary lineage
- `SIMILAR_TO`: Structural similarity
- `CONTRADICTS`: Incompatible approaches
- `ENABLES`: One solution unlocks another
- `SUCCEEDED_BECAUSE` / `FAILED_BECAUSE`: Causal explanations

```python
from openevolve.hierarchy import EvolutionaryMemoryGraph, EMGNode, NodeType

emg = EvolutionaryMemoryGraph()

# Add solution node
solution_node = EMGNode(
    id="sol_001",
    node_type=NodeType.SOLUTION,
    layer_type=LayerType.L2_IMPLEMENTATION_PATTERNS,
    score=0.85,
    description="Adam optimizer with adaptive LR"
)
emg.add_node(solution_node)

# Query the graph
recent_solutions = emg.query_nodes(
    node_type=NodeType.SOLUTION,
    layer_type=LayerType.L2_IMPLEMENTATION_PATTERNS,
    min_score=0.7,
    generation_window=(90, 100)
)
```

### 3. Context Compilation (`openevolve/hierarchy/context.py`)

Implements the 7-phase context compilation algorithm:

1. **Constraint Propagation**: Collect requirements from ancestor layers
2. **Sibling Analysis**: Analyze recent attempts at same level
3. **Cousin Transfer**: Find successful patterns in parallel branches
4. **Causal Chain Analysis**: Trace backwards from breakthroughs
5. **Conflict Detection**: Identify failing combinations
6. **Trend Prediction**: Analyze trajectory and predict next steps
7. **Assembly & Prioritization**: Compile with smart truncation

```python
from openevolve.hierarchy import ContextCompiler, ContextQuery, LayerType

compiler = ContextCompiler(emg)

query = ContextQuery(
    focus_node_id="node_123",
    layers=[LayerType.L2_IMPLEMENTATION_PATTERNS],
    recent_weight=0.7,
    local_weight=0.6,
    max_tokens=50000
)

context_bundle = compiler.compile_context(query)
formatted_context = context_bundle.get_formatted_context(max_tokens=50000)
```

### 4. Tiered Model Selection (`openevolve/hierarchy/model_tiers.py`)

Uses different models for different layers based on reasoning complexity:

- **Tier 0**: Fastest models (haiku, gpt-3.5-turbo) for L1 local search
- **Tier 1**: Standard models (sonnet, gpt-4o) for L2 patterns
- **Tier 2**: Strong models (opus, gpt-4) for L3 architecture, L4 paradigms
- **Tier 3**: Reasoning models (o1, o3-mini) for L4/L5 strategic pivots

```python
from openevolve.hierarchy import TieredModelSelector, ModelTier

selector = TieredModelSelector(tier_configs)

# Get ensemble for a specific layer
ensemble = selector.get_ensemble_for_layer(
    LayerType.L3_ARCHITECTURAL_COMPONENTS,
    phase="plateau"  # Use stronger models during plateau
)
```

### 5. Layer Transition Manager (`openevolve/hierarchy/transitions.py`)

Detects plateaus and triggers transitions between abstraction layers:

**Evolution Phases:**
- `NORMAL`: Standard evolution (95% of time)
- `EXPLORATION`: Active exploration when L1 plateaus (4%)
- `PLATEAU`: L3 plateau detected, seeking breakthrough
- `CRISIS`: Multiple layers exhausted, major pivot needed (1%)

```python
from openevolve.hierarchy import LayerTransitionManager, TransitionTriggers

triggers = TransitionTriggers(
    l2_plateau_iterations=5,
    l3_plateau_iterations=20,
    l4_plateau_iterations=100,
    l5_plateau_iterations=500
)

manager = LayerTransitionManager(triggers)

# Update after evolution attempt
improved = manager.update_layer_status(
    layer_type=LayerType.L1_CODE_DETAILS,
    score=0.85,
    previous_best=0.82,
    iteration=100
)

# Get current phase
phase = manager.get_evolution_phase(iteration=100)

# Check if layer should evolve
should_evolve_l2 = manager.should_evolve_layer(
    LayerType.L2_IMPLEMENTATION_PATTERNS,
    iteration=100
)
```

### 6. Insight Extraction (`openevolve/hierarchy/insights.py`)

Periodically analyzes evolution history to extract patterns:

- Performance patterns (what correlates with high performance)
- Failure patterns (what consistently fails)
- Breakthrough patterns (what led to major improvements)
- Combination patterns (what works well together)

```python
from openevolve.hierarchy import InsightExtractor

extractor = InsightExtractor(emg, llm_ensemble)

# Extract insights from last 100 generations
insights = extractor.extract_insights(generation_window=(900, 1000))

# Get relevant insights for a layer
relevant = extractor.get_relevant_insights(
    LayerType.L2_IMPLEMENTATION_PATTERNS,
    max_results=5
)
```

## Configuration

See `examples/hierarchical_evolution_config.yaml` for a complete example.

Key configuration sections:

```yaml
hierarchical:
  enabled: true

  # Model tiers
  tier0_models:  # Fast models for L1
    - name: "gpt-4o-mini"
  tier1_models:  # Standard for L2
    - name: "gpt-4o"
  tier2_models:  # Strong for L3/L4
    - name: "gpt-4o"
  tier3_models:  # Reasoning for L4/L5
    - name: "o3-mini"
      reasoning_effort: "high"

  # Transition triggers
  l2_plateau_iterations: 5
  l3_plateau_iterations: 20
  l4_plateau_iterations: 100
  l5_plateau_iterations: 500

  # Insight extraction
  enable_insight_extraction: true
  insight_extraction_interval: 100

  # EMG
  emg_enabled: true
```

## How It Solves the Creativity Problem

### Current Systems

AlphaEvolve uses an evolutionary approach where evolution selects for heuristics effective at improving already high-quality solutions. This creates incremental improvement bias.

### Hierarchical Abstraction

**Layer 5 changes** are rare but enable **paradigm shifts**:
- Don't throw away strategic direction
- Enable fundamental rethinking when stuck

**Layer 4 changes** allow **strategic pivots** without starting from scratch:
- Switch mathematical frameworks
- Maintain problem understanding

**Layer 3 changes** provide **architectural innovation**:
- Add new components (e.g., "discretization loss")
- Unlock new optimization space
- Preserve strategic coherence

**Layer 2/1** provide **continuous optimization**:
- Refine implementation
- Tune hyperparameters
- Maintain architectural stability

### Concrete Example: Matrix Multiplication

1. **Generations 1-100**: Bootstrap
   - L5: "Decompose into independent subproblems" (fixed)
   - L4: "Find tensor decomposition" (fixed)
   - L3: "Gradient descent on tensor factors"
   - L2/L1: Optimize Adam, learning rate, etc.
   - Result: Score 0.65

2. **Generations 101-200**: L1 optimization
   - Evolve hyperparameters
   - Result: Score 0.70 (incremental)

3. **Generations 201-300**: L2 plateau → exploration
   - Try new patterns (AdamW, gradient clipping)
   - Result: Score 0.73

4. **Generations 301-600**: L3 architectural innovation
   - Add discretization loss
   - Add noise injection
   - Add annealing schedule
   - Result: Score 0.82 (major breakthrough!)

5. **Generations 601-1000**: Optimize new architecture
   - Fine-tune all components
   - Result: Discovered improved matrix multiplication algorithm

**The breakthrough came from:**
- L5/L4 staying constant (strategic coherence)
- L3 evolving (architectural innovation)
- L2/L1 optimizing (continuous refinement)

## Usage Example

```python
from openevolve import OpenEvolve
from openevolve.config import load_config

# Load config with hierarchical evolution enabled
config = load_config("examples/hierarchical_evolution_config.yaml")

# Create OpenEvolve instance
openevolve = OpenEvolve(
    initial_program_path="examples/my_problem/initial_program.py",
    evaluation_file="examples/my_problem/evaluator.py",
    config=config
)

# Run evolution
import asyncio
best_program = asyncio.run(openevolve.run(iterations=1000))
```

The hierarchical system will automatically:
1. Evolve L1 (code) every generation
2. Trigger L2 (patterns) when L1 plateaus
3. Trigger L3 (architecture) when L2 exhausts
4. Extract insights every 100 generations
5. Use appropriate model tiers for each layer
6. Compile rich context from the EMG
7. Adapt exploration/exploitation based on phase

## Benefits

1. **Breakthrough Creativity**: Can make strategic pivots and architectural changes
2. **Efficient Resource Use**: Fast models for frequent L1, expensive models for rare L5
3. **Strategic Coherence**: Higher layers provide stable direction
4. **Rich Context**: EMG provides multi-dimensional context beyond simple similarity
5. **Adaptive Evolution**: Automatically adjusts strategy based on phase
6. **Knowledge Compression**: Insights extracted and reused across evolution

## Performance Considerations

- **Memory**: EMG grows over time; insights compress knowledge
- **Compute**: Tiered models minimize cost (fast for L1, expensive only for L4/L5)
- **Token Usage**: Context compilation manages token budgets with smart truncation
- **Storage**: Periodic insight extraction and EMG persistence to disk

## Future Enhancements

- Cross-domain analogy database
- Hybrid storage (hot/warm/cold for EMG)
- LLM-based insight extraction (currently rule-based)
- Interactive layer visualization
- Multi-objective layer optimization

## References

Based on the design document "Hierarchical Abstraction Layers: Detailed Design" which describes:
- Five-layer abstraction hierarchy
- Evolutionary Memory Graph (EMG)
- Multi-dimensional context compilation
- Tiered model selection
- Layer transition mechanisms
- Insight extraction and compression
