# Hierarchical Abstraction Layer System

This module implements a five-layer abstraction hierarchy for breakthrough creativity in evolutionary code optimization.

## Modules

- **`layers.py`**: L1-L5 layer abstractions and hierarchical program data structures
- **`emg.py`**: Evolutionary Memory Graph for tracking solutions, insights, and relationships
- **`context.py`**: Multi-dimensional context compilation (7-phase algorithm)
- **`model_tiers.py`**: Tiered model selection (Tier 0-3 for different layers)
- **`transitions.py`**: Layer transition manager and plateau detection
- **`insights.py`**: Insight extraction and compression from evolution history

## Quick Start

```python
from openevolve.hierarchy import (
    HierarchicalProgram,
    EvolutionaryMemoryGraph,
    ContextCompiler,
    TieredModelSelector,
    LayerTransitionManager,
    InsightExtractor
)

# Enable via configuration
config.hierarchical.enabled = True
```

## Documentation

See `/docs/HIERARCHICAL_EVOLUTION.md` for complete documentation.

## Key Features

1. **Five-Layer Hierarchy**: From concrete code (L1) to abstract principles (L5)
2. **Evolutionary Memory Graph**: Rich graph structure encoding relationships
3. **7-Phase Context Compilation**: Multi-dimensional context retrieval
4. **Tiered Models**: Different models for different abstraction levels
5. **Automatic Transitions**: Plateau detection and layer escalation
6. **Insight Extraction**: Pattern recognition and knowledge compression

## Configuration

See `examples/hierarchical_evolution_config.yaml` for configuration example.
