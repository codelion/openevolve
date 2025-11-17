"""
Hierarchical Abstraction Layer System for OpenEvolve

This module implements a five-layer abstraction hierarchy for breakthrough creativity:
- L5: Meta-Principles (most abstract)
- L4: Algorithmic Paradigms (strategic)
- L3: Architectural Components (tactical)
- L2: Implementation Patterns (operational)
- L1: Code Details (concrete)

Each layer evolves at different frequencies with specialized models and context.
"""

from openevolve.hierarchy.layers import (
    Layer,
    LayerType,
    L1CodeDetails,
    L2ImplementationPatterns,
    L3ArchitecturalComponents,
    L4AlgorithmicParadigms,
    L5MetaPrinciples,
    HierarchicalProgram,
)

from openevolve.hierarchy.emg import (
    EMGNode,
    EMGEdge,
    EdgeType,
    EvolutionaryMemoryGraph,
)

from openevolve.hierarchy.context import (
    ContextQuery,
    ContextCompiler,
    ContextBundle,
)

from openevolve.hierarchy.model_tiers import (
    ModelTier,
    TieredModelSelector,
)

from openevolve.hierarchy.transitions import (
    LayerTransitionManager,
    EvolutionPhase,
)

from openevolve.hierarchy.insights import (
    Insight,
    InsightExtractor,
)

from openevolve.hierarchy.orchestrator import (
    HierarchicalOrchestrator,
)

__all__ = [
    # Layers
    "Layer",
    "LayerType",
    "L1CodeDetails",
    "L2ImplementationPatterns",
    "L3ArchitecturalComponents",
    "L4AlgorithmicParadigms",
    "L5MetaPrinciples",
    "HierarchicalProgram",
    # EMG
    "EMGNode",
    "EMGEdge",
    "EdgeType",
    "NodeType",
    "EvolutionaryMemoryGraph",
    # Context
    "ContextQuery",
    "ContextCompiler",
    "ContextBundle",
    # Model Tiers
    "ModelTier",
    "TieredModelSelector",
    # Transitions
    "LayerTransitionManager",
    "EvolutionPhase",
    "TransitionTriggers",
    # Insights
    "Insight",
    "InsightExtractor",
    # Orchestrator
    "HierarchicalOrchestrator",
]
