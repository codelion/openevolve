"""
Layer Abstraction Definitions for Hierarchical Evolution

Implements the five-layer abstraction hierarchy where each layer
represents a different level of abstraction in code evolution.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LayerType(Enum):
    """Types of abstraction layers in the hierarchy"""

    L1_CODE_DETAILS = "code_details"  # Concrete code, hyperparameters, values
    L2_IMPLEMENTATION_PATTERNS = "implementation_patterns"  # Algorithms, data structures
    L3_ARCHITECTURAL_COMPONENTS = "architectural_components"  # Structures and relationships
    L4_ALGORITHMIC_PARADIGMS = "algorithmic_paradigms"  # Mathematical strategies
    L5_META_PRINCIPLES = "meta_principles"  # Fundamental philosophical approaches


@dataclass
class Layer:
    """
    Base class for abstraction layers

    Each layer maintains:
    - Content: The actual representation at this abstraction level
    - Metadata: Evolution history, performance, constraints
    - Lineage: Parent and children relationships
    """

    id: str
    layer_type: LayerType
    content: str
    description: str

    # Performance metrics
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Lineage tracking
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Evolution metadata
    generation: int = 0
    iteration_found: int = 0
    timestamp: float = field(default_factory=time.time)

    # Constraints and guidance
    constraints: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)

    # History
    attempts_count: int = 0  # Number of mutations tried at this layer
    success_count: int = 0  # Number of successful improvements
    last_improvement_iteration: int = 0
    plateau_iterations: int = 0  # Iterations since last improvement

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "layer_type": self.layer_type.value,
            "content": self.content,
            "description": self.description,
            "score": self.score,
            "metrics": self.metrics,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "generation": self.generation,
            "iteration_found": self.iteration_found,
            "timestamp": self.timestamp,
            "constraints": self.constraints,
            "anti_patterns": self.anti_patterns,
            "attempts_count": self.attempts_count,
            "success_count": self.success_count,
            "last_improvement_iteration": self.last_improvement_iteration,
            "plateau_iterations": self.plateau_iterations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Layer":
        """Create from dictionary representation"""
        data = data.copy()
        data["layer_type"] = LayerType(data["layer_type"])
        return cls(**data)


@dataclass
class L1CodeDetails(Layer):
    """
    Layer 1: Code Details (Most Concrete)

    Represents actual code, hyperparameters, and specific values.
    This is what currently gets evolved in systems like AlphaEvolve.

    Examples:
    - learning_rate = 0.023
    - batch_size = 64
    - Exact variable names, loop structures
    """

    code: str = ""  # The actual code
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.layer_type != LayerType.L1_CODE_DETAILS:
            self.layer_type = LayerType.L1_CODE_DETAILS


@dataclass
class L2ImplementationPatterns(Layer):
    """
    Layer 2: Implementation Patterns (Operational)

    Represents specific algorithms, data structures, and code patterns.

    Examples:
    - "Use Adam optimizer with learning rate schedule"
    - "Implement memoization with LRU cache"
    - "Store sparse matrices in CSR format"
    """

    patterns: List[str] = field(default_factory=list)
    data_structures: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.layer_type != LayerType.L2_IMPLEMENTATION_PATTERNS:
            self.layer_type = LayerType.L2_IMPLEMENTATION_PATTERNS


@dataclass
class L3ArchitecturalComponents(Layer):
    """
    Layer 3: Architectural Components (Tactical)

    Represents concrete structures and their relationships.

    Examples:
    - "Represent solutions as permutation matrices"
    - "State space: (position, resources_remaining, time)"
    - "Define loss function based on orbit equivalence"
    """

    components: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.layer_type != LayerType.L3_ARCHITECTURAL_COMPONENTS:
            self.layer_type = LayerType.L3_ARCHITECTURAL_COMPONENTS


@dataclass
class L4AlgorithmicParadigms(Layer):
    """
    Layer 4: Algorithmic Paradigms (Strategic)

    Represents specific mathematical/computational strategies.

    Examples:
    - "Use group theory to identify symmetries"
    - "Apply dynamic programming on discrete states"
    - "Use gradient descent with continuous relaxation"
    """

    paradigm_name: str = ""
    mathematical_framework: str = ""
    key_assumptions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.layer_type != LayerType.L4_ALGORITHMIC_PARADIGMS:
            self.layer_type = LayerType.L4_ALGORITHMIC_PARADIGMS


@dataclass
class L5MetaPrinciples(Layer):
    """
    Layer 5: Meta-Principles (Most Abstract)

    Represents fundamental philosophical approaches and frameworks.

    Examples:
    - "Exploit problem symmetry"
    - "Transform continuous to discrete"
    - "Decompose into independent subproblems"
    - "Use probabilistic vs deterministic methods"
    """

    principle_name: str = ""
    philosophy: str = ""
    applicable_domains: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.layer_type != LayerType.L5_META_PRINCIPLES:
            self.layer_type = LayerType.L5_META_PRINCIPLES


@dataclass
class HierarchicalProgram:
    """
    A complete program with representations at all abstraction layers

    This links together the different layer representations to form
    a coherent hierarchical structure.
    """

    id: str

    # Layer representations (from concrete to abstract)
    l1_code: Optional[L1CodeDetails] = None
    l2_patterns: Optional[L2ImplementationPatterns] = None
    l3_architecture: Optional[L3ArchitecturalComponents] = None
    l4_paradigm: Optional[L4AlgorithmicParadigms] = None
    l5_principle: Optional[L5MetaPrinciples] = None

    # Overall metrics
    combined_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Evolution tracking
    generation: int = 0
    iteration_found: int = 0
    timestamp: float = field(default_factory=time.time)

    # Active layer (which layer is currently being evolved)
    active_layer: LayerType = LayerType.L1_CODE_DETAILS

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_layer(self, layer_type: LayerType) -> Optional[Layer]:
        """Get the layer representation for a specific layer type"""
        layer_map = {
            LayerType.L1_CODE_DETAILS: self.l1_code,
            LayerType.L2_IMPLEMENTATION_PATTERNS: self.l2_patterns,
            LayerType.L3_ARCHITECTURAL_COMPONENTS: self.l3_architecture,
            LayerType.L4_ALGORITHMIC_PARADIGMS: self.l4_paradigm,
            LayerType.L5_META_PRINCIPLES: self.l5_principle,
        }
        return layer_map.get(layer_type)

    def set_layer(self, layer: Layer) -> None:
        """Set the layer representation for a specific layer type"""
        layer_map = {
            LayerType.L1_CODE_DETAILS: "l1_code",
            LayerType.L2_IMPLEMENTATION_PATTERNS: "l2_patterns",
            LayerType.L3_ARCHITECTURAL_COMPONENTS: "l3_architecture",
            LayerType.L4_ALGORITHMIC_PARADIGMS: "l4_paradigm",
            LayerType.L5_META_PRINCIPLES: "l5_principle",
        }
        attr_name = layer_map.get(layer.layer_type)
        if attr_name:
            setattr(self, attr_name, layer)

    def get_constraint_chain(self) -> List[str]:
        """
        Get the full constraint chain from top to bottom

        Returns constraints from L5 → L4 → L3 → L2 → L1
        """
        constraints = []
        for layer_type in [
            LayerType.L5_META_PRINCIPLES,
            LayerType.L4_ALGORITHMIC_PARADIGMS,
            LayerType.L3_ARCHITECTURAL_COMPONENTS,
            LayerType.L2_IMPLEMENTATION_PATTERNS,
            LayerType.L1_CODE_DETAILS,
        ]:
            layer = self.get_layer(layer_type)
            if layer:
                constraints.extend(layer.constraints)
        return constraints

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "l1_code": self.l1_code.to_dict() if self.l1_code else None,
            "l2_patterns": self.l2_patterns.to_dict() if self.l2_patterns else None,
            "l3_architecture": self.l3_architecture.to_dict() if self.l3_architecture else None,
            "l4_paradigm": self.l4_paradigm.to_dict() if self.l4_paradigm else None,
            "l5_principle": self.l5_principle.to_dict() if self.l5_principle else None,
            "combined_score": self.combined_score,
            "metrics": self.metrics,
            "generation": self.generation,
            "iteration_found": self.iteration_found,
            "timestamp": self.timestamp,
            "active_layer": self.active_layer.value,
            "metadata": self.metadata,
        }
