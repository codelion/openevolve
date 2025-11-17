"""
Evolutionary Memory Graph (EMG)

A rich graph structure that explicitly encodes relationships between
solutions, mutations, failures, insights, and constraints across all layers.

This replaces simple semantic search with structured multi-dimensional context retrieval.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from openevolve.hierarchy.layers import Layer, LayerType, HierarchicalProgram

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the EMG"""

    SOLUTION = "solution"  # A solution at a specific layer
    MUTATION = "mutation"  # A transition between solutions
    FAILURE = "failure"  # A failed attempt with diagnostic info
    INSIGHT = "insight"  # Extracted pattern or learning
    CONSTRAINT = "constraint"  # Active requirement
    ANALOGY = "analogy"  # Cross-domain analogy


class EdgeType(Enum):
    """Types of edges in the EMG"""

    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    ENABLES = "enables"
    REQUIRES = "requires"
    INSPIRED_BY = "inspired_by"
    FAILED_BECAUSE = "failed_because"
    SUCCEEDED_BECAUSE = "succeeded_because"
    EXTRACTED_FROM = "extracted_from"
    APPLIES_TO = "applies_to"


@dataclass
class EMGNode:
    """Node in the Evolutionary Memory Graph"""

    id: str
    node_type: NodeType
    layer_type: Optional[LayerType] = None

    # Content
    content: Any = None  # Can be Layer, Program, Insight, etc.
    description: str = ""

    # Metrics and scoring
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Temporal information
    generation: int = 0
    iteration_found: int = 0
    timestamp: float = field(default_factory=time.time)

    # Graph connectivity (stored as node IDs)
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Evidence and confidence (for insights)
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "layer_type": self.layer_type.value if self.layer_type else None,
            "content": str(self.content),  # Simplified serialization
            "description": self.description,
            "score": self.score,
            "metrics": self.metrics,
            "generation": self.generation,
            "iteration_found": self.iteration_found,
            "timestamp": self.timestamp,
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
        }


@dataclass
class EMGEdge:
    """Edge in the Evolutionary Memory Graph"""

    id: str
    edge_type: EdgeType
    source_id: str
    target_id: str

    # Edge weight (for similarity, importance, etc.)
    weight: float = 1.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "edge_type": self.edge_type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class EvolutionaryMemoryGraph:
    """
    Main graph database for hierarchical evolution

    Stores nodes (solutions, insights, failures, constraints) and
    edges (relationships) to enable multi-dimensional context queries.
    """

    def __init__(self):
        # Node storage
        self.nodes: Dict[str, EMGNode] = {}

        # Edge storage (indexed by edge type for fast queries)
        self.edges: Dict[str, EMGEdge] = {}
        self.edges_by_type: Dict[EdgeType, Set[str]] = {
            edge_type: set() for edge_type in EdgeType
        }

        # Adjacency lists for fast graph traversal
        self.outgoing_edges: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        self.incoming_edges: Dict[str, List[str]] = {}  # node_id -> [edge_ids]

        # Layer-specific indexes
        self.nodes_by_layer: Dict[LayerType, Set[str]] = {
            layer_type: set() for layer_type in LayerType
        }

        # Type-specific indexes
        self.nodes_by_type: Dict[NodeType, Set[str]] = {
            node_type: set() for node_type in NodeType
        }

        # Temporal index (for recent queries)
        self.nodes_by_generation: Dict[int, Set[str]] = {}

        # High-value nodes (breakthroughs, key insights)
        self.high_value_nodes: Set[str] = set()

        logger.info("Initialized Evolutionary Memory Graph")

    def add_node(self, node: EMGNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.id] = node

        # Update indexes
        if node.layer_type:
            self.nodes_by_layer[node.layer_type].add(node.id)

        self.nodes_by_type[node.node_type].add(node.id)

        if node.generation not in self.nodes_by_generation:
            self.nodes_by_generation[node.generation] = set()
        self.nodes_by_generation[node.generation].add(node.id)

        # Mark high-value nodes
        if node.score > 0.9 or node.node_type == NodeType.INSIGHT:
            self.high_value_nodes.add(node.id)

        # Initialize adjacency lists
        if node.id not in self.outgoing_edges:
            self.outgoing_edges[node.id] = []
        if node.id not in self.incoming_edges:
            self.incoming_edges[node.id] = []

        logger.debug(f"Added node {node.id} (type: {node.node_type.value})")
        return node.id

    def add_edge(self, edge: EMGEdge) -> str:
        """Add an edge to the graph"""
        self.edges[edge.id] = edge
        self.edges_by_type[edge.edge_type].add(edge.id)

        # Update adjacency lists
        if edge.source_id not in self.outgoing_edges:
            self.outgoing_edges[edge.source_id] = []
        self.outgoing_edges[edge.source_id].append(edge.id)

        if edge.target_id not in self.incoming_edges:
            self.incoming_edges[edge.target_id] = []
        self.incoming_edges[edge.target_id].append(edge.id)

        # Update node parent/children lists
        if edge.edge_type == EdgeType.PARENT_OF:
            source_node = self.nodes.get(edge.source_id)
            if source_node and edge.target_id not in source_node.children_ids:
                source_node.children_ids.append(edge.target_id)

            target_node = self.nodes.get(edge.target_id)
            if target_node and edge.source_id not in target_node.parent_ids:
                target_node.parent_ids.append(edge.source_id)

        logger.debug(
            f"Added edge {edge.id} ({edge.edge_type.value}): {edge.source_id} -> {edge.target_id}"
        )
        return edge.id

    def get_node(self, node_id: str) -> Optional[EMGNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[EMGEdge]:
        """Get an edge by ID"""
        return self.edges.get(edge_id)

    def get_neighbors(
        self, node_id: str, edge_type: Optional[EdgeType] = None, direction: str = "outgoing"
    ) -> List[EMGNode]:
        """
        Get neighboring nodes

        Args:
            node_id: Source node ID
            edge_type: Filter by edge type (None for all types)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of neighboring nodes
        """
        edge_ids = []

        if direction in ["outgoing", "both"]:
            edge_ids.extend(self.outgoing_edges.get(node_id, []))

        if direction in ["incoming", "both"]:
            edge_ids.extend(self.incoming_edges.get(node_id, []))

        neighbors = []
        for edge_id in edge_ids:
            edge = self.edges.get(edge_id)
            if not edge:
                continue

            # Filter by edge type if specified
            if edge_type and edge.edge_type != edge_type:
                continue

            # Get the neighbor node
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor = self.nodes.get(neighbor_id)
            if neighbor:
                neighbors.append(neighbor)

        return neighbors

    def get_ancestors(
        self, node_id: str, max_depth: int = None, layer_type: Optional[LayerType] = None
    ) -> List[EMGNode]:
        """
        Get ancestor nodes (parents, grandparents, etc.)

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse (None for unlimited)
            layer_type: Filter by layer type

        Returns:
            List of ancestor nodes
        """
        ancestors = []
        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)

            # Skip the starting node
            if current_id != node_id:
                current_node = self.nodes.get(current_id)
                if current_node:
                    if layer_type is None or current_node.layer_type == layer_type:
                        ancestors.append(current_node)

            # Stop if max depth reached
            if max_depth is not None and depth >= max_depth:
                continue

            # Add parents to queue
            current_node = self.nodes.get(current_id)
            if current_node:
                for parent_id in current_node.parent_ids:
                    if parent_id not in visited:
                        queue.append((parent_id, depth + 1))

        return ancestors

    def get_descendants(
        self, node_id: str, max_depth: int = None, layer_type: Optional[LayerType] = None
    ) -> List[EMGNode]:
        """
        Get descendant nodes (children, grandchildren, etc.)

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse (None for unlimited)
            layer_type: Filter by layer type

        Returns:
            List of descendant nodes
        """
        descendants = []
        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)

            # Skip the starting node
            if current_id != node_id:
                current_node = self.nodes.get(current_id)
                if current_node:
                    if layer_type is None or current_node.layer_type == layer_type:
                        descendants.append(current_node)

            # Stop if max depth reached
            if max_depth is not None and depth >= max_depth:
                continue

            # Add children to queue
            current_node = self.nodes.get(current_id)
            if current_node:
                for child_id in current_node.children_ids:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))

        return descendants

    def query_nodes(
        self,
        node_type: Optional[NodeType] = None,
        layer_type: Optional[LayerType] = None,
        min_score: float = 0.0,
        max_results: int = None,
        generation_window: Optional[Tuple[int, int]] = None,
    ) -> List[EMGNode]:
        """
        Query nodes with filters

        Args:
            node_type: Filter by node type
            layer_type: Filter by layer type
            min_score: Minimum score threshold
            max_results: Maximum number of results
            generation_window: (min_gen, max_gen) tuple for temporal filtering

        Returns:
            List of matching nodes
        """
        # Start with all nodes
        candidate_ids = set(self.nodes.keys())

        # Apply filters
        if node_type:
            candidate_ids &= self.nodes_by_type.get(node_type, set())

        if layer_type:
            candidate_ids &= self.nodes_by_layer.get(layer_type, set())

        if generation_window:
            min_gen, max_gen = generation_window
            gen_ids = set()
            for gen in range(min_gen, max_gen + 1):
                gen_ids |= self.nodes_by_generation.get(gen, set())
            candidate_ids &= gen_ids

        # Filter by score and sort
        results = []
        for node_id in candidate_ids:
            node = self.nodes.get(node_id)
            if node and node.score >= min_score:
                results.append(node)

        # Sort by score (descending)
        results.sort(key=lambda n: n.score, reverse=True)

        # Limit results
        if max_results:
            results = results[:max_results]

        return results

    def get_causal_chain(self, node_id: str, max_depth: int = 5) -> List[Tuple[EMGNode, str]]:
        """
        Trace causal chain backwards from a node

        Returns list of (node, reason) tuples explaining how we got to this node.

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to trace

        Returns:
            List of (node, reason) tuples
        """
        chain = []
        visited = set()
        current_id = node_id
        depth = 0

        while current_id and depth < max_depth:
            if current_id in visited:
                break

            visited.add(current_id)
            current_node = self.nodes.get(current_id)

            if not current_node:
                break

            # Find causal edges
            reason = "Unknown"
            next_id = None

            for edge_id in self.incoming_edges.get(current_id, []):
                edge = self.edges.get(edge_id)
                if not edge:
                    continue

                if edge.edge_type in [
                    EdgeType.SUCCEEDED_BECAUSE,
                    EdgeType.ENABLED,
                    EdgeType.INSPIRED_BY,
                ]:
                    reason = edge.edge_type.value
                    next_id = edge.source_id
                    break
                elif edge.edge_type == EdgeType.PARENT_OF:
                    reason = "evolved_from"
                    next_id = edge.source_id

            chain.append((current_node, reason))
            current_id = next_id
            depth += 1

        return chain

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {
                node_type.value: len(node_ids)
                for node_type, node_ids in self.nodes_by_type.items()
            },
            "nodes_by_layer": {
                layer_type.value: len(node_ids)
                for layer_type, node_ids in self.nodes_by_layer.items()
            },
            "high_value_nodes": len(self.high_value_nodes),
            "generations": len(self.nodes_by_generation),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to dictionary (for serialization)"""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            "statistics": self.get_statistics(),
        }
