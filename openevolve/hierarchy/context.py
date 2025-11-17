"""
Context Compilation System

Implements multi-dimensional context queries and the 7-phase
context compilation algorithm for hierarchical evolution.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from openevolve.hierarchy.emg import (
    EdgeType,
    EvolutionaryMemoryGraph,
    EMGNode,
    NodeType,
)
from openevolve.hierarchy.layers import LayerType

logger = logging.getLogger(__name__)


@dataclass
class ContextQuery:
    """
    Structured query for multi-dimensional context retrieval

    Instead of a single semantic query, this specifies what dimensions matter.
    """

    focus_node_id: str
    layers: List[LayerType] = field(default_factory=list)

    # Temporal weights
    recent_weight: float = 0.7  # Favor recent over old
    window_generations: int = 100  # Last N generations
    temporal_decay: str = "exponential"  # How quickly to discount old info

    # Spatial weights
    local_weight: float = 0.6  # Same branch vs other branches
    cross_branch: bool = True  # Include other branches?
    cross_parent: bool = True  # Include cousins (different L3, same L4)?

    # Causal weights
    success_weight: float = 0.8  # Favor successes over failures
    include_failures: bool = True  # But do include failures
    causal_depth: int = 2  # How many hops in causal chain?

    # Relational flags
    include_composition_patterns: bool = True  # What works together?
    include_conflict_patterns: bool = True  # What interferes?
    include_dependency_chains: bool = True  # What requires what?

    # Constraint handling
    active_constraints_only: bool = True
    constraint_propagation_depth: int = 3  # How many layers up to check?

    # Token budget
    max_tokens: int = 50000


@dataclass
class ContextItem:
    """Single item in the context bundle"""

    content: str
    priority: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    tokens: int
    source: str  # Description of where this came from
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBundle:
    """
    Compiled context bundle ready to be used in a prompt

    Contains all gathered context organized by priority.
    """

    items: List[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False

    def add_item(self, item: ContextItem) -> None:
        """Add an item to the bundle"""
        self.items.append(item)
        self.total_tokens += item.tokens

    def get_formatted_context(self, max_tokens: int = None) -> str:
        """
        Get formatted context string with smart truncation

        Guarantees CRITICAL items always included, then fills by priority.
        """
        if max_tokens is None:
            max_tokens = float("inf")

        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_items = sorted(self.items, key=lambda x: priority_order.get(x.priority, 4))

        # Build context
        context_parts = []
        current_tokens = 0

        for item in sorted_items:
            if current_tokens + item.tokens <= max_tokens or item.priority == "CRITICAL":
                context_parts.append(f"## {item.source}\n\n{item.content}\n")
                current_tokens += item.tokens
            else:
                self.truncated = True
                break

        return "\n".join(context_parts)


class ContextCompiler:
    """
    Implements the 7-phase context compilation algorithm

    Phases:
    1. Constraint Propagation (Top-Down)
    2. Sibling Analysis (Same Level, Same Parent)
    3. Cousin Transfer (Same Level, Different Parent, Same Grandparent)
    4. Causal Chain Analysis
    5. Conflict Detection
    6. Trend Prediction
    7. Assembly & Prioritization
    """

    def __init__(self, emg: EvolutionaryMemoryGraph):
        self.emg = emg

    def compile_context(self, query: ContextQuery) -> ContextBundle:
        """
        Compile context using all 7 phases

        Args:
            query: Context query specification

        Returns:
            ContextBundle ready to be used in a prompt
        """
        bundle = ContextBundle()

        logger.info(f"Compiling context for node {query.focus_node_id}")

        # Phase 1: Constraint Propagation
        constraints = self._phase1_constraint_propagation(query)
        if constraints:
            bundle.add_item(constraints)

        # Phase 2: Sibling Analysis
        sibling_analysis = self._phase2_sibling_analysis(query)
        if sibling_analysis:
            bundle.add_item(sibling_analysis)

        # Phase 3: Cousin Transfer
        cousin_patterns = self._phase3_cousin_transfer(query)
        if cousin_patterns:
            bundle.add_item(cousin_patterns)

        # Phase 4: Causal Chain Analysis
        causal_insights = self._phase4_causal_chain_analysis(query)
        if causal_insights:
            bundle.add_item(causal_insights)

        # Phase 5: Conflict Detection
        conflicts = self._phase5_conflict_detection(query)
        if conflicts:
            bundle.add_item(conflicts)

        # Phase 6: Trend Prediction
        trajectory = self._phase6_trend_prediction(query)
        if trajectory:
            bundle.add_item(trajectory)

        # Phase 7: Assembly & Prioritization is done in get_formatted_context()

        logger.info(
            f"Compiled context: {len(bundle.items)} items, {bundle.total_tokens} tokens"
        )

        return bundle

    def _phase1_constraint_propagation(self, query: ContextQuery) -> Optional[ContextItem]:
        """
        Phase 1: Constraint Propagation (Top-Down)

        Walk up the hierarchy collecting active constraints and anti-patterns.
        """
        focus_node = self.emg.get_node(query.focus_node_id)
        if not focus_node:
            return None

        constraints = []
        anti_patterns = []

        # Walk up ancestor chain
        ancestors = self.emg.get_ancestors(
            query.focus_node_id, max_depth=query.constraint_propagation_depth
        )

        for ancestor in ancestors:
            # Get constraints from ancestor
            if hasattr(ancestor.content, "constraints"):
                constraints.extend(ancestor.content.constraints)
            if hasattr(ancestor.content, "anti_patterns"):
                anti_patterns.extend(ancestor.content.anti_patterns)

            # Check for constraint nodes
            constraint_neighbors = self.emg.get_neighbors(
                ancestor.id, edge_type=EdgeType.REQUIRES, direction="incoming"
            )
            for constraint_node in constraint_neighbors:
                if constraint_node.node_type == NodeType.CONSTRAINT:
                    constraints.append(constraint_node.description)

        if not constraints and not anti_patterns:
            return None

        # Format content
        content_parts = []

        if constraints:
            content_parts.append("**Active Constraints:**")
            for i, constraint in enumerate(constraints, 1):
                content_parts.append(f"{i}. {constraint}")

        if anti_patterns:
            content_parts.append("\n**Anti-Patterns to Avoid:**")
            for i, pattern in enumerate(anti_patterns, 1):
                content_parts.append(f"{i}. {pattern}")

        content = "\n".join(content_parts)

        return ContextItem(
            content=content,
            priority="CRITICAL",
            tokens=len(content.split()),
            source="Constraint Chain",
            metadata={"phase": 1},
        )

    def _phase2_sibling_analysis(self, query: ContextQuery) -> Optional[ContextItem]:
        """
        Phase 2: Sibling Analysis (Same Level, Same Parent)

        Analyze recent attempts at this level under same parent.
        """
        focus_node = self.emg.get_node(query.focus_node_id)
        if not focus_node or not focus_node.parent_ids:
            return None

        # Get parent
        parent_id = focus_node.parent_ids[0]
        parent_node = self.emg.get_node(parent_id)
        if not parent_node:
            return None

        # Get siblings (other children of the same parent)
        siblings = []
        for child_id in parent_node.children_ids:
            if child_id != query.focus_node_id:
                sibling = self.emg.get_node(child_id)
                if sibling and sibling.layer_type == focus_node.layer_type:
                    siblings.append(sibling)

        if not siblings:
            return None

        # Sort by recency
        siblings.sort(key=lambda n: n.iteration_found, reverse=True)

        # Analyze recent siblings
        recent_siblings = siblings[: min(20, len(siblings))]

        content_parts = ["**Recent Sibling Attempts:**\n"]

        # Group by success/failure
        successes = [s for s in recent_siblings if s.score > 0.7]
        failures = [s for s in recent_siblings if s.score <= 0.5]

        if failures:
            content_parts.append(f"**Failed Approaches ({len(failures)}):**")
            for failure in failures[:5]:  # Top 5 failures
                content_parts.append(f"- {failure.description} (score: {failure.score:.3f})")

                # Get failure reason
                failed_because = self.emg.get_neighbors(
                    failure.id, edge_type=EdgeType.FAILED_BECAUSE, direction="outgoing"
                )
                if failed_because:
                    content_parts.append(f"  Reason: {failed_because[0].description}")

        if successes:
            content_parts.append(f"\n**Successful Approaches ({len(successes)}):**")
            for success in successes[:3]:  # Top 3 successes
                content_parts.append(f"- {success.description} (score: {success.score:.3f})")

        # Detect convergence
        if len(recent_siblings) >= 5:
            # Simple convergence check: are recent siblings similar?
            recent_scores = [s.score for s in recent_siblings[:5]]
            score_variance = sum((s - sum(recent_scores) / len(recent_scores)) ** 2 for s in recent_scores) / len(recent_scores)

            if score_variance < 0.01:
                content_parts.append("\n⚠️ **Convergence Detected**: Recent attempts show little variation")

        content = "\n".join(content_parts)

        return ContextItem(
            content=content,
            priority="HIGH",
            tokens=len(content.split()),
            source="Sibling Analysis",
            metadata={"phase": 2, "siblings_analyzed": len(recent_siblings)},
        )

    def _phase3_cousin_transfer(self, query: ContextQuery) -> Optional[ContextItem]:
        """
        Phase 3: Cousin Transfer (Same Level, Different Parent, Same Grandparent)

        Find successful solutions in parallel branches.
        """
        focus_node = self.emg.get_node(query.focus_node_id)
        if not focus_node or not focus_node.parent_ids:
            return None

        # Get parent and grandparent
        parent_node = self.emg.get_node(focus_node.parent_ids[0])
        if not parent_node or not parent_node.parent_ids:
            return None

        grandparent_id = parent_node.parent_ids[0]
        grandparent_node = self.emg.get_node(grandparent_id)
        if not grandparent_node:
            return None

        # Get all children of grandparent (uncles/aunts)
        uncle_ids = [
            child_id
            for child_id in grandparent_node.children_ids
            if child_id != focus_node.parent_ids[0]
        ]

        # Get cousins (children of uncles)
        cousins = []
        for uncle_id in uncle_ids:
            uncle_node = self.emg.get_node(uncle_id)
            if not uncle_node:
                continue

            for cousin_id in uncle_node.children_ids:
                cousin = self.emg.get_node(cousin_id)
                if cousin and cousin.layer_type == focus_node.layer_type and cousin.score > 0.7:
                    cousins.append((cousin, uncle_node))

        if not cousins:
            return None

        # Sort by score
        cousins.sort(key=lambda x: x[0].score, reverse=True)

        content_parts = ["**Transferable Patterns from Parallel Branches:**\n"]

        for cousin, uncle in cousins[:5]:  # Top 5 cousins
            content_parts.append(
                f"- **{cousin.description}** (score: {cousin.score:.3f})"
            )
            content_parts.append(f"  Context: Under {uncle.description}")

            # Extract what made this successful
            succeeded_because = self.emg.get_neighbors(
                cousin.id, edge_type=EdgeType.SUCCEEDED_BECAUSE, direction="outgoing"
            )
            if succeeded_because:
                content_parts.append(f"  Key: {succeeded_because[0].description}")

        content = "\n".join(content_parts)

        return ContextItem(
            content=content,
            priority="MEDIUM",
            tokens=len(content.split()),
            source="Cousin Transfer",
            metadata={"phase": 3, "cousins_found": len(cousins)},
        )

    def _phase4_causal_chain_analysis(self, query: ContextQuery) -> Optional[ContextItem]:
        """
        Phase 4: Causal Chain Analysis

        Trace backwards from major breakthroughs to understand what led to success.
        """
        # Find breakthrough nodes (high score improvements)
        breakthroughs = self.emg.query_nodes(
            node_type=NodeType.SOLUTION, min_score=0.9, max_results=10
        )

        if not breakthroughs:
            return None

        content_parts = ["**Insights from Past Breakthroughs:**\n"]

        for breakthrough in breakthroughs[:3]:  # Top 3 breakthroughs
            # Get causal chain
            chain = self.emg.get_causal_chain(breakthrough.id, max_depth=query.causal_depth)

            if len(chain) > 1:
                content_parts.append(f"\n**Breakthrough: {breakthrough.description}**")
                content_parts.append("Causal chain:")

                for node, reason in chain[:5]:  # Limit chain length
                    content_parts.append(f"  ← {reason}: {node.description}")

        content = "\n".join(content_parts)

        return ContextItem(
            content=content,
            priority="MEDIUM",
            tokens=len(content.split()),
            source="Causal Chain Analysis",
            metadata={"phase": 4, "breakthroughs_analyzed": len(breakthroughs)},
        )

    def _phase5_conflict_detection(self, query: ContextQuery) -> Optional[ContextItem]:
        """
        Phase 5: Conflict Detection

        Identify combinations that consistently fail.
        """
        # Find contradiction edges
        conflict_edge_ids = self.emg.edges_by_type.get(EdgeType.CONTRADICTS, set())

        if not conflict_edge_ids:
            return None

        conflicts = []
        for edge_id in conflict_edge_ids:
            edge = self.emg.get_edge(edge_id)
            if not edge:
                continue

            source = self.emg.get_node(edge.source_id)
            target = self.emg.get_node(edge.target_id)

            if source and target:
                conflicts.append((source.description, target.description))

        if not conflicts:
            return None

        content_parts = ["**Known Conflicts to Avoid:**\n"]

        for source_desc, target_desc in conflicts[:10]:  # Top 10 conflicts
            content_parts.append(f"- ❌ **{source_desc}** + **{target_desc}**")

        content = "\n".join(content_parts)

        return ContextItem(
            content=content,
            priority="HIGH",
            tokens=len(content.split()),
            source="Conflict Detection",
            metadata={"phase": 5, "conflicts_found": len(conflicts)},
        )

    def _phase6_trend_prediction(self, query: ContextQuery) -> Optional[ContextItem]:
        """
        Phase 6: Trend Prediction

        Analyze trajectory to predict what's likely needed next.
        """
        focus_node = self.emg.get_node(query.focus_node_id)
        if not focus_node:
            return None

        # Get recent nodes at the same layer
        recent_nodes = self.emg.query_nodes(
            layer_type=focus_node.layer_type,
            generation_window=(
                max(0, focus_node.generation - query.window_generations),
                focus_node.generation,
            ),
        )

        if len(recent_nodes) < 5:
            return None

        # Analyze trend
        scores = [node.score for node in recent_nodes]
        generations = [node.generation for node in recent_nodes]

        # Simple linear trend
        if len(scores) >= 2:
            avg_improvement = (scores[-1] - scores[0]) / max(1, generations[-1] - generations[0])

            content_parts = ["**Evolution Trajectory:**\n"]

            if avg_improvement > 0.01:
                content_parts.append(f"✅ **Improving** (avg: +{avg_improvement:.4f} per generation)")
                content_parts.append("Recommendation: Continue current approach")
            elif avg_improvement > -0.001:
                content_parts.append("⚠️ **Plateauing** (minimal improvement)")
                content_parts.append("Recommendation: Consider exploring alternative patterns")
            else:
                content_parts.append(f"📉 **Declining** (avg: {avg_improvement:.4f} per generation)")
                content_parts.append("Recommendation: Escalate to higher layer")

            # Check velocity (recent vs older)
            if len(scores) >= 10:
                recent_avg = sum(scores[-5:]) / 5
                older_avg = sum(scores[:5]) / 5
                acceleration = recent_avg - older_avg

                if abs(acceleration) > 0.05:
                    content_parts.append(
                        f"\nAcceleration: {acceleration:+.4f} (recent vs older generations)"
                    )

            content = "\n".join(content_parts)

            return ContextItem(
                content=content,
                priority="LOW",
                tokens=len(content.split()),
                source="Trend Prediction",
                metadata={
                    "phase": 6,
                    "avg_improvement": avg_improvement,
                    "nodes_analyzed": len(recent_nodes),
                },
            )

        return None
