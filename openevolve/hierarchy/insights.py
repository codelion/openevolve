"""
Insight Extraction and Compression

Periodically analyzes evolution history to extract generalizable patterns,
compress knowledge, and create high-level insights that guide future evolution.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from openevolve.hierarchy.emg import (
    EMGNode,
    EdgeType,
    EvolutionaryMemoryGraph,
    NodeType,
)
from openevolve.hierarchy.layers import LayerType

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """
    A compressed piece of knowledge extracted from evolution history

    Insights are first-class citizens in the EMG, retrieved alongside solutions.
    """

    id: str
    content: str  # The actual insight
    insight_type: str  # "performance_pattern", "failure_pattern", "breakthrough_pattern", etc.

    # Evidence
    evidence_node_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0

    # Applicability
    applicable_layers: List[LayerType] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)

    # Metadata
    generation_created: int = 0
    generation_window: tuple = (0, 0)  # (start, end) of analyzed generations
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "insight_type": self.insight_type,
            "evidence_node_ids": self.evidence_node_ids,
            "confidence": self.confidence,
            "applicable_layers": [layer.value for layer in self.applicable_layers],
            "applicable_contexts": self.applicable_contexts,
            "generation_created": self.generation_created,
            "generation_window": self.generation_window,
            "metadata": self.metadata,
        }


class InsightExtractor:
    """
    Extracts insights from evolution history

    Runs periodically (e.g., every 100 generations) to:
    1. Analyze patterns in successful/failed solutions
    2. Extract generalizable knowledge
    3. Compress and summarize evolution history
    4. Create insight nodes in the EMG
    """

    def __init__(
        self,
        emg: EvolutionaryMemoryGraph,
        llm_ensemble=None,  # Optional LLM for meta-analysis
    ):
        """
        Initialize insight extractor

        Args:
            emg: Evolutionary Memory Graph
            llm_ensemble: Optional LLM ensemble for insight generation
        """
        self.emg = emg
        self.llm_ensemble = llm_ensemble

        # Track when insights were last extracted
        self.last_extraction_generation = 0
        self.extraction_interval = 100  # Extract insights every N generations

        # Insight storage
        self.insights: Dict[str, Insight] = {}

        logger.info("Initialized InsightExtractor")

    def should_extract(self, current_generation: int) -> bool:
        """
        Determine if insights should be extracted at this generation

        Args:
            current_generation: Current generation number

        Returns:
            True if insights should be extracted
        """
        return (
            current_generation - self.last_extraction_generation
            >= self.extraction_interval
        )

    def extract_insights(
        self, generation_window: Optional[tuple] = None
    ) -> List[Insight]:
        """
        Extract insights from a generation window

        Args:
            generation_window: (start, end) generations to analyze (None for last 100)

        Returns:
            List of extracted insights
        """
        if generation_window is None:
            # Analyze last 100 generations
            max_gen = max(
                (node.generation for node in self.emg.nodes.values()), default=0
            )
            generation_window = (max(0, max_gen - 100), max_gen)

        start_gen, end_gen = generation_window

        logger.info(f"Extracting insights from generations {start_gen} to {end_gen}")

        insights = []

        # Extract different types of insights
        insights.extend(self._extract_performance_patterns(generation_window))
        insights.extend(self._extract_failure_patterns(generation_window))
        insights.extend(self._extract_breakthrough_patterns(generation_window))
        insights.extend(self._extract_combination_patterns(generation_window))

        # Store insights
        for insight in insights:
            self.insights[insight.id] = insight

            # Add to EMG as insight node
            insight_node = EMGNode(
                id=insight.id,
                node_type=NodeType.INSIGHT,
                content=insight,
                description=insight.content,
                confidence=insight.confidence,
                evidence_ids=insight.evidence_node_ids,
                generation=end_gen,
                metadata={"insight_type": insight.insight_type},
            )
            self.emg.add_node(insight_node)

            # Link to evidence nodes
            for evidence_id in insight.evidence_node_ids:
                from openevolve.hierarchy.emg import EMGEdge
                import uuid

                edge = EMGEdge(
                    id=str(uuid.uuid4()),
                    edge_type=EdgeType.EXTRACTED_FROM,
                    source_id=insight.id,
                    target_id=evidence_id,
                )
                self.emg.add_edge(edge)

        self.last_extraction_generation = end_gen

        logger.info(f"Extracted {len(insights)} insights")

        return insights

    def _extract_performance_patterns(
        self, generation_window: tuple
    ) -> List[Insight]:
        """Extract patterns that correlate with high performance"""
        insights = []

        start_gen, end_gen = generation_window

        # Get high-performing solutions in this window
        high_performers = self.emg.query_nodes(
            node_type=NodeType.SOLUTION,
            min_score=0.8,
            generation_window=generation_window,
        )

        if len(high_performers) < 3:
            return insights

        # Analyze common characteristics
        # This is a simplified version - in practice, you'd use LLM for deeper analysis

        # Example: Check if certain layer types perform better
        layer_scores = {}
        for node in high_performers:
            if node.layer_type:
                if node.layer_type not in layer_scores:
                    layer_scores[node.layer_type] = []
                layer_scores[node.layer_type].append(node.score)

        for layer_type, scores in layer_scores.items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)

                insight = Insight(
                    id=f"insight_perf_{layer_type.value}_{end_gen}",
                    content=f"Layer {layer_type.value} solutions achieve average score of {avg_score:.3f} "
                    f"based on {len(scores)} high-performing instances",
                    insight_type="performance_pattern",
                    evidence_node_ids=[node.id for node in high_performers if node.layer_type == layer_type],
                    confidence=min(0.9, len(scores) / 10.0),  # Higher confidence with more evidence
                    applicable_layers=[layer_type],
                    generation_created=end_gen,
                    generation_window=generation_window,
                )
                insights.append(insight)

        return insights

    def _extract_failure_patterns(self, generation_window: tuple) -> List[Insight]:
        """Extract patterns that correlate with failures"""
        insights = []

        start_gen, end_gen = generation_window

        # Get failed solutions
        failures = self.emg.query_nodes(
            node_type=NodeType.FAILURE,
            generation_window=generation_window,
        )

        if len(failures) < 3:
            return insights

        # Group failures by description patterns
        failure_types: Dict[str, List[EMGNode]] = {}
        for failure in failures:
            # Simple grouping by first 50 chars of description
            key = failure.description[:50] if failure.description else "unknown"
            if key not in failure_types:
                failure_types[key] = []
            failure_types[key].append(failure)

        # Create insights for common failure patterns
        for pattern, failure_nodes in failure_types.items():
            if len(failure_nodes) >= 3:
                insight = Insight(
                    id=f"insight_fail_{hash(pattern) % 10000}_{end_gen}",
                    content=f"Common failure pattern: '{pattern}' occurred {len(failure_nodes)} times",
                    insight_type="failure_pattern",
                    evidence_node_ids=[node.id for node in failure_nodes],
                    confidence=min(0.95, len(failure_nodes) / 10.0),
                    applicable_layers=[LayerType.L1_CODE_DETAILS, LayerType.L2_IMPLEMENTATION_PATTERNS],
                    generation_created=end_gen,
                    generation_window=generation_window,
                )
                insights.append(insight)

        return insights

    def _extract_breakthrough_patterns(
        self, generation_window: tuple
    ) -> List[Insight]:
        """Extract patterns that led to breakthroughs"""
        insights = []

        start_gen, end_gen = generation_window

        # Get solutions with significant improvements
        all_solutions = self.emg.query_nodes(
            node_type=NodeType.SOLUTION,
            generation_window=generation_window,
        )

        # Find breakthroughs (significant score jumps)
        breakthroughs = []
        for node in all_solutions:
            if node.parent_ids:
                parent = self.emg.get_node(node.parent_ids[0])
                if parent and node.score > parent.score + 0.1:  # 0.1+ improvement
                    breakthroughs.append((node, parent))

        if len(breakthroughs) < 2:
            return insights

        # Analyze what led to breakthroughs
        breakthrough_layers = {}
        for child, parent in breakthroughs:
            if child.layer_type:
                if child.layer_type not in breakthrough_layers:
                    breakthrough_layers[child.layer_type] = []
                breakthrough_layers[child.layer_type].append((child, parent))

        for layer_type, breakthrough_list in breakthrough_layers.items():
            if len(breakthrough_list) >= 2:
                avg_improvement = sum(
                    child.score - parent.score for child, parent in breakthrough_list
                ) / len(breakthrough_list)

                insight = Insight(
                    id=f"insight_breakthrough_{layer_type.value}_{end_gen}",
                    content=f"Breakthroughs at layer {layer_type.value}: {len(breakthrough_list)} instances "
                    f"with average improvement of {avg_improvement:.3f}",
                    insight_type="breakthrough_pattern",
                    evidence_node_ids=[child.id for child, _ in breakthrough_list],
                    confidence=min(0.9, len(breakthrough_list) / 5.0),
                    applicable_layers=[layer_type],
                    generation_created=end_gen,
                    generation_window=generation_window,
                    metadata={"avg_improvement": avg_improvement},
                )
                insights.append(insight)

        return insights

    def _extract_combination_patterns(
        self, generation_window: tuple
    ) -> List[Insight]:
        """Extract patterns about what works well together"""
        insights = []

        # Find nodes that enable each other
        enables_edges = self.emg.edges_by_type.get(EdgeType.ENABLES, set())

        if len(enables_edges) < 3:
            return insights

        # Analyze successful combinations
        successful_combinations: Dict[str, List[str]] = {}

        for edge_id in enables_edges:
            edge = self.emg.get_edge(edge_id)
            if not edge:
                continue

            source = self.emg.get_node(edge.source_id)
            target = self.emg.get_node(edge.target_id)

            if source and target and target.score > 0.7:
                key = f"{source.description[:30]}+{target.description[:30]}"
                if key not in successful_combinations:
                    successful_combinations[key] = []
                successful_combinations[key].append(target.id)

        # Create insights for successful combinations
        for combo_desc, target_ids in successful_combinations.items():
            if len(target_ids) >= 2:
                insight = Insight(
                    id=f"insight_combo_{hash(combo_desc) % 10000}",
                    content=f"Successful combination pattern: {combo_desc} led to "
                    f"{len(target_ids)} high-performing solutions",
                    insight_type="combination_pattern",
                    evidence_node_ids=target_ids,
                    confidence=min(0.85, len(target_ids) / 5.0),
                    applicable_layers=[LayerType.L2_IMPLEMENTATION_PATTERNS, LayerType.L3_ARCHITECTURAL_COMPONENTS],
                    generation_created=generation_window[1],
                    generation_window=generation_window,
                )
                insights.append(insight)

        return insights

    async def extract_insights_with_llm(
        self, generation_window: tuple
    ) -> List[Insight]:
        """
        Use LLM to extract deeper insights from evolution history

        This provides more sophisticated pattern recognition than rule-based extraction.

        Args:
            generation_window: (start, end) generations to analyze

        Returns:
            List of LLM-generated insights
        """
        if not self.llm_ensemble:
            logger.warning("No LLM ensemble configured for insight extraction")
            return []

        start_gen, end_gen = generation_window

        # Gather context about this generation window
        solutions = self.emg.query_nodes(
            node_type=NodeType.SOLUTION,
            generation_window=generation_window,
            max_results=50,
        )

        if len(solutions) < 5:
            return []

        # Build prompt for LLM
        context_parts = ["Analyze the following evolution history and extract key insights:\n"]

        for solution in solutions[:20]:  # Limit to avoid token overflow
            context_parts.append(
                f"- Gen {solution.generation}: {solution.description} "
                f"(score: {solution.score:.3f}, layer: {solution.layer_type.value if solution.layer_type else 'unknown'})"
            )

        context = "\n".join(context_parts)

        prompt = f"""{context}

Based on this evolution history, extract 3-5 key insights:
1. What patterns led to high performance?
2. What patterns led to failures?
3. What combinations work well together?
4. What does this tell us about the problem structure?
5. What should we try next?

Format each insight as:
INSIGHT: [one-sentence insight]
CONFIDENCE: [0.0-1.0]
APPLICABLE_LAYERS: [comma-separated layer types]
"""

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message="You are an expert at analyzing evolutionary algorithms and extracting patterns.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response into insights
            # This is simplified - you'd want more robust parsing
            insights = self._parse_llm_insights(response, generation_window)

            logger.info(f"Extracted {len(insights)} insights using LLM")

            return insights

        except Exception as e:
            logger.error(f"Error extracting insights with LLM: {e}")
            return []

    def _parse_llm_insights(self, response: str, generation_window: tuple) -> List[Insight]:
        """Parse LLM response into Insight objects"""
        insights = []

        # Simple parsing - split by "INSIGHT:"
        parts = response.split("INSIGHT:")

        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            lines = part.strip().split("\n")
            if not lines:
                continue

            content = lines[0].strip()

            # Extract confidence if present
            confidence = 0.7  # Default
            for line in lines[1:]:
                if line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except:
                        pass

            # Extract applicable layers if present
            applicable_layers = [LayerType.L1_CODE_DETAILS]  # Default
            for line in lines[1:]:
                if line.startswith("APPLICABLE_LAYERS:"):
                    layer_str = line.split(":")[1].strip()
                    # Parse layer types
                    # This is simplified

            insight = Insight(
                id=f"insight_llm_{i}_{generation_window[1]}",
                content=content,
                insight_type="llm_generated",
                confidence=confidence,
                applicable_layers=applicable_layers,
                generation_created=generation_window[1],
                generation_window=generation_window,
            )
            insights.append(insight)

        return insights

    def get_relevant_insights(
        self, layer_type: LayerType, max_results: int = 5
    ) -> List[Insight]:
        """
        Get the most relevant insights for a specific layer

        Args:
            layer_type: Layer to get insights for
            max_results: Maximum number of insights to return

        Returns:
            List of relevant insights sorted by confidence
        """
        relevant = [
            insight
            for insight in self.insights.values()
            if layer_type in insight.applicable_layers
        ]

        # Sort by confidence
        relevant.sort(key=lambda x: x.confidence, reverse=True)

        return relevant[:max_results]
