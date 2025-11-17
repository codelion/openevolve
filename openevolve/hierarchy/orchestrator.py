"""
Hierarchical Evolution Orchestrator

Integrates the hierarchical abstraction layer system with the existing
OpenEvolve controller and process-parallel execution.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import uuid

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.hierarchy import (
    EvolutionaryMemoryGraph,
    ContextCompiler,
    ContextQuery,
    TieredModelSelector,
    LayerTransitionManager,
    InsightExtractor,
    TransitionTriggers,
    HierarchicalProgram,
    L1CodeDetails,
    EMGNode,
    EMGEdge,
    NodeType,
    EdgeType,
    LayerType,
)
from openevolve.hierarchy.model_tiers import create_default_tier_configs, ModelTier, TierConfig
from openevolve.llm.ensemble import LLMEnsemble

logger = logging.getLogger(__name__)


class HierarchicalOrchestrator:
    """
    Orchestrates hierarchical evolution

    Integrates with the existing OpenEvolve controller to add:
    - Layer-based evolution (L1-L5)
    - Evolutionary Memory Graph tracking
    - Multi-dimensional context compilation
    - Tiered model selection
    - Plateau detection and layer transitions
    - Insight extraction
    """

    def __init__(
        self,
        config: Config,
        database: ProgramDatabase,
        llm_ensemble: LLMEnsemble,
        output_dir: str = None,
    ):
        """
        Initialize hierarchical orchestrator

        Args:
            config: OpenEvolve configuration with hierarchical settings
            database: Program database
            llm_ensemble: LLM ensemble (used as fallback for tier configs)
            output_dir: Output directory for EMG and insights
        """
        self.config = config
        self.database = database
        self.llm_ensemble = llm_ensemble
        self.output_dir = output_dir or "."

        # Check if hierarchical evolution is enabled
        if not config.hierarchical.enabled:
            logger.info("Hierarchical evolution is disabled")
            self.enabled = False
            return

        self.enabled = True
        logger.info("Initializing Hierarchical Evolution Orchestrator")

        # Initialize Evolutionary Memory Graph
        self.emg = EvolutionaryMemoryGraph() if config.hierarchical.emg_enabled else None

        # Initialize Context Compiler
        self.context_compiler = ContextCompiler(self.emg) if self.emg else None

        # Initialize Layer Transition Manager
        triggers = TransitionTriggers(
            l2_plateau_iterations=config.hierarchical.l2_plateau_iterations,
            l3_plateau_iterations=config.hierarchical.l3_plateau_iterations,
            l4_plateau_iterations=config.hierarchical.l4_plateau_iterations,
            l5_plateau_iterations=config.hierarchical.l5_plateau_iterations,
            min_improvement_threshold=config.hierarchical.min_improvement_threshold,
        )
        self.transition_manager = LayerTransitionManager(triggers)

        # Initialize Tiered Model Selector
        self.model_selector = self._initialize_model_selector()

        # Initialize Insight Extractor
        insight_llm = self.llm_ensemble if config.hierarchical.use_llm_for_insights else None
        self.insight_extractor = (
            InsightExtractor(self.emg, insight_llm)
            if config.hierarchical.enable_insight_extraction and self.emg
            else None
        )

        # Track programs in hierarchical representation
        self.hierarchical_programs: Dict[str, HierarchicalProgram] = {}

        # Current generation for insight extraction
        self.current_generation = 0

        # Best scores for plateau detection
        self.best_scores_by_layer: Dict[LayerType, float] = {
            layer: 0.0 for layer in LayerType
        }

        logger.info("Hierarchical Evolution Orchestrator initialized")
        logger.info(f"EMG enabled: {self.emg is not None}")
        logger.info(f"Insight extraction enabled: {self.insight_extractor is not None}")
        logger.info(f"Model tiers configured: {len(self.model_selector.tier_ensembles)}")

    def _initialize_model_selector(self) -> TieredModelSelector:
        """Initialize tiered model selector from config"""
        # Create tier configurations
        tier_configs = {}

        # Build tier configs from hierarchical config
        hconfig = self.config.hierarchical

        if hconfig.tier0_models:
            tier_configs[ModelTier.TIER_0] = TierConfig(
                tier=ModelTier.TIER_0,
                models=hconfig.tier0_models,
                temperature=0.8,
                max_tokens=2048,
            )

        if hconfig.tier1_models:
            tier_configs[ModelTier.TIER_1] = TierConfig(
                tier=ModelTier.TIER_1,
                models=hconfig.tier1_models,
                temperature=0.7,
                max_tokens=4096,
            )

        if hconfig.tier2_models:
            tier_configs[ModelTier.TIER_2] = TierConfig(
                tier=ModelTier.TIER_2,
                models=hconfig.tier2_models,
                temperature=0.6,
                max_tokens=8192,
            )

        if hconfig.tier3_models:
            tier_configs[ModelTier.TIER_3] = TierConfig(
                tier=ModelTier.TIER_3,
                models=hconfig.tier3_models,
                temperature=0.5,
                max_tokens=16384,
                use_chain_of_thought=True,
            )

        # Fallback: use default tier configs if none specified
        if not tier_configs:
            logger.warning("No tier models configured, using default configuration")
            tier_configs = create_default_tier_configs(
                base_models=self.config.llm.models,
                tier0_models=self.config.llm.models,
                tier3_models=self.config.llm.models,
            )

        return TieredModelSelector(tier_configs)

    def get_ensemble_for_iteration(self, iteration: int) -> LLMEnsemble:
        """
        Get the appropriate LLM ensemble for the current iteration

        Based on:
        - Current evolution phase
        - Active layer being evolved
        - Plateau status

        Args:
            iteration: Current iteration number

        Returns:
            LLMEnsemble to use for this iteration
        """
        if not self.enabled:
            return self.llm_ensemble

        # Determine active layer
        active_layer = self.transition_manager.get_active_layer(iteration)

        # Determine phase
        phase = self.transition_manager.get_evolution_phase(iteration)

        # Get ensemble for this layer and phase
        try:
            ensemble = self.model_selector.get_ensemble_for_layer(
                active_layer, phase=phase.value
            )
            logger.debug(
                f"Using {active_layer.value} ensemble in {phase.value} phase "
                f"(iteration {iteration})"
            )
            return ensemble
        except Exception as e:
            logger.warning(f"Failed to get tiered ensemble: {e}, using default")
            return self.llm_ensemble

    def enhance_prompt_context(
        self, base_prompt: Dict[str, str], parent_program: Program, iteration: int
    ) -> Dict[str, str]:
        """
        Enhance the base prompt with hierarchical context

        Args:
            base_prompt: Base prompt from PromptSampler
            parent_program: Parent program being evolved
            iteration: Current iteration

        Returns:
            Enhanced prompt with hierarchical context
        """
        if not self.enabled or not self.context_compiler:
            return base_prompt

        try:
            # Create or get hierarchical program representation
            if parent_program.id not in self.hierarchical_programs:
                self.hierarchical_programs[parent_program.id] = HierarchicalProgram(
                    id=parent_program.id,
                    l1_code=L1CodeDetails(
                        id=f"l1_{parent_program.id}",
                        layer_type=LayerType.L1_CODE_DETAILS,
                        content=parent_program.code,
                        description="Code implementation",
                        code=parent_program.code,
                    ),
                    generation=parent_program.generation,
                    iteration_found=parent_program.iteration_found,
                    combined_score=parent_program.metrics.get("combined_score", 0.0),
                )

            # Find corresponding EMG node
            emg_node_id = f"sol_{parent_program.id}"
            emg_node = self.emg.get_node(emg_node_id)

            if not emg_node:
                # Node doesn't exist yet, just return base prompt
                return base_prompt

            # Create context query
            query = ContextQuery(
                focus_node_id=emg_node_id,
                layers=[LayerType.L1_CODE_DETAILS, LayerType.L2_IMPLEMENTATION_PATTERNS],
                recent_weight=self.config.hierarchical.recent_weight,
                local_weight=self.config.hierarchical.local_weight,
                success_weight=self.config.hierarchical.success_weight,
                max_tokens=self.config.hierarchical.context_max_tokens,
            )

            # Compile context
            context_bundle = self.context_compiler.compile_context(query)

            # Get formatted context
            hierarchical_context = context_bundle.get_formatted_context(
                max_tokens=self.config.hierarchical.context_max_tokens
            )

            # Enhance system message with hierarchical context
            enhanced_system = base_prompt["system"]
            if hierarchical_context:
                enhanced_system += f"\n\n# Hierarchical Evolution Context\n\n{hierarchical_context}"

            return {
                "system": enhanced_system,
                "user": base_prompt["user"],
            }

        except Exception as e:
            logger.warning(f"Failed to enhance prompt with hierarchical context: {e}")
            return base_prompt

    def record_iteration_result(
        self,
        iteration: int,
        parent: Program,
        child: Optional[Program],
        success: bool,
        score_improvement: float = 0.0,
    ) -> None:
        """
        Record the result of an iteration

        Updates:
        - EMG with new nodes and edges
        - Layer transition manager with scores
        - Hierarchical program representations

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program (None if failed)
            success: Whether the iteration was successful
            score_improvement: Score improvement achieved
        """
        if not self.enabled:
            return

        try:
            # Update generation counter
            self.current_generation = max(self.current_generation, iteration)

            # Get active layer
            active_layer = self.transition_manager.get_active_layer(iteration)

            # Get current and previous best scores
            current_score = child.metrics.get("combined_score", 0.0) if child else 0.0
            previous_best = self.best_scores_by_layer[active_layer]

            # Update layer status
            improved = self.transition_manager.update_layer_status(
                active_layer, current_score, previous_best, iteration
            )

            if improved:
                self.best_scores_by_layer[active_layer] = current_score

            # Add to EMG if enabled
            if self.emg:
                self._add_to_emg(iteration, parent, child, success, active_layer)

            # Extract insights periodically
            if self.insight_extractor and self.insight_extractor.should_extract(
                self.current_generation
            ):
                logger.info(f"Extracting insights at generation {self.current_generation}")
                insights = self.insight_extractor.extract_insights()
                logger.info(f"Extracted {len(insights)} insights")

        except Exception as e:
            logger.error(f"Error recording iteration result: {e}")

    def _add_to_emg(
        self,
        iteration: int,
        parent: Program,
        child: Optional[Program],
        success: bool,
        layer: LayerType,
    ) -> None:
        """Add iteration result to EMG"""
        # Add parent node if not exists
        parent_node_id = f"sol_{parent.id}"
        if not self.emg.get_node(parent_node_id):
            parent_node = EMGNode(
                id=parent_node_id,
                node_type=NodeType.SOLUTION,
                layer_type=layer,
                content=parent,
                description=f"Program {parent.id[:8]}",
                score=parent.metrics.get("combined_score", 0.0),
                metrics=parent.metrics,
                generation=parent.generation,
                iteration_found=parent.iteration_found,
            )
            self.emg.add_node(parent_node)

        if child and success:
            # Add child node
            child_node_id = f"sol_{child.id}"
            child_node = EMGNode(
                id=child_node_id,
                node_type=NodeType.SOLUTION,
                layer_type=layer,
                content=child,
                description=f"Program {child.id[:8]}",
                score=child.metrics.get("combined_score", 0.0),
                metrics=child.metrics,
                generation=child.generation,
                iteration_found=child.iteration_found,
                parent_ids=[parent_node_id],
            )
            self.emg.add_node(child_node)

            # Add parent-child edge
            edge = EMGEdge(
                id=str(uuid.uuid4()),
                edge_type=EdgeType.PARENT_OF,
                source_id=parent_node_id,
                target_id=child_node_id,
                weight=1.0,
            )
            self.emg.add_edge(edge)

            # Add success causal edge if significant improvement
            if child.metrics.get("combined_score", 0) > parent.metrics.get("combined_score", 0) + 0.05:
                # Create insight node for the improvement
                insight_id = f"insight_{iteration}_{uuid.uuid4().hex[:8]}"
                insight_node = EMGNode(
                    id=insight_id,
                    node_type=NodeType.INSIGHT,
                    description=f"Significant improvement: {child.metrics.get('combined_score', 0):.4f}",
                    generation=iteration,
                )
                self.emg.add_node(insight_node)

                # Link child to insight
                success_edge = EMGEdge(
                    id=str(uuid.uuid4()),
                    edge_type=EdgeType.SUCCEEDED_BECAUSE,
                    source_id=child_node_id,
                    target_id=insight_id,
                )
                self.emg.add_edge(success_edge)

        elif not success:
            # Add failure node
            failure_id = f"fail_{iteration}_{uuid.uuid4().hex[:8]}"
            failure_node = EMGNode(
                id=failure_id,
                node_type=NodeType.FAILURE,
                layer_type=layer,
                description="Evolution attempt failed",
                generation=iteration,
            )
            self.emg.add_node(failure_node)

            # Link parent to failure
            fail_edge = EMGEdge(
                id=str(uuid.uuid4()),
                edge_type=EdgeType.FAILED_BECAUSE,
                source_id=parent_node_id,
                target_id=failure_id,
            )
            self.emg.add_edge(fail_edge)

    def save_state(self, checkpoint_dir: str) -> None:
        """
        Save hierarchical evolution state

        Args:
            checkpoint_dir: Directory to save state
        """
        if not self.enabled:
            return

        try:
            import json
            import pickle

            # Save EMG
            if self.emg:
                emg_path = os.path.join(checkpoint_dir, "emg.pkl")
                with open(emg_path, "wb") as f:
                    pickle.dump(self.emg.to_dict(), f)
                logger.info(f"Saved EMG to {emg_path}")

            # Save transition manager state
            transition_path = os.path.join(checkpoint_dir, "layer_transitions.json")
            with open(transition_path, "w") as f:
                json.dump(self.transition_manager.get_statistics(), f, indent=2)
            logger.info(f"Saved transition state to {transition_path}")

            # Save insights
            if self.insight_extractor:
                insights_path = os.path.join(checkpoint_dir, "insights.json")
                with open(insights_path, "w") as f:
                    insights_data = {
                        insight_id: insight.to_dict()
                        for insight_id, insight in self.insight_extractor.insights.items()
                    }
                    json.dump(insights_data, f, indent=2)
                logger.info(f"Saved {len(insights_data)} insights to {insights_path}")

        except Exception as e:
            logger.error(f"Error saving hierarchical state: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about hierarchical evolution"""
        if not self.enabled:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "current_generation": self.current_generation,
            "transition_manager": self.transition_manager.get_statistics(),
            "model_tiers": self.model_selector.get_statistics(),
        }

        if self.emg:
            stats["emg"] = self.emg.get_statistics()

        if self.insight_extractor:
            stats["insights_count"] = len(self.insight_extractor.insights)

        return stats
