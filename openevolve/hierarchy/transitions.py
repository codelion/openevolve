"""
Layer Transition Manager

Detects plateaus and triggers transitions between abstraction layers.
Manages exploration/exploitation phases across the hierarchy.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openevolve.hierarchy.layers import LayerType, HierarchicalProgram, Layer

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """Evolution phases for different exploration/exploitation strategies"""

    NORMAL = "normal"  # Standard evolution (95% of time)
    EXPLORATION = "exploration"  # Active exploration (4% of time)
    PLATEAU = "plateau"  # Detected plateau, seeking breakthrough
    CRISIS = "crisis"  # Multiple plateaus, major pivot needed (1% of time)


@dataclass
class LayerStatus:
    """Status of evolution at a specific layer"""

    layer_type: LayerType
    current_score: float = 0.0
    best_score: float = 0.0
    attempts_count: int = 0
    success_count: int = 0
    last_improvement_iteration: int = 0
    plateau_iterations: int = 0  # Iterations since last improvement

    # Thresholds for transition
    plateau_threshold: int = 10  # Iterations without improvement
    exhaustion_threshold: int = 50  # Consider layer exhausted

    def is_plateaued(self) -> bool:
        """Check if this layer has plateaued"""
        return self.plateau_iterations >= self.plateau_threshold

    def is_exhausted(self) -> bool:
        """Check if this layer is exhausted"""
        return self.plateau_iterations >= self.exhaustion_threshold

    def update(self, score: float, iteration: int, improved: bool = False) -> None:
        """Update layer status"""
        self.current_score = score
        self.attempts_count += 1

        if improved:
            self.success_count += 1
            self.best_score = max(self.best_score, score)
            self.last_improvement_iteration = iteration
            self.plateau_iterations = 0
        else:
            self.plateau_iterations += 1

    def get_success_rate(self) -> float:
        """Get success rate for this layer"""
        if self.attempts_count == 0:
            return 0.0
        return self.success_count / self.attempts_count


@dataclass
class TransitionTriggers:
    """Configuration for layer transition triggers"""

    # Layer 1 (Code) - evolves every generation
    l1_always: bool = True

    # Layer 2 (Patterns) - when Layer 1 plateaus
    l2_plateau_iterations: int = 5

    # Layer 3 (Architecture) - when Layer 2 exhausted
    l3_plateau_iterations: int = 20

    # Layer 4 (Paradigm) - when Layer 3 exhausted
    l4_plateau_iterations: int = 100

    # Layer 5 (Meta-Principle) - when Layer 4 exhausted
    l5_plateau_iterations: int = 500

    # Improvement thresholds
    min_improvement_threshold: float = 0.001  # Minimum score improvement to count


class LayerTransitionManager:
    """
    Manages transitions between abstraction layers

    Determines:
    - When to evolve each layer
    - What phase the evolution is in
    - When to escalate to higher layers
    """

    def __init__(self, triggers: Optional[TransitionTriggers] = None):
        """
        Initialize transition manager

        Args:
            triggers: Configuration for transition triggers
        """
        self.triggers = triggers or TransitionTriggers()

        # Track status for each layer
        self.layer_status: Dict[LayerType, LayerStatus] = {
            LayerType.L1_CODE_DETAILS: LayerStatus(
                LayerType.L1_CODE_DETAILS, plateau_threshold=self.triggers.l2_plateau_iterations
            ),
            LayerType.L2_IMPLEMENTATION_PATTERNS: LayerStatus(
                LayerType.L2_IMPLEMENTATION_PATTERNS,
                plateau_threshold=self.triggers.l3_plateau_iterations,
            ),
            LayerType.L3_ARCHITECTURAL_COMPONENTS: LayerStatus(
                LayerType.L3_ARCHITECTURAL_COMPONENTS,
                plateau_threshold=self.triggers.l4_plateau_iterations,
            ),
            LayerType.L4_ALGORITHMIC_PARADIGMS: LayerStatus(
                LayerType.L4_ALGORITHMIC_PARADIGMS,
                plateau_threshold=self.triggers.l5_plateau_iterations,
            ),
            LayerType.L5_META_PRINCIPLES: LayerStatus(
                LayerType.L5_META_PRINCIPLES, plateau_threshold=1000  # Rarely changes
            ),
        }

        # Current evolution phase
        self.current_phase = EvolutionPhase.NORMAL

        # Statistics
        self.phase_history: List[Tuple[int, EvolutionPhase]] = []
        self.layer_transition_history: List[Tuple[int, LayerType, str]] = []

        logger.info("Initialized LayerTransitionManager")

    def should_evolve_layer(self, layer_type: LayerType, iteration: int) -> bool:
        """
        Determine if a specific layer should be evolved at this iteration

        Args:
            layer_type: The layer to check
            iteration: Current iteration number

        Returns:
            True if this layer should be evolved
        """
        status = self.layer_status[layer_type]

        # L1 always evolves
        if layer_type == LayerType.L1_CODE_DETAILS:
            return True

        # L2 evolves when L1 plateaus
        if layer_type == LayerType.L2_IMPLEMENTATION_PATTERNS:
            l1_status = self.layer_status[LayerType.L1_CODE_DETAILS]
            return l1_status.is_plateaued()

        # L3 evolves when L2 exhausted
        if layer_type == LayerType.L3_ARCHITECTURAL_COMPONENTS:
            l2_status = self.layer_status[LayerType.L2_IMPLEMENTATION_PATTERNS]
            return l2_status.is_exhausted()

        # L4 evolves when L3 exhausted
        if layer_type == LayerType.L4_ALGORITHMIC_PARADIGMS:
            l3_status = self.layer_status[LayerType.L3_ARCHITECTURAL_COMPONENTS]
            return l3_status.is_exhausted()

        # L5 evolves when L4 exhausted (crisis mode)
        if layer_type == LayerType.L5_META_PRINCIPLES:
            l4_status = self.layer_status[LayerType.L4_ALGORITHMIC_PARADIGMS]
            return l4_status.is_exhausted()

        return False

    def get_active_layer(self, iteration: int) -> LayerType:
        """
        Get the highest layer that should be actively evolved

        Args:
            iteration: Current iteration number

        Returns:
            The layer type to evolve
        """
        # Check from highest to lowest
        for layer_type in [
            LayerType.L5_META_PRINCIPLES,
            LayerType.L4_ALGORITHMIC_PARADIGMS,
            LayerType.L3_ARCHITECTURAL_COMPONENTS,
            LayerType.L2_IMPLEMENTATION_PATTERNS,
            LayerType.L1_CODE_DETAILS,
        ]:
            if self.should_evolve_layer(layer_type, iteration):
                return layer_type

        # Default to L1
        return LayerType.L1_CODE_DETAILS

    def update_layer_status(
        self,
        layer_type: LayerType,
        score: float,
        previous_best: float,
        iteration: int,
    ) -> bool:
        """
        Update status for a layer after an evolution attempt

        Args:
            layer_type: Layer that was evolved
            score: New score achieved
            previous_best: Previous best score
            iteration: Current iteration

        Returns:
            True if this was an improvement
        """
        status = self.layer_status[layer_type]

        # Check if this is an improvement
        improved = score > previous_best + self.triggers.min_improvement_threshold

        # Update status
        status.update(score, iteration, improved)

        if improved:
            logger.info(
                f"Layer {layer_type.value} improved: {previous_best:.4f} → {score:.4f} "
                f"(+{score - previous_best:.4f})"
            )
        elif status.is_plateaued():
            logger.warning(
                f"Layer {layer_type.value} plateaued: {status.plateau_iterations} iterations "
                f"without improvement"
            )

        return improved

    def get_evolution_phase(self, iteration: int) -> EvolutionPhase:
        """
        Determine current evolution phase

        Args:
            iteration: Current iteration

        Returns:
            Current evolution phase
        """
        # Check for crisis mode (multiple layers exhausted)
        exhausted_count = sum(
            1 for status in self.layer_status.values() if status.is_exhausted()
        )

        if exhausted_count >= 2:
            new_phase = EvolutionPhase.CRISIS
        elif self.layer_status[LayerType.L3_ARCHITECTURAL_COMPONENTS].is_plateaued():
            new_phase = EvolutionPhase.PLATEAU
        elif self.layer_status[LayerType.L1_CODE_DETAILS].is_plateaued():
            new_phase = EvolutionPhase.EXPLORATION
        else:
            new_phase = EvolutionPhase.NORMAL

        # Log phase changes
        if new_phase != self.current_phase:
            logger.info(
                f"Evolution phase changed: {self.current_phase.value} → {new_phase.value} "
                f"(iteration {iteration})"
            )
            self.phase_history.append((iteration, new_phase))
            self.current_phase = new_phase

        return self.current_phase

    def get_exploration_exploitation_ratio(self) -> Tuple[float, float]:
        """
        Get exploration/exploitation ratios based on current phase

        Returns:
            (exploration_ratio, exploitation_ratio) tuple
        """
        phase_ratios = {
            EvolutionPhase.NORMAL: (0.2, 0.7),  # Normal: 20% explore, 70% exploit
            EvolutionPhase.EXPLORATION: (0.5, 0.4),  # Exploration: 50% explore, 40% exploit
            EvolutionPhase.PLATEAU: (0.6, 0.3),  # Plateau: 60% explore, 30% exploit
            EvolutionPhase.CRISIS: (0.7, 0.2),  # Crisis: 70% explore, 20% exploit
        }

        return phase_ratios.get(self.current_phase, (0.2, 0.7))

    def suggest_layer_change(
        self, current_layer: LayerType, iteration: int
    ) -> Optional[LayerType]:
        """
        Suggest changing to a different layer if beneficial

        Args:
            current_layer: Currently active layer
            iteration: Current iteration

        Returns:
            Suggested layer to switch to, or None to continue current layer
        """
        status = self.layer_status[current_layer]

        # If current layer is exhausted, escalate
        if status.is_exhausted():
            next_layer_map = {
                LayerType.L1_CODE_DETAILS: LayerType.L2_IMPLEMENTATION_PATTERNS,
                LayerType.L2_IMPLEMENTATION_PATTERNS: LayerType.L3_ARCHITECTURAL_COMPONENTS,
                LayerType.L3_ARCHITECTURAL_COMPONENTS: LayerType.L4_ALGORITHMIC_PARADIGMS,
                LayerType.L4_ALGORITHMIC_PARADIGMS: LayerType.L5_META_PRINCIPLES,
            }

            next_layer = next_layer_map.get(current_layer)
            if next_layer:
                logger.warning(
                    f"Layer {current_layer.value} exhausted, suggesting escalation to "
                    f"{next_layer.value}"
                )
                self.layer_transition_history.append((iteration, next_layer, "escalation"))
                return next_layer

        # If current layer is doing well, continue
        if status.get_success_rate() > 0.3:  # 30% success rate
            return None

        # If lower layers are performing better, de-escalate
        if current_layer != LayerType.L1_CODE_DETAILS:
            prev_layer_map = {
                LayerType.L5_META_PRINCIPLES: LayerType.L4_ALGORITHMIC_PARADIGMS,
                LayerType.L4_ALGORITHMIC_PARADIGMS: LayerType.L3_ARCHITECTURAL_COMPONENTS,
                LayerType.L3_ARCHITECTURAL_COMPONENTS: LayerType.L2_IMPLEMENTATION_PATTERNS,
                LayerType.L2_IMPLEMENTATION_PATTERNS: LayerType.L1_CODE_DETAILS,
            }

            prev_layer = prev_layer_map.get(current_layer)
            if prev_layer:
                prev_status = self.layer_status[prev_layer]

                # If previous layer has better success rate, de-escalate
                if prev_status.get_success_rate() > status.get_success_rate() + 0.1:
                    logger.info(
                        f"De-escalating from {current_layer.value} to {prev_layer.value} "
                        f"(better performance)"
                    )
                    self.layer_transition_history.append((iteration, prev_layer, "de-escalation"))
                    return prev_layer

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about layer transitions and phases"""
        return {
            "current_phase": self.current_phase.value,
            "layer_status": {
                layer_type.value: {
                    "current_score": status.current_score,
                    "best_score": status.best_score,
                    "attempts": status.attempts_count,
                    "successes": status.success_count,
                    "success_rate": status.get_success_rate(),
                    "plateau_iterations": status.plateau_iterations,
                    "is_plateaued": status.is_plateaued(),
                    "is_exhausted": status.is_exhausted(),
                }
                for layer_type, status in self.layer_status.items()
            },
            "phase_history": [
                {"iteration": it, "phase": phase.value}
                for it, phase in self.phase_history[-10:]  # Last 10 phase changes
            ],
            "layer_transitions": [
                {"iteration": it, "layer": layer.value, "reason": reason}
                for it, layer, reason in self.layer_transition_history[-10:]  # Last 10
            ],
        }

    def reset_layer(self, layer_type: LayerType) -> None:
        """
        Reset a layer's status (e.g., after a successful change)

        Args:
            layer_type: Layer to reset
        """
        status = self.layer_status[layer_type]
        status.plateau_iterations = 0
        status.attempts_count = 0
        status.success_count = 0

        logger.info(f"Reset layer {layer_type.value} status")
