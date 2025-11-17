"""
Tiered Model Selection System

Uses different models for different layers based on reasoning complexity:
- Tier 0: Fastest models for local search (L1)
- Tier 1: Standard models for operational tasks (L2)
- Tier 2: Stronger models for tactical decisions (L3, L4)
- Tier 3: Reasoning models for strategic pivots (L4, L5)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from openevolve.config import LLMModelConfig
from openevolve.hierarchy.layers import LayerType
from openevolve.llm.ensemble import LLMEnsemble

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers for different complexity levels"""

    TIER_0 = "tier_0"  # Fastest: haiku, gpt-3.5-turbo (L1 local search)
    TIER_1 = "tier_1"  # Standard: sonnet, gpt-4o (L2 patterns, L1 refinement)
    TIER_2 = "tier_2"  # Strong: opus, gpt-4 (L3 architecture, L4 paradigms)
    TIER_3 = "tier_3"  # Reasoning: o1, o3-mini (L4, L5 strategic pivots)


@dataclass
class TierConfig:
    """Configuration for a model tier"""

    tier: ModelTier
    models: List[LLMModelConfig] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096

    # Tier-specific parameters
    reasoning_effort: Optional[str] = None  # For reasoning models (low/medium/high)
    use_chain_of_thought: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "tier": self.tier.value,
            "models": [
                {
                    "name": m.name,
                    "weight": m.weight,
                    "api_base": m.api_base,
                }
                for m in self.models
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "use_chain_of_thought": self.use_chain_of_thought,
        }


class TieredModelSelector:
    """
    Selects appropriate model tier based on layer and evolution phase

    Usage:
        selector = TieredModelSelector(tier_configs)
        ensemble = selector.get_ensemble_for_layer(LayerType.L3_ARCHITECTURAL_COMPONENTS)
    """

    def __init__(self, tier_configs: Dict[ModelTier, TierConfig]):
        """
        Initialize tiered model selector

        Args:
            tier_configs: Dictionary mapping tiers to configurations
        """
        self.tier_configs = tier_configs

        # Create ensembles for each tier
        self.tier_ensembles: Dict[ModelTier, LLMEnsemble] = {}

        for tier, config in tier_configs.items():
            if config.models:
                self.tier_ensembles[tier] = LLMEnsemble(config.models)
                logger.info(
                    f"Initialized {tier.value} with {len(config.models)} model(s): "
                    f"{', '.join(m.name for m in config.models)}"
                )

        # Define default layer-to-tier mapping
        self.layer_tier_mapping: Dict[LayerType, ModelTier] = {
            LayerType.L1_CODE_DETAILS: ModelTier.TIER_0,
            LayerType.L2_IMPLEMENTATION_PATTERNS: ModelTier.TIER_1,
            LayerType.L3_ARCHITECTURAL_COMPONENTS: ModelTier.TIER_2,
            LayerType.L4_ALGORITHMIC_PARADIGMS: ModelTier.TIER_2,
            LayerType.L5_META_PRINCIPLES: ModelTier.TIER_3,
        }

        # Phase-aware adjustments
        self.phase_adjustments: Dict[str, Dict[LayerType, ModelTier]] = {
            "exploration": {
                # Use stronger models during exploration
                LayerType.L1_CODE_DETAILS: ModelTier.TIER_1,
                LayerType.L2_IMPLEMENTATION_PATTERNS: ModelTier.TIER_2,
            },
            "plateau": {
                # Use reasoning models when stuck
                LayerType.L2_IMPLEMENTATION_PATTERNS: ModelTier.TIER_2,
                LayerType.L3_ARCHITECTURAL_COMPONENTS: ModelTier.TIER_3,
            },
            "crisis": {
                # Use best models for crisis mode
                LayerType.L3_ARCHITECTURAL_COMPONENTS: ModelTier.TIER_3,
                LayerType.L4_ALGORITHMIC_PARADIGMS: ModelTier.TIER_3,
            },
        }

        logger.info(
            f"Initialized TieredModelSelector with {len(self.tier_ensembles)} tiers"
        )

    def get_tier_for_layer(
        self, layer_type: LayerType, phase: str = "normal"
    ) -> ModelTier:
        """
        Get the appropriate tier for a layer and evolution phase

        Args:
            layer_type: The layer being evolved
            phase: Evolution phase ("normal", "exploration", "plateau", "crisis")

        Returns:
            ModelTier to use
        """
        # Check for phase-specific adjustments
        if phase in self.phase_adjustments:
            adjusted_tier = self.phase_adjustments[phase].get(layer_type)
            if adjusted_tier and adjusted_tier in self.tier_ensembles:
                logger.debug(
                    f"Phase adjustment: {layer_type.value} → {adjusted_tier.value} (phase: {phase})"
                )
                return adjusted_tier

        # Default mapping
        tier = self.layer_tier_mapping.get(layer_type, ModelTier.TIER_1)

        # Fallback if tier not available
        if tier not in self.tier_ensembles:
            # Try to find next available tier
            tier_order = [ModelTier.TIER_0, ModelTier.TIER_1, ModelTier.TIER_2, ModelTier.TIER_3]
            for fallback_tier in tier_order:
                if fallback_tier in self.tier_ensembles:
                    logger.warning(
                        f"Tier {tier.value} not available, falling back to {fallback_tier.value}"
                    )
                    return fallback_tier

            # No tiers available
            raise ValueError("No model tiers configured")

        return tier

    def get_ensemble_for_layer(
        self, layer_type: LayerType, phase: str = "normal"
    ) -> LLMEnsemble:
        """
        Get the LLM ensemble for a specific layer

        Args:
            layer_type: The layer being evolved
            phase: Evolution phase

        Returns:
            LLMEnsemble configured for this layer
        """
        tier = self.get_tier_for_layer(layer_type, phase)
        ensemble = self.tier_ensembles.get(tier)

        if not ensemble:
            raise ValueError(f"No ensemble configured for tier {tier.value}")

        return ensemble

    def get_config_for_layer(
        self, layer_type: LayerType, phase: str = "normal"
    ) -> TierConfig:
        """
        Get the configuration for a specific layer

        Args:
            layer_type: The layer being evolved
            phase: Evolution phase

        Returns:
            TierConfig for this layer
        """
        tier = self.get_tier_for_layer(layer_type, phase)
        config = self.tier_configs.get(tier)

        if not config:
            raise ValueError(f"No config for tier {tier.value}")

        return config

    def update_layer_mapping(
        self, layer_type: LayerType, tier: ModelTier, phase: str = "normal"
    ) -> None:
        """
        Update the layer-to-tier mapping

        Args:
            layer_type: The layer to update
            tier: The new tier to use
            phase: Phase to update ("normal" for default, or specific phase)
        """
        if phase == "normal":
            self.layer_tier_mapping[layer_type] = tier
            logger.info(f"Updated default mapping: {layer_type.value} → {tier.value}")
        else:
            if phase not in self.phase_adjustments:
                self.phase_adjustments[phase] = {}
            self.phase_adjustments[phase][layer_type] = tier
            logger.info(
                f"Updated phase mapping: {layer_type.value} → {tier.value} (phase: {phase})"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tier usage"""
        return {
            "available_tiers": [tier.value for tier in self.tier_ensembles.keys()],
            "layer_mappings": {
                layer_type.value: tier.value
                for layer_type, tier in self.layer_tier_mapping.items()
            },
            "phase_adjustments": {
                phase: {
                    layer_type.value: tier.value
                    for layer_type, tier in adjustments.items()
                }
                for phase, adjustments in self.phase_adjustments.items()
            },
        }


def create_default_tier_configs(
    base_models: List[LLMModelConfig],
    tier0_models: Optional[List[LLMModelConfig]] = None,
    tier3_models: Optional[List[LLMModelConfig]] = None,
) -> Dict[ModelTier, TierConfig]:
    """
    Create default tier configurations

    Args:
        base_models: Models to use for Tier 1 and 2 (standard/strong)
        tier0_models: Optional models for Tier 0 (fast)
        tier3_models: Optional models for Tier 3 (reasoning)

    Returns:
        Dictionary of tier configurations
    """
    configs = {}

    # Tier 0: Fast models for local search
    if tier0_models:
        configs[ModelTier.TIER_0] = TierConfig(
            tier=ModelTier.TIER_0,
            models=tier0_models,
            temperature=0.8,  # Higher temperature for exploration
            max_tokens=2048,  # Shorter responses
        )
    else:
        # Use base models with adjusted parameters
        configs[ModelTier.TIER_0] = TierConfig(
            tier=ModelTier.TIER_0,
            models=base_models,
            temperature=0.8,
            max_tokens=2048,
        )

    # Tier 1: Standard models for operational tasks
    configs[ModelTier.TIER_1] = TierConfig(
        tier=ModelTier.TIER_1,
        models=base_models,
        temperature=0.7,
        max_tokens=4096,
    )

    # Tier 2: Strong models for tactical decisions
    configs[ModelTier.TIER_2] = TierConfig(
        tier=ModelTier.TIER_2,
        models=base_models,
        temperature=0.6,  # Lower temperature for more focused responses
        max_tokens=8192,
    )

    # Tier 3: Reasoning models for strategic pivots
    if tier3_models:
        configs[ModelTier.TIER_3] = TierConfig(
            tier=ModelTier.TIER_3,
            models=tier3_models,
            temperature=1.0,  # Max temperature for reasoning models
            max_tokens=16384,
            reasoning_effort="high",  # For o1/o3 models
            use_chain_of_thought=True,
        )
    else:
        # Fallback to strong models
        configs[ModelTier.TIER_3] = TierConfig(
            tier=ModelTier.TIER_3,
            models=base_models,
            temperature=0.5,  # Lower temperature for careful reasoning
            max_tokens=8192,
            use_chain_of_thought=True,
        )

    return configs
