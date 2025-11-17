"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import yaml

if TYPE_CHECKING:
    from openevolve.llm.base import LLMInterface


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # Custom LLM client
    init_client: Optional[Callable] = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None
    
    # Reasoning parameters
    reasoning_effort: Optional[str] = None


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(default_factory=list)

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None
    
    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # Only validate if this looks like a user config (has some model info)
        # Don't validate during internal/default initialization
        if (
            self.primary_model
            or self.secondary_model
            or self.primary_model_weight
            or self.secondary_model_weight
        ) and not self.models:
            raise ValueError(
                "No LLM models configured. Please specify 'models' array or "
                "'primary_model' in your configuration."
            )

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

    def rebuild_models(self) -> None:
        """Rebuild the models list after primary_model/secondary_model field changes"""
        # Clear existing models lists
        self.models = []
        self.evaluator_models = []
        
        # Re-run model generation logic from __post_init__
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = (
        500  # Suggest simplifying if program exceeds this many characters
    )
    include_changes_under_chars: Optional[int] = (
        100  # Include change descriptions in features if under this length
    )
    concise_implementation_max_lines: Optional[int] = (
        10  # Label as "concise" if program has this many lines or fewer
    )
    comprehensive_implementation_min_lines: Optional[int] = (
        50  # Label as "comprehensive" if program has this many lines or more
    )

    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = (
        None  # Deprecated: use suggest_simplification_after_chars
    )


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5
    programs_per_island: Optional[int] = None

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    # CRITICAL: For custom dimensions, evaluators must return RAW VALUES, not bin indices
    # Built-in: "complexity", "diversity", "score" (always available)
    # Custom: Any metric from your evaluator (must be continuous values)
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
        metadata={
            "help": "List of feature dimensions for MAP-Elites grid. "
            "Built-in dimensions: 'complexity', 'diversity', 'score'. "
            "Custom dimensions: Must match metric names from evaluator. "
            "IMPORTANT: Evaluators must return raw continuous values for custom dimensions, "
            "NOT pre-computed bin indices. OpenEvolve handles all scaling and binning internally."
        },
    )
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30

    novelty_llm: Optional["LLMInterface"] = None
    embedding_model: Optional[str] = None
    similarity_threshold: float = 0.99

@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program


@dataclass
class EvolutionTraceConfig:
    """Configuration for evolution trace logging"""

    enabled: bool = False
    format: str = "jsonl"  # Options: "jsonl", "json", "hdf5"
    include_code: bool = False
    include_prompts: bool = True
    output_path: Optional[str] = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical abstraction layer system"""

    # Enable hierarchical evolution
    enabled: bool = False

    # Model tier configurations
    # Tier 0: Fast models (L1 local search)
    tier0_models: List[LLMModelConfig] = field(default_factory=list)
    # Tier 1: Standard models (L2 patterns, L1 refinement)
    tier1_models: List[LLMModelConfig] = field(default_factory=list)
    # Tier 2: Strong models (L3 architecture, L4 paradigms)
    tier2_models: List[LLMModelConfig] = field(default_factory=list)
    # Tier 3: Reasoning models (L4, L5 strategic pivots)
    tier3_models: List[LLMModelConfig] = field(default_factory=list)

    # Layer transition triggers
    l2_plateau_iterations: int = 5  # L1 plateau → evolve L2
    l3_plateau_iterations: int = 20  # L2 plateau → evolve L3
    l4_plateau_iterations: int = 100  # L3 plateau → evolve L4
    l5_plateau_iterations: int = 500  # L4 plateau → evolve L5

    # Minimum improvement threshold
    min_improvement_threshold: float = 0.001

    # Insight extraction
    enable_insight_extraction: bool = True
    insight_extraction_interval: int = 100  # Extract insights every N generations
    use_llm_for_insights: bool = False  # Use LLM for deeper insight extraction

    # Context compilation
    context_max_tokens: int = 50000
    recent_weight: float = 0.7
    local_weight: float = 0.6
    success_weight: float = 0.8

    # EMG (Evolutionary Memory Graph) settings
    emg_enabled: bool = True
    emg_save_path: Optional[str] = None  # Path to save EMG (defaults to output_dir/emg)

    # Phase adjustments
    exploration_phase_threshold: float = 0.3  # Success rate below this triggers exploration
    crisis_exhausted_layers: int = 2  # Number of exhausted layers to trigger crisis


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None
    file_suffix: str = ".py"

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000

    # Early stopping settings
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator", "evolution_trace", "hierarchical"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])
        if "evolution_trace" in config_dict:
            config.evolution_trace = EvolutionTraceConfig(**config_dict["evolution_trace"])
        if "hierarchical" in config_dict:
            hierarchical_dict = config_dict["hierarchical"]
            # Parse tier model configs
            if "tier0_models" in hierarchical_dict:
                hierarchical_dict["tier0_models"] = [
                    LLMModelConfig(**m) for m in hierarchical_dict["tier0_models"]
                ]
            if "tier1_models" in hierarchical_dict:
                hierarchical_dict["tier1_models"] = [
                    LLMModelConfig(**m) for m in hierarchical_dict["tier1_models"]
                ]
            if "tier2_models" in hierarchical_dict:
                hierarchical_dict["tier2_models"] = [
                    LLMModelConfig(**m) for m in hierarchical_dict["tier2_models"]
                ]
            if "tier3_models" in hierarchical_dict:
                hierarchical_dict["tier3_models"] = [
                    LLMModelConfig(**m) for m in hierarchical_dict["tier3_models"]
                ]
            config.hierarchical = HierarchicalConfig(**hierarchical_dict)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                # Note: diversity_metric fixed to "edit_distance"
                # "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator.memory_limit_mb,
                # "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
            },
            "evolution_trace": {
                "enabled": self.evolution_trace.enabled,
                "format": self.evolution_trace.format,
                "include_code": self.evolution_trace.include_code,
                "include_prompts": self.evolution_trace.include_prompts,
                "output_path": self.evolution_trace.output_path,
                "buffer_size": self.evolution_trace.buffer_size,
                "compress": self.evolution_trace.compress,
            },
            "hierarchical": {
                "enabled": self.hierarchical.enabled,
                "tier0_models": [{"name": m.name, "weight": m.weight} for m in self.hierarchical.tier0_models],
                "tier1_models": [{"name": m.name, "weight": m.weight} for m in self.hierarchical.tier1_models],
                "tier2_models": [{"name": m.name, "weight": m.weight} for m in self.hierarchical.tier2_models],
                "tier3_models": [{"name": m.name, "weight": m.weight} for m in self.hierarchical.tier3_models],
                "l2_plateau_iterations": self.hierarchical.l2_plateau_iterations,
                "l3_plateau_iterations": self.hierarchical.l3_plateau_iterations,
                "l4_plateau_iterations": self.hierarchical.l4_plateau_iterations,
                "l5_plateau_iterations": self.hierarchical.l5_plateau_iterations,
                "min_improvement_threshold": self.hierarchical.min_improvement_threshold,
                "enable_insight_extraction": self.hierarchical.enable_insight_extraction,
                "insight_extraction_interval": self.hierarchical.insight_extraction_interval,
                "use_llm_for_insights": self.hierarchical.use_llm_for_insights,
                "context_max_tokens": self.hierarchical.context_max_tokens,
                "emg_enabled": self.hierarchical.emg_enabled,
                "emg_save_path": self.hierarchical.emg_save_path,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "max_code_length": self.max_code_length,
            # Early stopping settings
            "early_stopping_patience": self.early_stopping_patience,
            "convergence_threshold": self.convergence_threshold,
            "early_stopping_metric": self.early_stopping_metric,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def _get_api_key_for_base(api_base: str) -> Optional[str]:
    """
    Automatically detect and return the appropriate API key based on the api_base URL.

    Args:
        api_base: The API base URL

    Returns:
        API key from environment variable, or None if not found
    """
    if not api_base:
        return os.environ.get("OPENAI_API_KEY")

    api_base_lower = api_base.lower()

    # KIMI / Moonshot AI
    if "moonshot.cn" in api_base_lower:
        return os.environ.get("KIMI_API_KEY")

    # GLM / Zhipu AI
    if "bigmodel.cn" in api_base_lower or "z.ai" in api_base_lower:
        return os.environ.get("GLM_API_KEY")

    # OpenAI
    if "openai.com" in api_base_lower:
        return os.environ.get("OPENAI_API_KEY")

    # Default: try OPENAI_API_KEY as fallback
    return os.environ.get("OPENAI_API_KEY")


def _set_api_keys_for_models(models: List[LLMModelConfig]) -> None:
    """
    Automatically set API keys for models based on their api_base if not already set.

    Args:
        models: List of LLMModelConfig objects to update
    """
    for model in models:
        if model.api_key is None and model.api_base:
            detected_key = _get_api_key_for_base(model.api_base)
            if detected_key:
                model.api_key = detected_key


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Automatically set API keys for all models based on their api_base
    _set_api_keys_for_models(config.llm.models)
    _set_api_keys_for_models(config.llm.evaluator_models)

    # Also set API keys for hierarchical tier models
    if config.hierarchical.enabled:
        _set_api_keys_for_models(config.hierarchical.tier0_models)
        _set_api_keys_for_models(config.hierarchical.tier1_models)
        _set_api_keys_for_models(config.hierarchical.tier2_models)
        _set_api_keys_for_models(config.hierarchical.tier3_models)

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
