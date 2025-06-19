"""
Prompt sampling for OpenEvolve
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(config.template_dir)

        # Initialize the random number generator
        random.seed()

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        logger.info("Initialized prompt sampler")

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        language: str = "python",
        evolution_round: int = 0,
        allow_full_rewrite: bool = False,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs
            language: Programming language
            evolution_round: Current evolution round
            allow_full_rewrite: Whether to allow a full rewrite
            template_key: Optional override for template key
            program_artifacts: Optional artifacts from program evaluation
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on whether we want a full rewrite (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        else:
            # Default behavior
            user_template_key = "full_rewrite_user" if allow_full_rewrite else "diff_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )

        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, language
        )

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            **kwargs,
        )

        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for the prompt using safe formatting"""
        # Use safe formatting to handle mixed numeric and string values
        formatted_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """Identify potential areas for improvement"""
        # This method could be expanded to include more sophisticated analysis
        # For now, we'll use a simple approach

        improvement_areas = []

        # Check program length
        if len(current_program) > 500:
            improvement_areas.append(
                "Consider simplifying the code to improve readability and maintainability"
            )

        # Check for performance patterns in previous attempts
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]
            metrics_improved = []
            metrics_regressed = []

            for metric, value in metrics.items():
                # Only compare numeric metrics
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue

                improved = True
                regressed = True

                for attempt in recent_attempts:
                    attempt_value = attempt["metrics"].get(metric, 0)
                    # Only compare if both values are numeric
                    if isinstance(value, (int, float)) and isinstance(attempt_value, (int, float)):
                        if attempt_value <= value:
                            regressed = False
                        if attempt_value >= value:
                            improved = False
                    else:
                        # If either value is non-numeric, skip comparison
                        improved = False
                        regressed = False

                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)

            if metrics_improved:
                improvement_areas.append(
                    f"Metrics showing improvement: {', '.join(metrics_improved)}. "
                    "Consider continuing with similar changes."
                )

            if metrics_regressed:
                improvement_areas.append(
                    f"Metrics showing regression: {', '.join(metrics_regressed)}. "
                    "Consider reverting or revising recent changes in these areas."
                )

        # If we don't have specific improvements to suggest
        if not improvement_areas:
            improvement_areas.append(
                "Focus on optimizing the code for better performance on the target metrics"
            )

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("changes", "Unknown changes")

            # Format performance metrics using safe formatting
            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome based on comparison with parent (only numeric metrics)
            parent_metrics = program.get("parent_metrics", {})
            outcome = "Mixed results"

            # Safely compare only numeric metrics
            program_metrics = program.get("metrics", {})

            # Check if all numeric metrics improved
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Extract a snippet (first 10 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:10])
            if len(program_code.split("\n")) > 10:
                program_snippet += "\n# ... (truncated for brevity)"

            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))

            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_snippet,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            # Skip the top programs we already included
            remaining_programs = top_programs[self.config.num_top_programs :]

            # Sample diverse programs from the remaining
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Extract a snippet (first 5 lines for diversity)
                    program_code = program.get("code", "")
                    program_snippet = "\n".join(program_code.split("\n")[:5])
                    if len(program_code.split("\n")) > 5:
                        program_snippet += "\n# ... (truncated)"

                    # Calculate a composite score using safe numeric average
                    score = safe_numeric_average(program.get("metrics", {}))

                    # Extract key features
                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]  # Just first 2 metrics
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_snippet,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
        )

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []

        # Process all artifacts using .items()
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            # Truncate if too long
            if len(content) > self.config.max_artifact_bytes:
                content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
