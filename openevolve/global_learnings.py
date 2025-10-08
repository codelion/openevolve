"""
Global learnings system for OpenEvolve

Aggregates and tracks common failures and successful patterns across all islands
and iterations to provide insights that help avoid repeated mistakes.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import GlobalLearningsConfig

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Represents a failure pattern observed during evolution"""

    pattern_type: str  # "syntax", "runtime", "performance_regression"
    description: str
    count: int = 1
    first_seen: int = 0  # iteration number
    last_seen: int = 0
    example_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailurePattern":
        return cls(**data)


@dataclass
class SuccessPattern:
    """Represents a successful pattern observed during evolution"""

    description: str
    count: int = 1
    avg_improvement: float = 0.0
    first_seen: int = 0
    last_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessPattern":
        return cls(**data)


class GlobalLearnings:
    """
    Tracks and aggregates learnings from evolution across all islands and iterations
    """

    def __init__(self, config: GlobalLearningsConfig):
        self.config = config
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.iteration_history: List[int] = []  # Track which iterations we've seen
        self.last_update_iteration: int = 0

        logger.info(f"Initialized GlobalLearnings (enabled={config.enabled})")

    def update_from_iteration(
        self,
        iteration: int,
        result: Any,
        parent_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update learnings from an iteration result

        Args:
            iteration: Current iteration number
            result: Iteration result containing child_program, metrics, artifacts, etc.
            parent_metrics: Parent program metrics for comparison
        """
        if not self.config.enabled:
            return

        self.iteration_history.append(iteration)

        # Trim history to window size
        if len(self.iteration_history) > self.config.window_size:
            self.iteration_history = self.iteration_history[-self.config.window_size :]

        # Track failures
        if self.config.track_failures or self.config.track_both:
            self._track_failures(iteration, result)

        # Track successes
        if self.config.track_successes or self.config.track_both:
            self._track_successes(iteration, result, parent_metrics)

        self.last_update_iteration = iteration

    def _track_failures(self, iteration: int, result: Any) -> None:
        """Track failure patterns from iteration result"""
        artifacts = getattr(result, "artifacts", None)
        if not artifacts:
            return

        # Extract syntax errors
        if self.config.include_syntax_errors:
            syntax_errors = self._extract_syntax_errors(artifacts)
            for error_desc in syntax_errors:
                self._add_failure_pattern("syntax", error_desc, iteration, error_desc)

        # Extract runtime errors
        if self.config.include_runtime_errors:
            runtime_errors = self._extract_runtime_errors(artifacts)
            for error_desc in runtime_errors:
                self._add_failure_pattern("runtime", error_desc, iteration, error_desc)

        # Track performance regressions
        if self.config.include_performance_regressions:
            child_metrics = getattr(result, "child_metrics", None)
            parent = getattr(result, "parent", None)
            if child_metrics and parent and hasattr(parent, "metrics"):
                regressions = self._detect_performance_regressions(
                    parent.metrics, child_metrics
                )
                for regression_desc in regressions:
                    self._add_failure_pattern(
                        "performance_regression", regression_desc, iteration
                    )

    def _track_successes(
        self, iteration: int, result: Any, parent_metrics: Optional[Dict[str, float]]
    ) -> None:
        """Track success patterns from iteration result"""
        child_metrics = getattr(result, "child_metrics", None)
        child_program = getattr(result, "child_program", None)

        if not child_metrics or not child_program or not parent_metrics:
            return

        # Calculate improvement
        improvement = self._calculate_improvement(parent_metrics, child_metrics)

        if improvement >= self.config.min_improvement_threshold:
            # Extract what changed
            changes = getattr(child_program, "metadata", {}).get("changes", "Unknown")
            if changes and changes != "Unknown":
                self._add_success_pattern(changes, iteration, improvement)

    def _extract_syntax_errors(self, artifacts: Dict[str, Any]) -> List[str]:
        """Extract syntax errors from artifacts"""
        errors = []

        # Check stderr for syntax errors
        stderr = artifacts.get("stderr", "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        # Common Python syntax error patterns
        syntax_patterns = [
            (r"SyntaxError: (.+)", lambda m: f"SyntaxError: {m.group(1)}"),
            (r"IndentationError: (.+)", lambda m: f"IndentationError: {m.group(1)}"),
            (r"NameError: name '(\w+)' is not defined", lambda m: f"Undefined variable: {m.group(1)}"),
            (r"invalid syntax", lambda m: "Invalid syntax"),
        ]

        for pattern, formatter in syntax_patterns:
            matches = re.finditer(pattern, stderr)
            for match in matches:
                errors.append(formatter(match))

        return errors

    def _extract_runtime_errors(self, artifacts: Dict[str, Any]) -> List[str]:
        """Extract runtime errors from artifacts"""
        errors = []

        stderr = artifacts.get("stderr", "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        # Common runtime error patterns
        runtime_patterns = [
            (r"IndexError: (.+)", lambda m: f"IndexError: {m.group(1)}"),
            (r"KeyError: (.+)", lambda m: f"KeyError: {m.group(1)}"),
            (r"ValueError: (.+)", lambda m: f"ValueError: {m.group(1)}"),
            (r"TypeError: (.+)", lambda m: f"TypeError: {m.group(1)}"),
            (r"AttributeError: (.+)", lambda m: f"AttributeError: {m.group(1)}"),
            (r"ZeroDivisionError", lambda m: "Division by zero"),
        ]

        for pattern, formatter in runtime_patterns:
            matches = re.finditer(pattern, stderr)
            for match in matches:
                errors.append(formatter(match))

        return errors

    def _detect_performance_regressions(
        self, parent_metrics: Dict[str, float], child_metrics: Dict[str, float]
    ) -> List[str]:
        """Detect performance regressions"""
        regressions = []

        for metric_name, child_value in child_metrics.items():
            if metric_name not in parent_metrics:
                continue

            parent_value = parent_metrics[metric_name]

            # Only compare numeric values
            if not isinstance(child_value, (int, float)) or not isinstance(
                parent_value, (int, float)
            ):
                continue

            # Check for significant regression (>10%)
            if parent_value > 0 and child_value < parent_value * 0.9:
                regression_pct = ((parent_value - child_value) / parent_value) * 100
                regressions.append(
                    f"{metric_name} decreased by {regression_pct:.1f}% "
                    f"({parent_value:.3f} → {child_value:.3f})"
                )

        return regressions

    def _calculate_improvement(
        self, parent_metrics: Dict[str, float], child_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall improvement score"""
        improvements = []

        for metric_name, child_value in child_metrics.items():
            if metric_name not in parent_metrics:
                continue

            parent_value = parent_metrics[metric_name]

            if not isinstance(child_value, (int, float)) or not isinstance(
                parent_value, (int, float)
            ):
                continue

            if parent_value > 0:
                improvement = (child_value - parent_value) / parent_value
                improvements.append(improvement)

        if improvements:
            return sum(improvements) / len(improvements)
        return 0.0

    def _add_failure_pattern(
        self,
        pattern_type: str,
        description: str,
        iteration: int,
        example_error: Optional[str] = None,
    ) -> None:
        """Add or update a failure pattern"""
        # Normalize description for grouping
        key = f"{pattern_type}:{description}"

        if key in self.failure_patterns:
            pattern = self.failure_patterns[key]
            pattern.count += 1
            pattern.last_seen = iteration
        else:
            self.failure_patterns[key] = FailurePattern(
                pattern_type=pattern_type,
                description=description,
                count=1,
                first_seen=iteration,
                last_seen=iteration,
                example_error=example_error,
            )

    def _add_success_pattern(
        self, description: str, iteration: int, improvement: float
    ) -> None:
        """Add or update a success pattern"""
        key = description

        if key in self.success_patterns:
            pattern = self.success_patterns[key]
            # Update average improvement
            total_improvement = pattern.avg_improvement * pattern.count + improvement
            pattern.count += 1
            pattern.avg_improvement = total_improvement / pattern.count
            pattern.last_seen = iteration
        else:
            self.success_patterns[key] = SuccessPattern(
                description=description,
                count=1,
                avg_improvement=improvement,
                first_seen=iteration,
                last_seen=iteration,
            )

    def get_top_failures(self, max_count: Optional[int] = None) -> List[FailurePattern]:
        """Get top failure patterns sorted by count"""
        max_count = max_count or self.config.max_learnings

        # Filter by minimum count
        filtered = [
            p for p in self.failure_patterns.values() if p.count >= self.config.min_failure_count
        ]

        # Sort by count (descending)
        sorted_patterns = sorted(filtered, key=lambda p: p.count, reverse=True)

        return sorted_patterns[:max_count]

    def get_top_successes(self, max_count: Optional[int] = None) -> List[SuccessPattern]:
        """Get top success patterns sorted by count and improvement"""
        max_count = max_count or self.config.max_learnings

        # Filter by minimum count
        filtered = [
            p for p in self.success_patterns.values() if p.count >= self.config.min_success_count
        ]

        # Sort by count * avg_improvement (descending)
        sorted_patterns = sorted(
            filtered, key=lambda p: p.count * p.avg_improvement, reverse=True
        )

        return sorted_patterns[:max_count]

    def generate_prompt_section(self) -> str:
        """
        Generate formatted section for prompt injection

        Returns:
            Formatted string with learnings, or empty string if disabled or no learnings
        """
        if not self.config.enabled:
            return ""

        sections = []

        # Add failures section
        if self.config.track_failures or self.config.track_both:
            failures = self.get_top_failures()
            if failures:
                sections.append(self._format_failures_section(failures))

        # Add successes section
        if self.config.track_successes or self.config.track_both:
            successes = self.get_top_successes()
            if successes:
                sections.append(self._format_successes_section(successes))

        if not sections:
            return ""

        # Combine sections
        header = "## Evolution Insights (Global Learnings)"
        if self.config.verbosity == "minimal":
            header = "## Common Patterns"

        return f"{header}\n\n" + "\n\n".join(sections)

    def _format_failures_section(self, failures: List[FailurePattern]) -> str:
        """Format failures section based on verbosity"""
        lines = []

        if self.config.verbosity == "minimal":
            lines.append("### Avoid:")
            for f in failures:
                lines.append(f"- {f.description} (seen {f.count}x)")
        elif self.config.verbosity == "concise":
            lines.append("### Common Pitfalls:")
            for f in failures:
                icon = "❌" if f.pattern_type == "syntax" else "⚠️"
                lines.append(f"{icon} {f.description} (seen {f.count}x)")
        else:  # detailed
            lines.append("### Common Pitfalls (from recent evolution):")
            for f in failures:
                icon = "❌" if f.pattern_type == "syntax" else "⚠️"
                lines.append(
                    f"{icon} **{f.pattern_type.replace('_', ' ').title()}**: "
                    f"{f.description} (seen {f.count}x, last at iteration {f.last_seen})"
                )

        return "\n".join(lines)

    def _format_successes_section(self, successes: List[SuccessPattern]) -> str:
        """Format successes section based on verbosity"""
        lines = []

        if self.config.verbosity == "minimal":
            lines.append("### Successful patterns:")
            for s in successes:
                lines.append(f"- {s.description} (seen {s.count}x)")
        elif self.config.verbosity == "concise":
            lines.append("### Successful Patterns:")
            for s in successes:
                lines.append(
                    f"✅ {s.description} (seen {s.count}x, avg improvement: +{s.avg_improvement:.2%})"
                )
        else:  # detailed
            lines.append("### Successful Patterns (from recent evolution):")
            for s in successes:
                lines.append(
                    f"✅ **Success**: {s.description} (seen {s.count}x, "
                    f"avg improvement: +{s.avg_improvement:.2%}, last at iteration {s.last_seen})"
                )

        return "\n".join(lines)

    def save(self, checkpoint_dir: Path) -> None:
        """
        Save global learnings to checkpoint directory

        Args:
            checkpoint_dir: Directory to save checkpoint data
        """
        if not self.config.enabled:
            return

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "failure_patterns": {k: v.to_dict() for k, v in self.failure_patterns.items()},
            "success_patterns": {k: v.to_dict() for k, v in self.success_patterns.items()},
            "iteration_history": self.iteration_history,
            "last_update_iteration": self.last_update_iteration,
        }

        save_path = checkpoint_dir / "global_learnings.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved global learnings to {save_path}")

    def load(self, checkpoint_dir: Path) -> None:
        """
        Load global learnings from checkpoint directory

        Args:
            checkpoint_dir: Directory containing checkpoint data
        """
        if not self.config.enabled:
            return

        checkpoint_dir = Path(checkpoint_dir)
        load_path = checkpoint_dir / "global_learnings.json"

        if not load_path.exists():
            logger.warning(f"Global learnings checkpoint not found at {load_path}")
            return

        with open(load_path, "r") as f:
            data = json.load(f)

        # Restore failure patterns
        self.failure_patterns = {
            k: FailurePattern.from_dict(v) for k, v in data.get("failure_patterns", {}).items()
        }

        # Restore success patterns
        self.success_patterns = {
            k: SuccessPattern.from_dict(v) for k, v in data.get("success_patterns", {}).items()
        }

        # Restore metadata
        self.iteration_history = data.get("iteration_history", [])
        self.last_update_iteration = data.get("last_update_iteration", 0)

        logger.info(
            f"Loaded global learnings from {load_path} "
            f"({len(self.failure_patterns)} failures, {len(self.success_patterns)} successes)"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for logging"""
        return {
            "total_failures": len(self.failure_patterns),
            "total_successes": len(self.success_patterns),
            "iterations_tracked": len(self.iteration_history),
            "last_update": self.last_update_iteration,
            "top_failures": len(self.get_top_failures()),
            "top_successes": len(self.get_top_successes()),
        }
