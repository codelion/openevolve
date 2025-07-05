"""
Utilities module initialization
"""

from openevolve.utils.async_utils import (
    TaskPool,
    gather_with_concurrency,
    retry_async,
    run_in_executor,
)
from openevolve.utils.code_utils import (
    apply_diff,
    calculate_edit_distance,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)
from openevolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)
from openevolve.utils.metrics_utils import (
    safe_numeric_average,
    safe_numeric_sum,
)

__all__ = [
    "TaskPool",
    "gather_with_concurrency",
    "retry_async",
    "run_in_executor",
    "apply_diff",
    "calculate_edit_distance",
    "extract_code_language",
    "extract_diffs",
    "format_diff_summary",
    "parse_evolve_blocks",
    "parse_full_rewrite",
    "format_metrics_safe",
    "format_improvement_safe",
    "safe_numeric_average",
    "safe_numeric_sum",
]
