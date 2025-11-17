"""
Evaluator for polygon decomposition evolution.

Tests decomposition algorithm over N polygons with time limits.
Primary fitness: Number of polygons successfully decomposed within time limit.
Secondary: Average violations (rectangles with sides < threshold).
Tertiary: Average number of rectangles.
"""

import sys
import os
import time
from typing import Dict, Tuple
import multiprocessing as mp
from functools import partial

# Add parent directory to path for openevolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openevolve.evaluation_result import EvaluationResult

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from polygon_generator import RectilinearPolygonGenerator


def evaluate_single_polygon(polygon_data, decompose_fn, l_min, time_limit):
    """
    Evaluate decomposition on a single polygon with timeout.

    Args:
        polygon_data: Tuple of (polygon_id, polygon_vertices)
        decompose_fn: Function to decompose polygon
        l_min: Minimum acceptable side length
        time_limit: Time limit per polygon in seconds

    Returns:
        Dict with results: success, num_rectangles, num_violations, time_taken
    """
    polygon_id, polygon_vertices = polygon_data

    start_time = time.time()

    try:
        # Run decomposition with time limit
        rectangles, stats = decompose_fn(
            polygon_vertices,
            l_min=l_min,
            time_limit=time_limit
        )

        time_taken = time.time() - start_time

        # Check if decomposition succeeded
        success = stats.get("success", False) and len(rectangles) > 0

        return {
            "polygon_id": polygon_id,
            "success": success,
            "num_rectangles": stats.get("num_rectangles", 0) if success else 0,
            "num_violations": stats.get("num_violations", 0) if success else 0,
            "time_taken": time_taken,
            "error": stats.get("error", None) if not success else None,
        }

    except Exception as e:
        time_taken = time.time() - start_time
        return {
            "polygon_id": polygon_id,
            "success": False,
            "num_rectangles": 0,
            "num_violations": 0,
            "time_taken": time_taken,
            "error": str(e),
        }


def evaluate_program(
    decompose_polygon_fn,
    num_test_polygons: int = 20,
    l_min: int = 5,
    time_limit_per_polygon: float = 60.0,
    difficulty: str = "medium",
    seed: int = None,
) -> Tuple[Dict[str, float], str]:
    """
    Evaluate polygon decomposition program over multiple test polygons.

    Args:
        decompose_polygon_fn: Function that takes (polygon_vertices, l_min, time_limit)
                              and returns (rectangles, stats)
        num_test_polygons: Number of test polygons to evaluate on
        l_min: Minimum acceptable side length
        time_limit_per_polygon: Time limit for each polygon in seconds
        difficulty: Difficulty level ("easy", "medium", "hard")
        seed: Random seed for reproducibility

    Returns:
        Tuple of (metrics_dict, artifacts_string)
    """
    # Generate test polygons
    generator = RectilinearPolygonGenerator(seed=seed)
    test_set = generator.generate_test_set(
        num_polygons=num_test_polygons,
        difficulty=difficulty
    )

    # Prepare polygon data for parallel processing
    polygon_data = [(i, poly) for i, poly in enumerate(test_set)]

    # Evaluate each polygon
    results = []
    for data in polygon_data:
        result = evaluate_single_polygon(
            data,
            decompose_polygon_fn,
            l_min,
            time_limit_per_polygon
        )
        results.append(result)

    # Calculate aggregate metrics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    feasibility_count = len(successful)
    feasibility_rate = feasibility_count / num_test_polygons if num_test_polygons > 0 else 0.0

    # For successful decompositions
    if successful:
        avg_rectangles = sum(r["num_rectangles"] for r in successful) / len(successful)
        avg_violations = sum(r["num_violations"] for r in successful) / len(successful)
        avg_time = sum(r["time_taken"] for r in successful) / len(successful)
        max_time = max(r["time_taken"] for r in successful)

        # Violation rate per rectangle
        total_rects = sum(r["num_rectangles"] for r in successful)
        total_violations = sum(r["num_violations"] for r in successful)
        violation_rate = total_violations / total_rects if total_rects > 0 else 0.0
    else:
        avg_rectangles = 0.0
        avg_violations = 0.0
        avg_time = 0.0
        max_time = 0.0
        violation_rate = 0.0

    # Combined score calculation
    # Primary: Maximize feasibility count
    # Secondary: Minimize violations (among successful)
    # Tertiary: Minimize total rectangles (among successful)

    # Normalize components
    feasibility_score = feasibility_count  # 0 to num_test_polygons

    # Violation penalty: penalize based on violation rate
    # Lower is better, so we negate for maximization
    violation_penalty = violation_rate * 10  # Scale up to be significant

    # Rectangle penalty: penalize having too many rectangles
    # Normalized by number of successful polygons
    rectangle_penalty = avg_rectangles / 10.0 if avg_rectangles > 0 else 0.0

    # Lexicographic combination:
    # - Feasibility is most important (weight = 1000)
    # - Violations are secondary (weight = 10)
    # - Rectangles are tertiary (weight = 1)
    combined_score = (
        feasibility_score * 1000.0
        - violation_penalty * 10.0
        - rectangle_penalty * 1.0
    )

    # Prepare metrics
    metrics = {
        # Primary fitness
        "feasibility_count": float(feasibility_count),
        "feasibility_rate": float(feasibility_rate),

        # Secondary objectives (among successful)
        "avg_violations": float(avg_violations),
        "avg_rectangles": float(avg_rectangles),
        "violation_rate": float(violation_rate),

        # Timing
        "avg_time": float(avg_time),
        "max_time": float(max_time),

        # Summary
        "num_test_polygons": float(num_test_polygons),
        "num_successful": float(feasibility_count),
        "num_failed": float(len(failed)),

        # Combined score for optimization
        "combined_score": float(combined_score),
    }

    # Prepare artifacts (detailed results)
    artifacts = []
    artifacts.append(f"Polygon Decomposition Evaluation Results")
    artifacts.append(f"=" * 60)
    artifacts.append(f"Test Configuration:")
    artifacts.append(f"  - Number of polygons: {num_test_polygons}")
    artifacts.append(f"  - Minimum side length (l_min): {l_min}")
    artifacts.append(f"  - Time limit per polygon: {time_limit_per_polygon}s")
    artifacts.append(f"  - Difficulty: {difficulty}")
    artifacts.append("")

    artifacts.append(f"Overall Results:")
    artifacts.append(f"  - Successful: {feasibility_count}/{num_test_polygons} ({feasibility_rate*100:.1f}%)")
    artifacts.append(f"  - Failed: {len(failed)}/{num_test_polygons}")
    artifacts.append("")

    if successful:
        artifacts.append(f"Performance Metrics (Successful Polygons):")
        artifacts.append(f"  - Average rectangles: {avg_rectangles:.2f}")
        artifacts.append(f"  - Average violations: {avg_violations:.2f}")
        artifacts.append(f"  - Violation rate: {violation_rate*100:.1f}%")
        artifacts.append(f"  - Average time: {avg_time:.2f}s")
        artifacts.append(f"  - Max time: {max_time:.2f}s")
        artifacts.append("")

    artifacts.append(f"Detailed Results:")
    artifacts.append(f"-" * 60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        artifacts.append(
            f"Polygon {r['polygon_id']:2d} {status}: "
            f"rects={r['num_rectangles']:3d}, "
            f"violations={r['num_violations']:3d}, "
            f"time={r['time_taken']:.2f}s"
        )
        if r["error"]:
            artifacts.append(f"           Error: {r['error']}")

    artifacts.append("")
    artifacts.append(f"Combined Score: {combined_score:.2f}")

    artifacts_string = "\n".join(artifacts)

    return metrics, artifacts_string


def evaluate(program_globals: dict) -> EvaluationResult:
    """
    Main evaluation entry point called by OpenEvolve.

    Args:
        program_globals: Dictionary containing the evolved program's global namespace

    Returns:
        EvaluationResult with metrics and artifacts
    """
    # Extract the decompose_polygon function from the evolved program
    if "decompose_polygon" not in program_globals:
        error_msg = "Error: decompose_polygon function not found in program"
        return EvaluationResult(
            metrics={
                "feasibility_count": 0.0,
                "feasibility_rate": 0.0,
                "avg_violations": 0.0,
                "avg_rectangles": 0.0,
                "combined_score": 0.0,
            },
            artifacts={"error": error_msg}
        )

    decompose_polygon_fn = program_globals["decompose_polygon"]

    # Configuration for evaluation
    # Start with smaller test set for faster evolution
    num_test_polygons = 10
    l_min = 5
    time_limit_per_polygon = 30.0  # 30 seconds per polygon
    difficulty = "medium"
    seed = 42  # Fixed seed for reproducibility

    try:
        metrics, artifacts = evaluate_program(
            decompose_polygon_fn,
            num_test_polygons=num_test_polygons,
            l_min=l_min,
            time_limit_per_polygon=time_limit_per_polygon,
            difficulty=difficulty,
            seed=seed,
        )
        return EvaluationResult(
            metrics=metrics,
            artifacts={"evaluation_log": artifacts}
        )

    except Exception as e:
        import traceback
        error_msg = f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
        return EvaluationResult(
            metrics={
                "feasibility_count": 0.0,
                "feasibility_rate": 0.0,
                "avg_violations": 0.0,
                "avg_rectangles": 0.0,
                "combined_score": 0.0,
            },
            artifacts={"error": error_msg}
        )


if __name__ == "__main__":
    # Test the evaluator with the initial program
    import sys
    import os

    # Import initial program
    sys.path.insert(0, os.path.dirname(__file__))
    from initial_program import decompose_polygon

    print("Testing evaluator with initial program...")
    print("-" * 80)

    # Create a mock program_globals
    program_globals = {
        "decompose_polygon": decompose_polygon
    }

    # Run evaluation
    metrics, artifacts = evaluate(program_globals)

    print("\nMetrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value}")

    print("\n" + artifacts)
