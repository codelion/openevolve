"""
Evaluator for circle packing example
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import threading
import traceback
import sys


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout_seconds} seconds"
            )


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def evaluate(program_path):
    """
    Evaluate the program by running it for n=26 and n=32 and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Target values from the paper
    TARGETS = {26: 2.635, 32: 2.937}  # AlphaEvolve result for n=26  # AlphaEvolve result for n=32

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_packing"):
            print(f"Error: program does not have 'run_packing' function")
            return {"sum_radii": 0.0, "validity": 0.0, "combined_score": 0.0}

        # Run for two different n values
        results = {}

        for n in [26, 32]:
            try:
                start_time = time.time()

                # Run packing with timeout
                centers, radii, sum_radii = run_with_timeout(
                    program.run_packing, args=(n,), timeout_seconds=30
                )

                end_time = time.time()

                # Ensure centers and radii are numpy arrays
                if not isinstance(centers, np.ndarray):
                    centers = np.array(centers)
                if not isinstance(radii, np.ndarray):
                    radii = np.array(radii)

                # Validate solution
                valid = validate_packing(centers, radii)

                # Check shape and size
                shape_valid = centers.shape == (n, 2) and radii.shape == (n,)
                if not shape_valid:
                    print(
                        f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected ({n}, 2) and ({n},)"
                    )
                    valid = False

                # Recalculate sum to verify
                actual_sum = np.sum(radii) if valid else 0.0

                # Make sure sum_radii matches the actual sum
                if abs(actual_sum - sum_radii) > 1e-6:
                    print(
                        f"Warning: Reported sum {sum_radii} doesn't match calculated sum {actual_sum}"
                    )

                target = TARGETS[n]

                # Store results
                results[n] = {
                    "valid": valid,
                    "sum_radii": actual_sum,
                    "time": end_time - start_time,
                    "target_ratio": actual_sum / target if valid else 0.0,
                }

                print(
                    f"n={n}: valid={valid}, sum_radii={actual_sum:.6f}, target={target}, ratio={actual_sum/target if valid else 0:.6f}"
                )

            except TimeoutError as e:
                print(f"Timeout running for n={n}: {str(e)}")
                results[n] = {
                    "valid": False,
                    "sum_radii": 0.0,
                    "time": 30.0,  # timeout value
                    "target_ratio": 0.0,
                }
            except Exception as e:
                print(f"Error running for n={n}: {str(e)}")
                traceback.print_exc()
                results[n] = {"valid": False, "sum_radii": 0.0, "time": 0.0, "target_ratio": 0.0}

        # Calculate combined metrics
        avg_ratio = (results[26]["target_ratio"] + results[32]["target_ratio"]) / 2
        validity = 1.0 if results[26]["valid"] and results[32]["valid"] else 0.0

        # Return metrics - higher values are better
        return {
            "sum_radii_26": float(results[26]["sum_radii"]),
            "sum_radii_32": float(results[32]["sum_radii"]),
            "target_ratio_26": float(results[26]["target_ratio"]),
            "target_ratio_32": float(results[32]["target_ratio"]),
            "validity": float(validity),
            "avg_target_ratio": float(avg_ratio),
            "combined_score": float(avg_ratio * validity),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "sum_radii_26": 0.0,
            "sum_radii_32": 0.0,
            "target_ratio_26": 0.0,
            "target_ratio_32": 0.0,
            "validity": 0.0,
            "avg_target_ratio": 0.0,
            "combined_score": 0.0,
        }


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check with only n=26
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_packing"):
            print(f"Error: program does not have 'run_packing' function")
            return {"validity": 0.0, "error": "Missing run_packing function"}

        try:
            # Run with a lower iteration count for quicker checking
            centers, radii, sum_radii = run_with_timeout(
                program.run_packing, args=(26,), timeout_seconds=10
            )

            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Validate solution (shapes and constraints)
            shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
            if not shape_valid:
                print(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}")
                return {"validity": 0.0, "error": "Invalid shapes"}

            valid = validate_packing(centers, radii)

            # Calculate sum
            actual_sum = np.sum(radii) if valid else 0.0

            # Target from paper
            target = 2.635

            # Return evaluation metrics
            return {
                "validity": 1.0 if valid else 0.0,
                "sum_radii": float(actual_sum),
                "target_ratio": float(actual_sum / target if valid else 0.0),
            }

        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"validity": 0.0, "error": "Timeout"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {"validity": 0.0, "error": str(e)}

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {e}")
        print(traceback.format_exc())
        return {"validity": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation with n=26 and n=32
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
