from typing import Dict, Any  # List removed as it's not used
import json
import os
from pathlib import Path
import numpy as np

# import time # Not used
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize
import importlib.util
import sys

# import traceback # Not used
# import json # Not used
# Example custom JSON encoder if you need to save results with numpy types
import json


class NumpyFloatJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyFloatJSONEncoder, self).default(obj)


def compute_output_base_metrics(y_pred: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Computes base metrics after filtering NaNs from predictions.
    Ensures inputs y_pred and y are treated as 1D arrays.
    """
    # Ensure y_pred and y are 1D arrays.
    y_pred_1d = np.asarray(y_pred).squeeze()
    y_1d = np.asarray(y).squeeze()

    # If squeeze results in 0-D (scalar), reshape to 1-D
    if y_pred_1d.ndim == 0:
        y_pred_1d = y_pred_1d.reshape(1)
    if y_1d.ndim == 0:
        y_1d = y_1d.reshape(1)

    base_metrics_nan = {
        "mse": float("nan"),
        "nmse": float("nan"),
        "r2": float("nan"),
        "kdt": float("nan"),
        "mape": float("nan"),
        "num_valid_points": 0,
    }

    if y_pred_1d.shape != y_1d.shape and not (y_pred_1d.size == 0 and y_1d.size == 0):
        return {
            **base_metrics_nan,
            "error": "y_pred and y have incompatible shapes after ensuring 1D.",
        }

    nonnan_mask = ~np.isnan(y_pred_1d)
    y_pred_filtered = y_pred_1d[nonnan_mask]
    y_filtered = y_1d[nonnan_mask]

    if y_pred_filtered.size == 0:  # All predictions were NaN or inputs were empty
        return {
            **base_metrics_nan,
            "error": "All predictions are NaN or no data to compare after filtering.",
        }

    mse = np.mean((y_filtered - y_pred_filtered) ** 2)
    var_y = np.var(y_filtered)

    if var_y == 0:
        nmse = 0.0 if mse == 0 else float("inf")  # Consistent if true values are constant
    else:
        nmse = mse / var_y

    sum_sq_res = np.sum((y_filtered - y_pred_filtered) ** 2)
    sum_sq_total = np.sum((y_filtered - np.mean(y_filtered)) ** 2)  # Use mean of filtered y

    if sum_sq_total == 0:  # True values (after filtering) are constant
        r2 = (
            1.0 if sum_sq_res == 0 else -float("inf")
        )  # Or 0.0 if mse is also 0, definition varies. Sklearn uses 1.0.
    else:
        r2 = 1 - (sum_sq_res / sum_sq_total)

    kdt = float("nan")
    try:
        if y_filtered.size >= 2:  # Kendall's tau requires at least 2 points
            kdt_val, _ = kendalltau(y_filtered, y_pred_filtered)
            kdt = float(kdt_val)  # Ensure it's a basic float (handles np.nan)
        # If size < 2, kdt remains float('nan')
    except ValueError:  # Should be less common with size check, but as a fallback
        kdt = float("nan")  # Explicitly set, though already NaN.

    mape = float("nan")
    try:
        valid_mape_indices = y_filtered != 0
        if np.sum(valid_mape_indices) > 0:
            mape = mean_absolute_percentage_error(
                y_filtered[valid_mape_indices], y_pred_filtered[valid_mape_indices]
            )
        elif y_filtered.size > 0:  # All true values are zero
            mape = 0.0 if np.all(y_pred_filtered == 0) else float("inf")
        # If y_filtered.size is 0, mape remains float('nan')
    except ValueError:  # Fallback for any other MAPE calculation issues
        mape = float("nan")

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "r2": float(r2),
        "kdt": kdt,  # Already a float
        "mape": (
            float(mape) if mape is not float("inf") else float("inf")
        ),  # Ensure float, preserve inf
        "num_valid_points": int(y_pred_filtered.size),
    }


def objective_function(
    params: np.ndarray, model_func: callable, X_matrix: np.ndarray, y_true_vector: np.ndarray
) -> float:
    """
    Objective function for scipy.optimize.minimize.
    Calculates MSE of the model_func with given params on X_matrix, y_true_vector.
    """
    # model_func callable status is checked before calling minimize in the evaluation function.
    try:
        predictions = model_func(X_matrix, params)
        if not isinstance(predictions, np.ndarray) or predictions.shape != y_true_vector.shape:
            # print(f"Debug: Objective func - Bad prediction shape/type. Got {type(predictions)}, shape {getattr(predictions, 'shape', 'N/A')}. Expected {y_true_vector.shape}")
            return float("inf")
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            # print("Debug: Objective func - Predictions contain NaN/Inf.")
            return float("inf")
    except Exception:  # Catch any error during model prediction
        # print(f"Debug: Objective func - Exception during model_func call: {e_obj}")
        return float("inf")

    mse = np.mean((predictions - y_true_vector) ** 2)
    return mse


def evaluation(
    program_path: str,
    data_path: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates a model by loading it, optimizing its parameters, and testing it.
    The model function from program_path is expected to be named 'func'.
    """
    base_error_metrics = {
        "mse": float("nan"),
        "nmse": float("nan"),
        "r2": float("nan"),
        "kdt": float("nan"),
        "mape": float("nan"),
        "num_valid_points": 0,
    }

    def _create_error_return(error_message: str) -> Dict[str, Dict[str, Any]]:
        print(f"Error: {error_message}")
        return {
            "train_metrics": {**base_error_metrics, "error": error_message},
            "test_metrics": {**base_error_metrics, "error": error_message},
            "ood_metrics": {**base_error_metrics, "error": error_message},
        }

    # 1. Load data
    try:
        p_data_path = Path(data_path)
        train_x = np.load(p_data_path / "X_train_for_eval.npy")
        train_y = np.load(p_data_path / "y_train_for_eval.npy").squeeze()  # Ensure 1D
        test_x = np.load(p_data_path / "X_test_for_eval.npy")
        test_y = np.load(p_data_path / "y_test_for_eval.npy").squeeze()  # Ensure 1D
        test_x_ood = np.load(p_data_path / "X_ood_test_for_eval.npy")
        test_y_ood = np.load(p_data_path / "y_ood_test_for_eval.npy").squeeze()  # Ensure 1D
    except FileNotFoundError as e:
        return _create_error_return(f"Data file not found: {e.filename}")
    except Exception as e:
        return _create_error_return(f"Error loading or processing data: {str(e)}")

    # 2. Load program (model function)
    model_func = None
    try:
        p_program_path = Path(program_path)
        if not p_program_path.is_file():
            raise FileNotFoundError(f"Program file not found: {program_path}")

        spec = importlib.util.spec_from_file_location("custom_model_module", str(p_program_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec from {program_path}")

        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)

        model_func = getattr(custom_model_module, "func", None)
        if not callable(model_func):
            raise AttributeError(f"'func' function not found or not callable in {program_path}")
    except Exception as e:
        return _create_error_return(
            f"Failed to load model function 'func' from '{program_path}': {str(e)}"
        )

    # 3. Optimize parameters on training data
    optimized_params = None
    num_attempts = 10  # Default number of attempts
    best_func_value = float("inf")
    optimization_critical_error_msg = None

    # Try to get num_params from the model if it provides it, otherwise default
    num_params_to_optimize = getattr(model_func, "num_params", 10)  # Default to 10 if not specified

    print(
        f"Starting optimization for {program_path} with {num_attempts} attempts (num_params: {num_params_to_optimize})..."
    )
    for i in range(num_attempts):
        print(f"Attempt {i+1}/{num_attempts}")
        initial_params = np.random.rand(num_params_to_optimize)
        try:
            optimization_result = minimize(
                objective_function,
                initial_params,
                args=(model_func, train_x, train_y),
                method="BFGS",
                # options={'maxiter': 1000, 'disp': False} # Example options
            )
            if optimization_result.success:
                print(f"Attempt {i+1} successful. Func value: {optimization_result.fun}")
                if optimization_result.fun < best_func_value:
                    best_func_value = optimization_result.fun
                    optimized_params = optimization_result.x
                    print(f"New best result found in attempt {i+1}. Func value: {best_func_value}")
            else:
                print(
                    f"Warning: Optimization attempt {i+1} did not converge. Message: {optimization_result.message}. Func value: {optimization_result.fun}"
                )
                if (
                    optimization_result.fun < best_func_value
                ):  # Still consider if it's the best so far
                    print(
                        f"Non-converged result from attempt {i+1} is an improvement. Func value: {optimization_result.fun}"
                    )
                    best_func_value = optimization_result.fun
                    optimized_params = optimization_result.x

        except Exception as e:
            optimization_critical_error_msg = (
                f"Critical error during optimization attempt {i+1} for {program_path}: {str(e)}"
            )
            print(f"Error: {optimization_critical_error_msg}")
            break

    if optimization_critical_error_msg:
        return _create_error_return(optimization_critical_error_msg)

    def _get_metrics_for_set(
        X_data: np.ndarray, y_data: np.ndarray, set_name: str
    ) -> Dict[str, Any]:
        if optimized_params is None:
            msg = f"Optimization failed to find parameters for {program_path}, cannot evaluate {set_name}."
            return {**base_error_metrics, "error": msg}
        try:
            pred_y = model_func(X_data, optimized_params)
            if not isinstance(pred_y, np.ndarray):
                raise ValueError(f"{set_name} predictions are not numpy arrays. Got {type(pred_y)}")

            metrics = compute_output_base_metrics(pred_y, y_data)
            if "error" in metrics and metrics["num_valid_points"] == 0:
                print(f"Warning for {set_name} ({program_path}): {metrics['error']}")
            return metrics
        except Exception as e:
            error_msg = f"{set_name} evaluation failed for '{program_path}': {str(e)}"
            print(f"Error: {error_msg}")
            return {**base_error_metrics, "error": error_msg}

    train_metrics = _get_metrics_for_set(train_x, train_y, "Train set")
    test_metrics = _get_metrics_for_set(test_x, test_y, "Test set")
    ood_metrics = _get_metrics_for_set(test_x_ood, test_y_ood, "OOD test set")

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_problems_directory_or_single_problem>")
        sys.exit(1)

    root_path_arg = sys.argv[1]
    path_obj = Path(root_path_arg)
    problem_dirs = []

    # Check if the path is a single problem directory
    # A problem directory is expected to contain data files directly and an openevolve_output subdir
    program_file_check = path_obj / "openevolve_output" / "best" / "best_program.py"
    data_file_check = path_obj / "X_train_for_eval.npy"

    if data_file_check.exists() and program_file_check.exists():
        problem_dirs.append(path_obj)
        print(f"Identified as single problem directory: {path_obj}")
    else:
        # Assume path is a parent directory containing multiple problem subdirectories
        print(
            f"Identified as parent directory: {path_obj}. Searching for problem subdirectories..."
        )
        try:
            if not path_obj.is_dir():
                print(f"Error: Root path {root_path_arg} is not a directory.")
                sys.exit(1)
            for d in path_obj.iterdir():
                if d.is_dir():
                    # Check if this subdirectory looks like a problem directory
                    if (d / "X_train_for_eval.npy").exists() and (
                        d / "openevolve_output" / "best" / "best_program.py"
                    ).exists():
                        problem_dirs.append(d)
                        print(f"  Found problem subdirectory: {d.name}")
                    else:
                        print(f"  Skipping subdirectory (missing data or program): {d.name}")
        except FileNotFoundError:
            print(f"Error: Root directory not found: {root_path_arg}")
            sys.exit(1)

    if not problem_dirs:
        print(
            f"No valid problem subdirectories found in '{root_path_arg}' or '{root_path_arg}' itself is not a valid problem directory."
        )
        sys.exit(1)

    all_results = {}
    for subdir_path in problem_dirs:
        problem_name = subdir_path.name
        # if "21" not in problem_name: continue
        print(f"\nProcessing problem: {problem_name}")
        program_file_path = subdir_path / "openevolve_output" / "best" / "best_program.py"
        data_dir_path = subdir_path

        if (
            not program_file_path.exists()
        ):  # Should have been caught by subdir check, but as a safeguard
            print(f"Skipping {problem_name}: best_program.py not found at {program_file_path}")
            all_results[problem_name] = {
                "train_metrics": {"error": "best_program.py not found"},
                "test_metrics": {"error": "best_program.py not found"},
                "ood_metrics": {"error": "best_program.py not found"},
            }
            continue

        print(f"  Program path: {program_file_path}")
        print(f"  Data path: {data_dir_path}")

        metrics_output = evaluation(  # Renamed from 'metrics' to avoid conflict
            program_path=str(program_file_path),
            data_path=str(data_dir_path),
        )
        print(f"  Metrics for {problem_name}: {metrics_output}")
        all_results[problem_name] = metrics_output

    print("\n--- All Evaluation Results ---")
    for problem, result in all_results.items():
        print(f"\nProblem: {problem}")
        print(f"  Train Metrics: {result.get('train_metrics')}")
        print(f"  Test Metrics: {result.get('test_metrics')}")
        print(f"  OOD Metrics: {result.get('ood_metrics')}")

    # --- Overall Performance Calculation ---
    overall_performance = {}
    # Metrics to aggregate: mse, nmse, r2, kdt, mape
    metric_keys = ["mse", "nmse", "r2", "kdt", "mape"]
    dataset_types = ["train_metrics", "test_metrics", "ood_metrics"]

    for d_type in dataset_types:
        overall_performance[d_type] = {}
        for m_key in metric_keys:
            all_scores = []
            for problem_name, results_data in all_results.items():
                # Ensure the dataset type (e.g., train_metrics) exists and doesn't have a top-level error
                if d_type in results_data and "error" not in results_data[d_type]:
                    score = results_data[d_type].get(m_key)
                    # Only include if score is a number (not nan, not None, not inf for some metrics initially)
                    # np.nanmean and np.nanmedian will handle internal NaNs gracefully.
                    # We explicitly exclude inf from aggregation here, as it can skew means badly.
                    # For R2, -inf is possible and should be handled by nanmedian/nanmean or filtered if desired.
                    if isinstance(score, (int, float)) and not np.isinf(
                        score
                    ):  # np.isnan(score) is fine for nan* functions
                        all_scores.append(score)
                    elif (
                        score == -float("inf") and m_key == "r2"
                    ):  # Special case for R2, allow -inf
                        all_scores.append(score)

            if all_scores:
                # Replace -inf with NaN for R2 mean calculation if desired, or handle as is.
                # For simplicity, we'll let nanmean/nanmedian handle it.
                # Extreme values can still affect the mean significantly.

                # Filter out inf values for mean calculation as they make it non-informative
                # but keep them for median if appropriate (or filter there too).
                # For simplicity here, we are filtering inf before both.
                # A more nuanced approach might replace inf with a very large/small number or handle per metric.

                scores_for_mean = [s for s in all_scores if s != -float("inf")]  # R2 can be -inf

                overall_performance[d_type][f"mean_{m_key}"] = (
                    np.nanmean(scores_for_mean) if scores_for_mean else float("nan")
                )
                overall_performance[d_type][f"median_{m_key}"] = (
                    np.nanmedian(all_scores) if all_scores else float("nan")
                )
                overall_performance[d_type][f"num_problems_for_{m_key}"] = len(all_scores)
            else:
                overall_performance[d_type][f"mean_{m_key}"] = float("nan")
                overall_performance[d_type][f"median_{m_key}"] = float("nan")
                overall_performance[d_type][f"num_problems_for_{m_key}"] = 0

    print("\n--- Overall Performance Summary ---")
    for d_type, metrics_summary in overall_performance.items():
        print(f"\n{d_type.replace('_', ' ').title()}:")
        if not metrics_summary:
            print("  No data for overall summary.")
            continue
        for stat_name, value in metrics_summary.items():
            if "num_problems_for_" in stat_name:  # Print count separately or alongside
                m_key = stat_name.replace("num_problems_for_", "")
                print(f"  Number of problems for {m_key.upper()} stats: {value}")
            elif "mean_" in stat_name or "median_" in stat_name:
                print(
                    f"  {stat_name.replace('_', ' ').title()}: {value:.4f}"
                    if isinstance(value, float) and not np.isnan(value)
                    else f"  {stat_name.replace('_', ' ').title()}: {value}"
                )

    # Add overall performance to the results to be saved
    all_results["overall_performance_summary"] = overall_performance

    # Optional: Save all_results to a JSON file
    # Determine the output file path. If root_path_arg is a file, save alongside it. If a dir, save inside it.
    if path_obj.is_file():  # Should not happen with current logic, but as a fallback
        output_results_file = path_obj.parent / "all_evaluation_results.json"
    else:  # path_obj is a directory
        output_results_file = path_obj / "all_evaluation_results.json"

    try:
        with open(output_results_file, "w") as f:
            json.dump(all_results, f, indent=4, cls=NumpyFloatJSONEncoder)
        print(f"\nAll results, including overall performance, saved to {output_results_file}")
    except Exception as e:
        print(f"\nError saving results to JSON: {e}")
