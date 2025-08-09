"""
Evaluator for the svm task with baseline comparison

This evaluator compares OpenEvolve's evolved solutions against the reference
AlgoTune baseline implementation to measure performance improvements.
The speedup becomes the primary fitness score for evolution.
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add AlgoTune to path for importing reference tasks
# These paths will be dynamically determined based on the AlgoTune installation
# The adapter will handle path setup when the evaluator is created

# Setup AlgoTune paths dynamically
def setup_algotune_paths():
    """Setup Python import paths for AlgoTune modules."""
    # The AlgoTune path should be passed as a parameter to the evaluator
    possible_algotune_paths = [
        Path(__file__).parent.parent.parent.parent / "AlgoTune",
        Path.home() / "github" / "AlgoTune",
    ]
    
    algotune_base = None
    for path in possible_algotune_paths:
        if path.exists():
            algotune_base = path
            break
    
    if algotune_base is None:
        print("Warning: Could not find AlgoTune installation")
        return False
    
    # Add AlgoTune base directory to path
    if str(algotune_base) not in sys.path:
        sys.path.insert(0, str(algotune_base))
    
    return True

# Setup paths and try to import AlgoTune tasks
if setup_algotune_paths():
    try:
        from AlgoTuneTasks.base import TASK_REGISTRY
        # Import the specific svm task to register it
        from AlgoTuneTasks.svm.svm import SVMTask
        print("Successfully imported AlgoTune tasks and svm")
    except ImportError as e:
        print(f"Error: Could not import AlgoTune tasks: {e}")
        print("Make sure AlgoTune is properly installed and accessible")
        TASK_REGISTRY = {}
else:
    print("Warning: Could not setup AlgoTune paths")
    TASK_REGISTRY = {}

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
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

def safe_convert(value):
    """Convert a value safely for evaluation"""
    try:
        if isinstance(value, (list, tuple)):
            return [safe_convert(v) for v in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    except Exception:
        return value

def calculate_speedup(baseline_time_ms: float, evolved_time_ms: float, is_valid: bool) -> Optional[float]:
    """
    Calculate speedup between baseline and evolved solution.
    
    Speedup = (Baseline Time) / (Evolved Time)
    Higher is better.
    
    Args:
        baseline_time_ms: Time taken by baseline implementation
        evolved_time_ms: Time taken by evolved solution
        is_valid: Whether the evolved solution is valid
        
    Returns:
        Speedup value or None if calculation is not possible
    """
    if not is_valid:
        return None
        
    if baseline_time_ms is None or baseline_time_ms <= 0:
        return None
        
    if evolved_time_ms is None:
        return None
        
    if evolved_time_ms <= 0:
        return float('inf')  # Infinite speedup for instant solution
        
    return baseline_time_ms / evolved_time_ms

def measure_baseline_performance(task_instance, problem, num_runs=3, warmup_runs=1):
    """
    Measure baseline performance using the original AlgoTune implementation.
    
    Args:
        task_instance: The AlgoTune task instance
        problem: Problem to solve
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with baseline timing results
    """
    try:
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                task_instance.solve(problem)
            except Exception:
                pass  # Ignore warmup errors
                
        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = task_instance.solve(problem)
                end_time = time.perf_counter()
                if result is not None:
                    elapsed_ms = (end_time - start_time) * 1000
                    times.append(elapsed_ms)
            except Exception as e:
                print(f"Baseline run failed: {e}")
                continue
                
        if not times:
            return {
                "success": False,
                "error": "All baseline runs failed",
                "avg_time_ms": None,
                "min_time_ms": None,
                "std_time_ms": None
            }
            
        return {
            "success": True,
            "avg_time_ms": float(np.mean(times)),
            "min_time_ms": float(np.min(times)),
            "std_time_ms": float(np.std(times)),
            "times": times
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "avg_time_ms": None,
            "min_time_ms": None,
            "std_time_ms": None
        }

def measure_evolved_performance(program, problem, num_runs=3, warmup_runs=1, timeout_seconds=30):
    """
    Measure evolved solution performance.
    
    Args:
        program: The evolved program module
        problem: Problem to solve
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
        timeout_seconds: Timeout per run
        
    Returns:
        Dictionary with evolved timing results
    """
    try:
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                run_with_timeout(program.run_solver, args=(problem,), timeout_seconds=timeout_seconds)
            except Exception:
                pass  # Ignore warmup errors
                
        # Timing runs
        times = []
        results = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = run_with_timeout(program.run_solver, args=(problem,), timeout_seconds=timeout_seconds)
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                results.append(result)
            except TimeoutError:
                print(f"Evolved solution timed out after {timeout_seconds} seconds")
                continue
            except Exception as e:
                print(f"Evolved run failed: {e}")
                continue
                
        if not times:
            return {
                "success": False,
                "error": "All evolved runs failed",
                "avg_time_ms": None,
                "min_time_ms": None,
                "std_time_ms": None,
                "results": []
            }
            
        return {
            "success": True,
            "avg_time_ms": float(np.mean(times)),
            "min_time_ms": float(np.min(times)),
            "std_time_ms": float(np.std(times)),
            "times": times,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "avg_time_ms": None,
            "min_time_ms": None,
            "std_time_ms": None,
            "results": []
        }

def evaluate(program_path, config=None):
    """
    Enhanced evaluation with baseline comparison for svm task.
    
    This evaluator:
    1. Loads the evolved solve method from initial_program.py
    2. Generates test problems using the original AlgoTune task
    3. Measures baseline performance using original AlgoTune implementation
    4. Measures evolved solution performance
    5. Calculates speedup as primary fitness score
    6. Validates correctness using the original task's validation method

    Args:
        program_path: Path to the evolved program file (initial_program.py)
        config: Configuration dictionary with evaluator settings

    Returns:
        Dictionary of metrics including speedup as primary fitness score
    """
    try:
        # Load configuration
        if config is None:
            config = {
                "algotune": {
                    "num_trials": 5,
                    "data_size": 5,
                    "timeout": 30,
                    "num_runs": 3,
                    "warmup_runs": 1
                }
            }
        
        # Extract AlgoTune task-specific settings from config
        algotune_config = config.get("algotune", {})
        num_trials = algotune_config.get("num_trials", 5)
        data_size = algotune_config.get("data_size", 5)
        timeout_seconds = algotune_config.get("timeout", 30)
        num_runs = algotune_config.get("num_runs", 3)
        warmup_runs = algotune_config.get("warmup_runs", 1)
        
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_solver"):
            print(f"Error: program does not have 'run_solver' function")
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "speedup_score": 0.0,  # Primary fitness score
                "baseline_comparison": {
                    "mean_speedup": None,
                    "median_speedup": None,
                    "success_rate": 0.0,
                    "baseline_times": [],
                    "evolved_times": [],
                    "speedups": []
                },
                "error": "Missing run_solver function",
            }

        # Get the original task for reference solutions and problem generation
        task_class = None
        if "svm" in TASK_REGISTRY:
            task_class = TASK_REGISTRY["svm"]
            print(f"Successfully loaded svm task from registry")
        else:
            print(f"Error: svm task not found in TASK_REGISTRY")
            print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
            raise Exception("Could not load svm task from AlgoTune registry")

        # Generate test problems and evaluate
        correctness_scores = []
        performance_scores = []
        baseline_times = []
        evolved_times = []
        speedups = []
        valid_count = 0
        success_count = 0

        for trial in range(num_trials):
            try:
                # Generate a test problem using the original task
                if task_class:
                    task_instance = task_class()
                    problem = task_instance.generate_problem(n=data_size, random_seed=trial)
                else:
                    raise Exception("Could not load original AlgoTune task for problem generation")

                # Measure baseline performance
                baseline_result = measure_baseline_performance(
                    task_instance, problem, num_runs, warmup_runs
                )
                
                if not baseline_result["success"]:
                    print(f"Trial {trial}: Baseline measurement failed: {baseline_result.get('error', 'Unknown error')}")
                    continue

                # Measure evolved performance
                evolved_result = measure_evolved_performance(
                    program, problem, num_runs, warmup_runs, timeout_seconds
                )
                
                if not evolved_result["success"]:
                    print(f"Trial {trial}: Evolved measurement failed: {evolved_result.get('error', 'Unknown error')}")
                    continue

                # Validate evolved solution
                correctness_score = 0.0
                is_valid = False
                
                if evolved_result["results"]:
                    # Use the first result for validation
                    evolved_solution = evolved_result["results"][0]
                    evolved_solution = safe_convert(evolved_solution)
                    
                    try:
                        is_valid = task_instance.is_solution(problem, evolved_solution)
                        correctness_score = 1.0 if is_valid else 0.0
                    except Exception as e:
                        print(f"Trial {trial}: Error checking solution validity: {e}")
                        correctness_score = 0.0
                        is_valid = False

                # Calculate speedup
                baseline_time = baseline_result["min_time_ms"]  # Use minimum time for fair comparison
                evolved_time = evolved_result["min_time_ms"]
                speedup = calculate_speedup(baseline_time, evolved_time, is_valid)

                # Store results
                correctness_scores.append(correctness_score)
                baseline_times.append(baseline_time)
                evolved_times.append(evolved_time)
                
                if speedup is not None:
                    speedups.append(speedup)
                    valid_count += 1
                
                # Performance score based on execution time
                performance_score = 1.0 / (1.0 + evolved_time) if evolved_time > 0 else 0.0
                performance_scores.append(performance_score)
                success_count += 1

            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "speedup_score": 0.0,  # Primary fitness score
                "baseline_comparison": {
                    "mean_speedup": None,
                    "median_speedup": None,
                    "success_rate": 0.0,
                    "baseline_times": [],
                    "evolved_times": [],
                    "speedups": []
                },
                "error": "All trials failed",
            }

        # Calculate metrics
        avg_correctness = float(np.mean(correctness_scores))
        avg_performance = float(np.mean(performance_scores))
        reliability_score = float(success_count / num_trials)

        # Calculate speedup as primary fitness score
        if speedups:
            mean_speedup = float(np.mean(speedups))
            # Use speedup as primary fitness score (higher is better)
            speedup_score = mean_speedup
        else:
            speedup_score = 0.0
            mean_speedup = None

        # Combined score prioritizing correctness (kept for compatibility)
        combined_score = float(
            0.7 * avg_correctness + 0.2 * avg_performance + 0.1 * reliability_score
        )

        # Calculate baseline comparison metrics
        baseline_comparison = {
            "mean_speedup": mean_speedup,
            "median_speedup": float(np.median(speedups)) if speedups else None,
            "success_rate": float(valid_count / success_count) if success_count > 0 else 0.0,
            "baseline_times": baseline_times,
            "evolved_times": evolved_times,
            "speedups": speedups,
            "num_valid_solutions": valid_count,
            "num_total_trials": success_count
        }

        return {
            "correctness_score": avg_correctness,
            "performance_score": avg_performance,
            "reliability_score": reliability_score,
            "combined_score": combined_score,
            "speedup_score": speedup_score,  # Primary fitness score for evolution
            "success_rate": reliability_score,
            "baseline_comparison": baseline_comparison,
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "correctness_score": 0.0,
            "performance_score": 0.0,
            "combined_score": 0.0,
            "speedup_score": 0.0,  # Primary fitness score
            "baseline_comparison": {
                "mean_speedup": None,
                "median_speedup": None,
                "success_rate": 0.0,
                "baseline_times": [],
                "evolved_times": [],
                "speedups": []
            },
            "error": str(e),
        }

# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path, config=None):
    """First stage evaluation with basic functionality check of the evolved solve method"""
    try:
        # Load configuration
        if config is None:
            config = {
                "algotune": {
                    "num_trials": 5,
                    "data_size": 5,
                    "timeout": 30
                }
            }
        
        algotune_config = config.get("algotune", {})
        data_size = algotune_config.get("data_size", 5)
        timeout_seconds = algotune_config.get("timeout", 30)
        
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_solver"):
            return {"runs_successfully": 0.0, "error": "Missing run_solver function"}

        # Get the original task for reference solutions and problem generation
        task_class = None
        if "svm" in TASK_REGISTRY:
            task_class = TASK_REGISTRY["svm"]
        else:
            print(f"Error: svm task not found in TASK_REGISTRY")
            print(f"Available tasks: {list(TASK_REGISTRY.keys())}")

        try:
            # Run a single trial with timeout using proper task-specific problem
            if task_class:
                task_instance = task_class()
                test_problem = task_instance.generate_problem(n=data_size, random_seed=42)
            else:
                # Generic fallback test problem
                test_problem = {"test_data": [1, 2, 3], "random_seed": 42}
            
            result = run_with_timeout(program.run_solver, args=(test_problem,), timeout_seconds=timeout_seconds)

            # Basic validity check
            if result is not None:
                return {
                    "runs_successfully": 1.0,
                    "basic_functionality": 1.0,
                }
            else:
                return {
                    "runs_successfully": 0.5,
                    "basic_functionality": 0.0,
                    "error": "Function returned None"
                }

        except TimeoutError as e:
            return {"runs_successfully": 0.0, "error": "Timeout"}
        except Exception as e:
            return {"runs_successfully": 0.0, "error": str(e)}

    except Exception as e:
        return {"runs_successfully": 0.0, "error": str(e)}

def evaluate_stage2(program_path, config=None):
    """Second stage evaluation with more thorough testing of the evolved solve method"""
    return evaluate(program_path, config)
