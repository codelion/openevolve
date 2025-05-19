"""
Evaluator for circle packing example (n=26)
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
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")

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
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False
    
    return True

def evaluate(program_path):
    """
    Evaluate the program by running it for n=26 and checking the sum of radii
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metrics
    """
    # Target value from the paper
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26
    
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_packing"):
            print(f"Error: program does not have 'run_packing' function")
            return {"sum_radii": 0.0, "validity": 0.0, "combined_score": 0.0}
        
        # Run multiple trials to assess reliability
        num_trials = 3
        successful_trials = 0
        best_sum = 0.0
        avg_sum = 0.0
        total_time = 0.0
        
        for trial in range(num_trials):
            try:
                start_time = time.time()
                
                # Run packing with timeout
                centers, radii, reported_sum = run_with_timeout(
                    program.run_packing, 
                    timeout_seconds=300
                )
                
                end_time = time.time()
                trial_time = end_time - start_time
                total_time += trial_time
                
                # Ensure centers and radii are numpy arrays
                if not isinstance(centers, np.ndarray):
                    centers = np.array(centers)
                if not isinstance(radii, np.ndarray):
                    radii = np.array(radii)
                
                # Validate solution
                valid = validate_packing(centers, radii)
                
                # Check shape and size
                shape_valid = (centers.shape == (26, 2) and radii.shape == (26,))
                if not shape_valid:
                    print(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)")
                    valid = False
                
                # Recalculate sum to verify
                calculated_sum = np.sum(radii) if valid else 0.0
                
                # Make sure reported_sum matches the calculated sum
                if abs(calculated_sum - reported_sum) > 1e-6:
                    print(f"Warning: Reported sum {reported_sum} doesn't match calculated sum {calculated_sum}")
                
                if valid:
                    successful_trials += 1
                    avg_sum += calculated_sum
                    best_sum = max(best_sum, calculated_sum)
                    
                print(f"Trial {trial+1}: valid={valid}, sum_radii={calculated_sum:.6f}, time={trial_time:.2f}s")
                
            except TimeoutError as e:
                print(f"Timeout in trial {trial+1}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error in trial {trial+1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Calculate metrics
        reliability = successful_trials / num_trials if num_trials > 0 else 0.0
        avg_sum = avg_sum / successful_trials if successful_trials > 0 else 0.0
        avg_time = total_time / num_trials if num_trials > 0 else 0.0
        
        # Target ratio (how close we are to the target)
        target_ratio = best_sum / TARGET_VALUE if best_sum > 0 else 0.0
        
        # Combined score - higher is better
        # - Weight reliability to reward consistency
        # - Weight target_ratio to reward proximity to target
        # - Small penalty for long running times
        combined_score = (0.3 * reliability + 0.7 * target_ratio) * (1.0 - min(1.0, avg_time / 30.0) * 0.1)
        
        print(f"Overall: best_sum={best_sum:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, reliability={reliability:.2f}")
        
        return {
            "sum_radii": float(best_sum),
            "avg_sum_radii": float(avg_sum),
            "target_ratio": float(target_ratio),
            "reliability": float(reliability),
            "avg_time": float(avg_time),
            "combined_score": float(combined_score)
        }
        
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "sum_radii": 0.0,
            "avg_sum_radii": 0.0,
            "target_ratio": 0.0,
            "reliability": 0.0,
            "avg_time": 0.0,
            "combined_score": 0.0
        }

# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
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
            # Run with a limited timeout for quick checking
            centers, radii, sum_radii = run_with_timeout(
                program.run_packing, 
                timeout_seconds=100
            )
            
            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)
            
            # Validate solution (shapes and constraints)
            shape_valid = (centers.shape == (26, 2) and radii.shape == (26,))
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
                "target_ratio": float(actual_sum / target if valid else 0.0)
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
    Second stage evaluation - full evaluation
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
