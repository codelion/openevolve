"""
Evaluator for the function minimization example
"""
import importlib.util
import numpy as np
import time
import concurrent.futures
import threading

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
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

def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metrics
    """
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.76
    GLOBAL_MIN_Y = -1.03
    GLOBAL_MIN_VALUE = -2.104
    
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Error: program does not have 'run_search' function")
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing run_search function"
            }
        
        # Run multiple trials
        num_trials = 10
        values = []
        distances = []
        times = []
        success_count = 0
        
        for trial in range(num_trials):
            try:
                start_time = time.time()
                
                # Run with timeout
                x, y, value = run_with_timeout(program.run_search, timeout_seconds=5)
                
                end_time = time.time()
                
                # Check if the result is valid (not NaN or infinite)
                if (np.isnan(x) or np.isnan(y) or np.isnan(value) or 
                    np.isinf(x) or np.isinf(y) or np.isinf(value)):
                    print(f"Trial {trial}: Invalid result, got x={x}, y={y}, value={value}")
                    continue
                
                # Ensure all values are float
                x, y, value = float(x), float(y), float(value)
                
                # Calculate metrics
                distance_to_global = np.sqrt((x - GLOBAL_MIN_X)**2 + (y - GLOBAL_MIN_Y)**2)
                value_difference = abs(value - GLOBAL_MIN_VALUE)
                
                values.append(value)
                distances.append(distance_to_global)
                times.append(end_time - start_time)
                success_count += 1
                
            except TimeoutError as e:
                print(f"Trial {trial}: {str(e)}")
                continue
            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                continue
        
        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "All trials failed"
            }
        
        # Calculate metrics
        avg_value = np.mean(values)
        avg_distance = np.mean(distances)
        avg_time = np.mean(times)
        
        # Convert to scores (higher is better)
        value_score = 1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE))  # Normalize and invert
        distance_score = 1.0 / (1.0 + avg_distance)
        speed_score = 1.0 / avg_time if avg_time > 0 else 0.0
        
        # Normalize speed score (so it doesn't dominate)
        speed_score = min(speed_score, 10.0) / 10.0
        
        # Add reliability score based on success rate
        reliability_score = success_count / num_trials
        
        return {
            "value_score": value_score,
            "distance_score": distance_score,
            "speed_score": speed_score,
            "reliability_score": reliability_score,
            "combined_score": 0.5 * value_score + 0.2 * distance_score + 0.1 * speed_score + 0.2 * reliability_score,
            "success_rate": reliability_score
        }
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        return {
            "value_score": 0.0,
            "distance_score": 0.0,
            "speed_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }

# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """First stage evaluation with fewer trials"""
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.76
    GLOBAL_MIN_Y = -1.03
    GLOBAL_MIN_VALUE = -2.104
    
    # Quick check to see if the program runs without errors
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Stage 1 validation: Program does not have 'run_search' function")
            return {"runs_successfully": 0.0, "error": "Missing run_search function"}
        
        try:
            # Run a single trial with timeout
            x, y, value = run_with_timeout(program.run_search, timeout_seconds=5)
            
            # Ensure all values are float
            x, y, value = float(x), float(y), float(value)
            
            # Check if the result is valid
            if np.isnan(x) or np.isnan(y) or np.isnan(value) or np.isinf(x) or np.isinf(y) or np.isinf(value):
                print(f"Stage 1 validation: Invalid result, got x={x}, y={y}, value={value}")
                return {"runs_successfully": 0.5, "error": "Invalid result values"}
            
            # Basic metrics
            return {
                "runs_successfully": 1.0,
                "value": float(value),
                "distance": float(np.sqrt((x - GLOBAL_MIN_X)**2 + (y - GLOBAL_MIN_Y)**2))  # Distance to known minimum
            }
        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"runs_successfully": 0.0, "error": "Timeout"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            return {"runs_successfully": 0.0, "error": str(e)}
            
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        return {"runs_successfully": 0.0, "error": str(e)}

def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing"""
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
