"""
Evaluator for the function minimization example
"""
import importlib.util
import numpy as np
import time

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
    
    # Load the program
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run multiple trials
    num_trials = 10
    values = []
    distances = []
    times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        x, y, value = program.run_search()
        end_time = time.time()
        
        # Calculate metrics
        distance_to_global = np.sqrt((x - GLOBAL_MIN_X)**2 + (y - GLOBAL_MIN_Y)**2)
        value_difference = abs(value - GLOBAL_MIN_VALUE)
        
        values.append(value)
        distances.append(distance_to_global)
        times.append(end_time - start_time)
    
    # Calculate metrics
    avg_value = np.mean(values)
    avg_distance = np.mean(distances)
    avg_time = np.mean(times)
    
    # Convert to scores (higher is better)
    value_score = 1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE))  # Normalize and invert
    distance_score = 1.0 / (1.0 + avg_distance)
    speed_score = 1.0 / avg_time
    
    # Normalize speed score (so it doesn't dominate)
    speed_score = min(speed_score, 10.0) / 10.0
    
    return {
        "value_score": value_score,
        "distance_score": distance_score,
        "speed_score": speed_score,
        "combined_score": 0.6 * value_score + 0.3 * distance_score + 0.1 * speed_score
    }

# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """First stage evaluation with fewer trials"""
    # Quick check to see if the program runs without errors
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Run a single trial
        x, y, value = program.run_search()
        
        # Basic metrics
        return {
            "runs_successfully": 1.0,
            "value": float(value)
        }
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        return {"runs_successfully": 0.0}

def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing"""
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
