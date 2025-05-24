"""
Evaluator for MLX kernel optimization example
"""

import importlib.util
import time
import traceback
import numpy as np
import mlx.core as mx
import psutil


def evaluate(program_path):
    """
    Evaluate the MLX kernel optimization program
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of performance metrics
    """
    
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_optimization"):
            return {
                "avg_gflops": 0.0,
                "total_time": 999.0,
                "efficiency_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing run_optimization function"
            }
        
        # Run the optimization with timeout
        start_time = time.time()
        
        try:
            results, avg_gflops, total_compute_time, device_info = program.run_optimization()
        except Exception as e:
            return {
                "avg_gflops": 0.0,
                "total_time": 999.0,
                "efficiency_score": 0.0,
                "combined_score": 0.0,
                "error": f"Execution failed: {str(e)}"
            }
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # Validate results
        if not isinstance(avg_gflops, (int, float)) or avg_gflops <= 0:
            return {
                "avg_gflops": 0.0,
                "total_time": 999.0,
                "efficiency_score": 0.0,
                "combined_score": 0.0,
                "error": "Invalid GFLOPS result"
            }
        
        if not isinstance(total_compute_time, (int, float)) or total_compute_time <= 0:
            return {
                "avg_gflops": 0.0,
                "total_time": 999.0,
                "efficiency_score": 0.0,
                "combined_score": 0.0,
                "error": "Invalid timing result"
            }
        
        # Calculate performance metrics
        
        # 1. GFLOPS score - higher is better
        # Baseline: ~100 GFLOPS is decent, 200+ is good, 500+ is excellent
        gflops_score = min(avg_gflops / 500.0, 2.0)  # Cap at 2.0 for 500+ GFLOPS
        
        # 2. Speed score - lower compute time is better
        # Baseline: ~0.1s total is good, less is better
        speed_score = min(1.0 / (total_compute_time + 0.01), 10.0)  # Cap at 10.0
        
        # 3. Efficiency score - balance between performance and time
        efficiency_score = gflops_score * speed_score / 10.0  # Normalize
        
        # 4. Memory efficiency - analyze tile choices
        memory_efficiency = calculate_memory_efficiency(results)
        
        # 5. Consistency score - how consistent performance is across different matrix sizes
        consistency_score = calculate_consistency_score(results)
        
        # 6. Overall combined score 
        # Emphasize GFLOPS performance but also consider efficiency and consistency
        combined_score = (
            0.5 * gflops_score +           # 50% - raw performance
            0.2 * efficiency_score +       # 20% - efficiency  
            0.15 * memory_efficiency +     # 15% - memory usage
            0.15 * consistency_score       # 15% - consistency
        )
        
        # Additional metrics for analysis
        return {
            "avg_gflops": float(avg_gflops),
            "total_time": float(total_compute_time),
            "evaluation_time": float(evaluation_time),
            "gflops_score": float(gflops_score),
            "speed_score": float(speed_score),
            "efficiency_score": float(efficiency_score),
            "memory_efficiency": float(memory_efficiency),
            "consistency_score": float(consistency_score),
            "combined_score": float(combined_score),
            "num_test_cases": len(results),
            "device_memory_gb": device_info.get("memory_gb", 0.0)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "avg_gflops": 0.0,
            "total_time": 999.0,
            "efficiency_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def calculate_memory_efficiency(results):
    """
    Calculate memory efficiency based on tile choices
    
    Args:
        results: List of benchmark results
        
    Returns:
        Memory efficiency score (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    total_efficiency = 0.0
    
    for result in results:
        matrix_size = result["matrix_size"]
        tile_size = result["tile_size"]
        metrics = result["metrics"]
        
        M, N, K = matrix_size
        tile_M, tile_N, tile_K = tile_size
        
        # Calculate tile utilization
        matrix_elements = M * N * K
        tile_elements = tile_M * tile_N * tile_K
        
        # Prefer tiles that are not too small (underutilize) or too large (memory pressure)
        if matrix_elements > 0:
            tile_ratio = tile_elements / matrix_elements
            
            # Optimal tile ratio is around 0.01 to 0.1 (1% to 10% of total matrix)
            if 0.001 <= tile_ratio <= 0.1:
                utilization_score = 1.0
            elif tile_ratio < 0.001:
                utilization_score = tile_ratio / 0.001  # Penalize very small tiles
            else:
                utilization_score = 0.1 / tile_ratio  # Penalize very large tiles
        else:
            utilization_score = 0.0
        
        # Also consider memory bandwidth utilization
        bandwidth_score = min(metrics.get("memory_bandwidth_gbs", 0) / 100.0, 1.0)
        
        # Combine utilization and bandwidth
        efficiency = 0.7 * utilization_score + 0.3 * bandwidth_score
        total_efficiency += efficiency
    
    return total_efficiency / len(results)


def calculate_consistency_score(results):
    """
    Calculate how consistent the performance is across different matrix sizes
    
    Args:
        results: List of benchmark results
        
    Returns:
        Consistency score (0.0 to 1.0)
    """
    if len(results) < 2:
        return 1.0
    
    # Extract GFLOPS values
    gflops_values = [result["metrics"]["gflops"] for result in results]
    
    if not gflops_values or max(gflops_values) == 0:
        return 0.0
    
    # Calculate coefficient of variation (std/mean)
    mean_gflops = np.mean(gflops_values)
    std_gflops = np.std(gflops_values)
    
    if mean_gflops == 0:
        return 0.0
    
    cv = std_gflops / mean_gflops
    
    # Convert to consistency score (lower coefficient of variation = higher consistency)
    # Good consistency has CV < 0.2, excellent has CV < 0.1
    consistency_score = max(0.0, 1.0 - cv / 0.3)  # Normalize so CV=0.3 gives score=0
    
    return consistency_score


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation
    """
    try:
        # Load and validate the program structure
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check required functions exist
        if not hasattr(program, "run_optimization"):
            return {"valid_structure": 0.0, "error": "Missing run_optimization function"}
        
        if not hasattr(program, "choose_tile_size"):
            return {"valid_structure": 0.0, "error": "Missing choose_tile_size function"}
        
        # Quick test of choose_tile_size function
        try:
            device_info = {"chip": "Test", "memory_gb": 8.0, "cpu_count": 8}
            tile_M, tile_N, tile_K = program.choose_tile_size(256, 256, 256, device_info)
            
            # Validate tile sizes are reasonable
            if not (1 <= tile_M <= 256 and 1 <= tile_N <= 256 and 1 <= tile_K <= 256):
                return {"valid_structure": 0.5, "error": "Invalid tile sizes returned"}
            
            return {
                "valid_structure": 1.0,
                "tile_example": [int(tile_M), int(tile_N), int(tile_K)]
            }
            
        except Exception as e:
            return {"valid_structure": 0.3, "error": f"Tile function error: {str(e)}"}
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """
    Second stage evaluation - limited performance test
    """
    try:
        # Run a subset of the full evaluation
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Test on just a few matrix sizes
        device_info = program.get_device_info()
        
        # Quick performance test
        M, N, K = 512, 512, 512
        tile_M, tile_N, tile_K = program.choose_tile_size(M, N, K, device_info)
        metrics = program.benchmark_configuration(M, N, K, tile_M, tile_N, tile_K, num_runs=2)
        
        # Basic performance scoring
        gflops = metrics["gflops"]
        gflops_score = min(gflops / 100.0, 2.0)  # Baseline 100 GFLOPS
        
        return {
            "valid_structure": 1.0,
            "quick_gflops": float(gflops),
            "quick_score": float(gflops_score),
            "passes_stage2": gflops_score > 0.5
        }
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": str(e)}


def evaluate_stage3(program_path):
    """
    Third stage evaluation - full performance evaluation
    """
    return evaluate(program_path)
