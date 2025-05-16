"""
Evaluation function for matrix multiplication optimization

This file defines the evaluation function that OpenEvolve will use to
score matrix multiplication implementations.
"""
import importlib.util
import logging
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate a matrix multiplication implementation
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metric name to score
    """
    # Import the program
    try:
        spec = importlib.util.spec_from_file_location("program_module", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {program_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["program_module"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, "matrix_multiply"):
            raise AttributeError(f"Program does not contain a 'matrix_multiply' function")
        
        matrix_multiply = module.matrix_multiply
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"correctness": 0.0, "performance": 0.0}
    
    # Test correctness
    correctness_score = evaluate_correctness(matrix_multiply)
    
    # If correctness fails, return early
    if correctness_score < 1.0:
        return {"correctness": correctness_score, "performance": 0.0}
    
    # Test performance
    performance_score = evaluate_performance(matrix_multiply)
    
    return {
        "correctness": correctness_score,
        "performance": performance_score,
    }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    First stage of evaluation: test correctness
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metric name to score
    """
    # Import the program
    try:
        spec = importlib.util.spec_from_file_location("program_module", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {program_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["program_module"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, "matrix_multiply"):
            raise AttributeError(f"Program does not contain a 'matrix_multiply' function")
        
        matrix_multiply = module.matrix_multiply
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"correctness": 0.0}
    
    # Test correctness
    correctness_score = evaluate_correctness(matrix_multiply)
    
    return {"correctness": correctness_score}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Second stage of evaluation: test performance
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metric name to score
    """
    # Import the program
    try:
        spec = importlib.util.spec_from_file_location("program_module", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {program_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["program_module"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, "matrix_multiply"):
            raise AttributeError(f"Program does not contain a 'matrix_multiply' function")
        
        matrix_multiply = module.matrix_multiply
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"performance": 0.0}
    
    # Test performance
    performance_score = evaluate_performance(matrix_multiply)
    
    return {"performance": performance_score}


def evaluate_correctness(matrix_multiply) -> float:
    """
    Test the correctness of a matrix multiplication implementation
    
    Args:
        matrix_multiply: Function to evaluate
        
    Returns:
        Correctness score (0.0 to 1.0)
    """
    # Define test cases focused on smaller matrices (as in the paper)
    test_sizes = [
        (2, 2, 2),
        (2, 3, 2),
        (3, 3, 3),
        (3, 4, 5),
        (4, 4, 4),
        (4, 5, 3),
        (5, 5, 5),
    ]
    
    passed = 0
    total = len(test_sizes)
    
    for m, n, p in test_sizes:
        try:
            # Create random matrices
            A = np.random.rand(m, n)
            B = np.random.rand(n, p)
            
            # Calculate reference result
            reference = np.matmul(A, B)
            
            # Calculate result with the implementation
            result = matrix_multiply(A, B)
            
            # Check if results are close
            if np.allclose(reference, result, rtol=1e-5, atol=1e-8):
                passed += 1
        except Exception as e:
            logger.warning(f"Error in correctness test for sizes {(m, n, p)}: {str(e)}")
    
    return passed / total


def evaluate_performance(matrix_multiply) -> float:
    """
    Test the performance of a matrix multiplication implementation
    
    Args:
        matrix_multiply: Function to evaluate
        
    Returns:
        Performance score (0.0 to 1.0)
    """
    # Define benchmark sizes focused on smaller matrices (as in the paper)
    benchmark_sizes = [
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (5, 5, 5),
        (3, 4, 5),
        (4, 3, 5),
    ]
    
    # Define baseline times for the naive triple-loop implementation
    # Calibrating based on typical performance of the naive implementation
    # These should be adjusted based on the actual machine running the benchmarks
    baseline_times = {
        "2x2x2": 0.00010,  # Small matrix, very fast
        "3x3x3": 0.00030,  # Still quite small
        "4x4x4": 0.00070,  # Medium sized 
        "5x5x5": 0.00150,  # Larger matrix
        "3x4x5": 0.00070,  # Rectangular matrices 
        "4x3x5": 0.00070,  # Rectangular matrices
    }
    
    # Define target speedups (what we're aiming for)
    # Based on Strassen's algorithm and other optimized approaches
    # We make these more ambitious to encourage more optimization
    target_speedups = {
        "2x2x2": 3.0,   # 3x faster than naive
        "3x3x3": 3.5,   # 3.5x faster than naive
        "4x4x4": 4.0,   # 4x faster than naive - Strassen's algorithm should be able to achieve this
        "5x5x5": 4.5,   # 4.5x faster than naive
        "3x4x5": 3.5,   # 3.5x faster than naive
        "4x3x5": 3.5,   # 3.5x faster than naive
    }
    
    # Run benchmark
    results = {}
    runs = 5  # More runs for better accuracy
    
    for m, n, p in benchmark_sizes:
        size_key = f"{m}x{n}x{p}"
        
        try:
            # Create random matrices
            A = np.random.rand(m, n)
            B = np.random.rand(n, p)
            
            # Warm-up
            matrix_multiply(A, B)
            
            # Benchmark
            times = []
            for _ in range(runs):
                start_time = time.time()
                matrix_multiply(A, B)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Record average time (remove fastest and slowest)
            times.sort()
            if len(times) > 2:
                times = times[1:-1]  # Remove extremes
            avg_time = sum(times) / len(times)
            results[size_key] = avg_time
        except Exception as e:
            logger.warning(f"Error in performance test for sizes {(m, n, p)}: {str(e)}")
            results[size_key] = baseline_times[size_key] * 2  # Penalize errors
    
    # Calculate speedups relative to baseline
    speedups = {}
    for size, time_taken in results.items():
        if time_taken > 0:
            speedups[size] = baseline_times[size] / time_taken
        else:
            speedups[size] = 0
    
    # Calculate relative performance to targets
    target_percentages = {}
    for size, speedup in speedups.items():
        target = target_speedups[size]
        # If speedup is below 1.0, it's worse than baseline (score 0.0-0.2)
        # If speedup equals baseline, score is 0.2
        # If speedup is between baseline and target, score is 0.2-0.8
        # If speedup reaches target, score is 0.8
        # If speedup exceeds target, score INCREASES BEYOND 0.8 proportionally
        if speedup < 1.0:
            target_percentages[size] = 0.2 * speedup
        elif speedup < target:
            # Linear interpolation between 0.2 and 0.8
            progress = (speedup - 1.0) / (target - 1.0)
            target_percentages[size] = 0.2 + 0.6 * progress
        else:
            # Speedup reached or exceeded target - NO CAP ON BONUS
            # This allows scores above 1.0 for exceptional performance
            bonus = (speedup - target) / target
            target_percentages[size] = 0.8 + 0.2 * bonus
    
    # Calculate overall score (average of target percentages)
    if not target_percentages:
        return 0.0
    
    # Calculate weighted average score - giving more weight to larger matrices
    # This encourages optimizations that work well on bigger matrices
    weights = {
        "2x2x2": 0.10,
        "3x3x3": 0.15,
        "4x4x4": 0.20,
        "5x5x5": 0.25,
        "3x4x5": 0.15,
        "4x3x5": 0.15
    }
    
    weighted_score = 0.0
    total_weight = 0.0
    
    for size, score in target_percentages.items():
        weight = weights.get(size, 1.0) 
        weighted_score += score * weight
        total_weight += weight
    
    avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
    
    # Log detailed results for debugging
    logger.info(f"Performance results:")
    for size in benchmark_sizes:
        size_key = f"{size[0]}x{size[1]}x{size[2]}"
        if size_key in results and size_key in speedups and size_key in target_percentages:
            logger.info(f"  {size_key}: time={results[size_key]:.6f}s, speedup={speedups[size_key]:.2f}x, score={target_percentages[size_key]:.2f}")
    
    return avg_score
