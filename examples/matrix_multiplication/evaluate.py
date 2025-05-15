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
    # Define test cases
    test_sizes = [
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (10, 10, 10),
        (3, 4, 5),
        (7, 3, 8),
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
    # Define benchmark sizes
    benchmark_sizes = [
        (10, 10, 10),
        (20, 20, 20),
        (30, 30, 30),
        (40, 40, 40),
    ]
    
    # Define baseline times (naive implementation)
    # These would be measured in advance for the baseline implementation
    baseline_times = {
        "10x10x10": 0.0015,  # seconds
        "20x20x20": 0.0120,  # seconds
        "30x30x30": 0.0400,  # seconds
        "40x40x40": 0.0950,  # seconds
    }
    
    # Run benchmark
    results = {}
    runs = 3
    
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
            
            # Record average time
            avg_time = sum(times) / runs
            results[size_key] = avg_time
        except Exception as e:
            logger.warning(f"Error in performance test for sizes {(m, n, p)}: {str(e)}")
            results[size_key] = baseline_times[size_key] * 2  # Penalize errors
    
    # Calculate speedups
    speedups = {}
    for size, time_taken in results.items():
        if time_taken > 0:
            speedups[size] = baseline_times[size] / time_taken
        else:
            speedups[size] = 0
    
    # Calculate overall score (geometric mean of speedups)
    if not speedups:
        return 0.0
    
    # Remove any zero speedups
    valid_speedups = [s for s in speedups.values() if s > 0]
    if not valid_speedups:
        return 0.0
    
    # Calculate geometric mean
    import math
    log_sum = sum(math.log(s) for s in valid_speedups)
    geom_mean = math.exp(log_sum / len(valid_speedups))
    
    # Normalize to 0.0-1.0 range (assuming baseline = 1.0)
    # Values above 1.0 indicate improvement, below 1.0 indicate regression
    # Cap at 5.0x speedup for scoring purposes
    normalized_score = min(geom_mean / 5.0, 1.0)
    
    return normalized_score
