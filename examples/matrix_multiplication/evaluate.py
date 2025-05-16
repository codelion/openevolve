"""
Evaluation function for matrix multiplication tensor decomposition

This file defines the evaluation function that OpenEvolve will use to
score tensor decomposition implementations for matrix multiplication.
"""
import importlib.util
import logging
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Target shapes to evaluate
TARGET_SHAPES = [
    (2, 2, 2),  # Classic 2x2 matrix multiplication 
    (3, 3, 3),  # 3x3 matrix multiplication
    (4, 4, 4),  # 4x4 matrix multiplication (Strassen gives 49 multiplications)
    (4, 4, 5),  # Rectangular matrices
]

# Known best ranks for each shape (from literature)
BEST_KNOWN_RANKS = {
    (2, 2, 2): 7,    # Strassen's algorithm
    (3, 3, 3): 23,   # Laderman's algorithm
    (4, 4, 4): 49,   # Recursive Strassen (paper found 48 with complex values)
    (4, 4, 5): 62,   # From the AlphaEvolve paper
}

# Maximum rank to try for each shape (to prevent excessive computation)
MAX_RANKS = {
    (2, 2, 2): 10,
    (3, 3, 3): 30,
    (4, 4, 4): 60,
    (4, 4, 5): 75
}

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate a tensor decomposition implementation for matrix multiplication
    
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
        
        if not hasattr(module, "TensorDecomposition"):
            raise AttributeError(f"Program does not contain a 'TensorDecomposition' class")
        
        TensorDecomposition = module.TensorDecomposition
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"correctness": 0.0, "rank_quality": 0.0, "time_efficiency": 0.0}
    
    # Test correctness of the decomposition
    correctness_score = evaluate_correctness(TensorDecomposition)
    
    # If correctness fails, return early
    if correctness_score < 0.8:  # Allow some numerical issues
        return {
            "correctness": correctness_score, 
            "rank_quality": 0.0, 
            "time_efficiency": 0.0
        }
    
    # Test rank quality (how low a rank we can achieve)
    rank_quality_score = evaluate_rank_quality(TensorDecomposition)
    
    # Test time efficiency (how fast we can find a good decomposition)
    time_efficiency_score = evaluate_time_efficiency(TensorDecomposition)
    
    # Overall score weighted heavily towards rank quality
    overall_score = 0.2 * correctness_score + 0.6 * rank_quality_score + 0.2 * time_efficiency_score
    
    return {
        "correctness": correctness_score,
        "rank_quality": rank_quality_score,
        "time_efficiency": time_efficiency_score,
        "overall_score": overall_score,
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
        
        if not hasattr(module, "TensorDecomposition"):
            raise AttributeError(f"Program does not contain a 'TensorDecomposition' class")
        
        TensorDecomposition = module.TensorDecomposition
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"correctness": 0.0}
    
    # Test correctness
    correctness_score = evaluate_correctness(TensorDecomposition)
    
    return {"correctness": correctness_score}


def evaluate_correctness(TensorDecomposition) -> float:
    """
    Test the correctness of a tensor decomposition implementation
    
    Args:
        TensorDecomposition: Class to evaluate
        
    Returns:
        Correctness score (0.0 to 1.0)
    """
    # Define small test cases that are quick to evaluate
    test_cases = [
        {"shape": (2, 2, 2), "rank": 7},  # Should work with Strassen's algorithm
        {"shape": (2, 3, 2), "rank": 11}, # Should be possible
    ]
    
    passed = 0
    total = len(test_cases)
    reconstruction_errors = []
    
    for case in test_cases:
        try:
            shape = case["shape"]
            rank = case["rank"]
            
            # Create and optimize tensor decomposition
            td = TensorDecomposition(target_shape=shape, rank=rank, 
                                   config={"training_steps": 1000})
            result = td.optimize(rng_seed=42)
            
            # Check reconstruction error
            error = result["reconstruction_error"]
            reconstruction_errors.append(error)
            
            # If error is small enough, consider it a pass
            if error < 1e-2:  # Relaxed threshold for numerical stability
                passed += 1
                
        except Exception as e:
            logger.warning(f"Error in correctness test for case {case}: {str(e)}")
    
    # Calculate score based on passes and errors
    pass_score = passed / total if total > 0 else 0.0
    
    # If we have any errors, also factor in their magnitude
    if reconstruction_errors:
        avg_error = sum(reconstruction_errors) / len(reconstruction_errors)
        error_score = max(0, 1.0 - min(1.0, avg_error * 10))  # Scale error up
        
        # Combine scores (weighted more towards passing tests)
        return 0.7 * pass_score + 0.3 * error_score
    else:
        return pass_score


def binary_search_min_rank(TensorDecomposition, shape, min_rank, max_rank, 
                          error_threshold=1e-2, max_attempts=3) -> int:
    """
    Binary search to find the minimum rank needed for a valid decomposition
    
    Args:
        TensorDecomposition: Class to evaluate
        shape: Target tensor shape
        min_rank: Minimum rank to try
        max_rank: Maximum rank to try
        error_threshold: Maximum allowed reconstruction error
        max_attempts: Number of attempts per rank
        
    Returns:
        Minimum rank for a valid decomposition
    """
    logger.info(f"Binary searching for minimum rank for shape {shape} between {min_rank} and {max_rank}")
    
    if min_rank > max_rank:
        return max_rank
    
    while min_rank < max_rank:
        mid_rank = (min_rank + max_rank) // 2
        
        # Try multiple attempts at this rank
        success = False
        for attempt in range(max_attempts):
            try:
                td = TensorDecomposition(target_shape=shape, rank=mid_rank, 
                                       config={"training_steps": 2000})
                result = td.optimize(rng_seed=42 + attempt)
                
                # If error is small enough, consider it a success
                if result["reconstruction_error"] < error_threshold:
                    success = True
                    break
            except Exception as e:
                logger.warning(f"Error in attempt {attempt} for rank {mid_rank}: {str(e)}")
        
        if success:
            # We can try a lower rank
            max_rank = mid_rank
        else:
            # We need a higher rank
            min_rank = mid_rank + 1
    
    return min_rank


def evaluate_rank_quality(TensorDecomposition) -> float:
    """
    Test the quality of ranks found by the tensor decomposition implementation
    
    Args:
        TensorDecomposition: Class to evaluate
        
    Returns:
        Rank quality score (0.0 to 1.0+)
    """
    scores = []
    
    for shape in TARGET_SHAPES:
        best_known = BEST_KNOWN_RANKS.get(shape, float('inf'))
        max_rank = MAX_RANKS.get(shape, 2 * best_known)
        
        try:
            # First, verify we can at least match a higher rank
            verification_rank = min(best_known + 5, max_rank)
            td = TensorDecomposition(target_shape=shape, rank=verification_rank, 
                                   config={"training_steps": 1000})
            result = td.optimize(rng_seed=42)
            
            # If verification fails, skip this shape
            if result["reconstruction_error"] > 1e-2:
                logger.warning(f"Verification failed for shape {shape} at rank {verification_rank}")
                scores.append(0.0)
                continue
            
            # Binary search for the minimum rank (with reduced search space)
            # Using simplified linear search for faster evaluation
            min_found_rank = max_rank
            
            # Try a few decreasing ranks from the best known
            for test_rank in range(best_known, best_known - 5, -1):
                if test_rank < 1:
                    break
                    
                td = TensorDecomposition(target_shape=shape, rank=test_rank, 
                                       config={"training_steps": 2000})
                result = td.optimize(rng_seed=42)
                
                if result["reconstruction_error"] < 1e-2:
                    min_found_rank = test_rank
                    break
            
            # If we couldn't improve, try to match best known
            if min_found_rank > best_known:
                td = TensorDecomposition(target_shape=shape, rank=best_known, 
                                       config={"training_steps": 3000})
                result = td.optimize(rng_seed=42)
                
                if result["reconstruction_error"] < 1e-2:
                    min_found_rank = best_known
                else:
                    min_found_rank = verification_rank  # Use the verified rank
            
            # Calculate rank quality based on comparison to best known
            # If we match or beat best known, score >= 1.0
            # If we're slightly worse, score is proportionally lower
            quality = best_known / min_found_rank  
            
            # Bonus for beating best known
            if min_found_rank < best_known:
                bonus = 0.2 * (best_known - min_found_rank) / best_known
                quality += bonus
            
            scores.append(quality)
            
            logger.info(f"Shape {shape}: best known {best_known}, found {min_found_rank}, quality {quality:.3f}")
            
        except Exception as e:
            logger.warning(f"Error in rank quality test for shape {shape}: {str(e)}")
            scores.append(0.0)
    
    # Average scores across all shapes, with higher weight on larger matrices
    weights = [0.1, 0.2, 0.4, 0.3]  # More weight on 4x4 and 4x4x5
    
    weighted_score = 0.0
    total_weight = 0.0
    
    for i, score in enumerate(scores):
        weight = weights[i] if i < len(weights) else 0.1
        weighted_score += score * weight
        total_weight += weight
    
    avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
    return avg_score


def evaluate_time_efficiency(TensorDecomposition) -> float:
    """
    Test how quickly the implementation can find a good decomposition
    
    Args:
        TensorDecomposition: Class to evaluate
        
    Returns:
        Time efficiency score (0.0 to 1.0)
    """
    # Define baseline times (seconds) for standard cases
    baseline_times = {
        (2, 2, 2): 1.0,   # Should be fairly quick
        (3, 3, 3): 5.0,   # More challenging
    }
    
    scores = []
    
    for shape, baseline in baseline_times.items():
        best_known = BEST_KNOWN_RANKS.get(shape, 7)
        
        try:
            # Measure time to find a valid decomposition
            start_time = time.time()
            
            td = TensorDecomposition(target_shape=shape, rank=best_known, 
                                   config={"training_steps": 1000})
            result = td.optimize(rng_seed=42)
            
            elapsed_time = time.time() - start_time
            
            # Check if the decomposition is valid
            if result["reconstruction_error"] < 1e-2:
                # Calculate speed score (higher is better)
                speed_score = min(2.0, baseline / max(0.1, elapsed_time))
                scores.append(speed_score)
                logger.info(f"Shape {shape}: time {elapsed_time:.3f}s, score {speed_score:.3f}")
            else:
                logger.warning(f"No valid decomposition found for shape {shape}")
                scores.append(0.0)
                
        except Exception as e:
            logger.warning(f"Error in time efficiency test for shape {shape}: {str(e)}")
            scores.append(0.0)
    
    # Average scores
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return min(1.0, avg_score)  # Cap at 1.0


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
        print(f"Evaluating {program_path}")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Run evaluation
        scores = evaluate(program_path)
        
        print("\nEvaluation results:")
        for name, value in scores.items():
            print(f"  {name}: {value:.4f}")
    else:
        print("Usage: python evaluate.py <program_path>")
