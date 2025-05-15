"""
Evaluation function for minimizing max/min distance ratio

This file defines the evaluation function that OpenEvolve will use to
score point configurations.
"""
import importlib.util
import logging
import sys
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
NUM_POINTS = 16  # This can be modified by the optimize.py script
SEEDS = [42, 123, 456, 789, 101112]  # Multiple seeds for robustness
REFERENCE_RATIO = 1.334  # Known baseline ratio for 16 points (from literature)


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate a point configuration generator
    
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
        
        if not hasattr(module, "optimize_points"):
            raise AttributeError(f"Program does not contain an 'optimize_points' function")
        
        optimize_points = module.optimize_points
        calculate_max_min_ratio = module.calculate_max_min_ratio
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"ratio_score": 0.0, "stability": 0.0}
    
    # Run optimization for multiple seeds
    all_ratios = []
    
    for seed in SEEDS:
        try:
            # Generate points with the current seed
            points = optimize_points(NUM_POINTS, iterations=100)
            
            # Calculate ratio
            ratio = calculate_max_min_ratio(points)
            
            # Store the ratio
            all_ratios.append(ratio)
        except Exception as e:
            logger.warning(f"Error in seed {seed}: {str(e)}")
            # Penalize errors with a high ratio
            all_ratios.append(REFERENCE_RATIO * 2)
    
    # Calculate best ratio
    if not all_ratios:
        return {"ratio_score": 0.0, "stability": 0.0}
    
    best_ratio = min(all_ratios)
    
    # Calculate stability (consistency across seeds)
    if len(all_ratios) > 1:
        stability = 1.0 - (np.std(all_ratios) / np.mean(all_ratios))
        stability = max(0.0, min(1.0, stability))
    else:
        stability = 0.0
    
    # Calculate ratio score
    # 1.0 means achieving the reference ratio
    # > 1.0 means improving beyond reference (normalized to avoid excessive rewards)
    ratio_score = min(1.5, REFERENCE_RATIO / best_ratio)
    
    return {
        "ratio_score": ratio_score,
        "stability": stability,
    }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    First stage of evaluation: quick check with single seed
    
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
        
        if not hasattr(module, "optimize_points"):
            raise AttributeError(f"Program does not contain an 'optimize_points' function")
        
        optimize_points = module.optimize_points
        calculate_max_min_ratio = module.calculate_max_min_ratio
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"quick_check": 0.0}
    
    # Run optimization with a single seed
    try:
        # Generate points
        points = optimize_points(NUM_POINTS, iterations=50)
        
        # Calculate ratio
        ratio = calculate_max_min_ratio(points)
        
        # Calculate score
        quick_score = min(1.0, REFERENCE_RATIO / ratio)
        
        return {"quick_check": quick_score}
    except Exception as e:
        logger.warning(f"Error in quick check: {str(e)}")
        return {"quick_check": 0.0}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Second stage of evaluation: comprehensive check with multiple seeds
    
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
        
        if not hasattr(module, "optimize_points"):
            raise AttributeError(f"Program does not contain an 'optimize_points' function")
        
        optimize_points = module.optimize_points
        calculate_max_min_ratio = module.calculate_max_min_ratio
    except Exception as e:
        logger.error(f"Error importing program: {str(e)}")
        return {"ratio_score": 0.0, "stability": 0.0}
    
    # Run optimization for multiple seeds
    all_ratios = []
    
    for seed in SEEDS:
        try:
            # Generate points with the current seed
            points = optimize_points(NUM_POINTS, iterations=100)
            
            # Calculate ratio
            ratio = calculate_max_min_ratio(points)
            
            # Store the ratio
            all_ratios.append(ratio)
        except Exception as e:
            logger.warning(f"Error in seed {seed}: {str(e)}")
            # Penalize errors with a high ratio
            all_ratios.append(REFERENCE_RATIO * 2)
    
    # Calculate best ratio
    if not all_ratios:
        return {"ratio_score": 0.0, "stability": 0.0}
    
    best_ratio = min(all_ratios)
    
    # Calculate stability (consistency across seeds)
    if len(all_ratios) > 1:
        stability = 1.0 - (np.std(all_ratios) / np.mean(all_ratios))
        stability = max(0.0, min(1.0, stability))
    else:
        stability = 0.0
    
    # Calculate ratio score
    # 1.0 means achieving the reference ratio
    # > 1.0 means improving beyond reference (normalized to avoid excessive rewards)
    ratio_score = min(1.5, REFERENCE_RATIO / best_ratio)
    
    return {
        "ratio_score": ratio_score,
        "stability": stability,
    }
