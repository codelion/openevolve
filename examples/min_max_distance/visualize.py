"""
Visualization utilities for point configurations
"""
import importlib.util
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def visualize_solution(program_path: str, output_file: Optional[str] = None):
    """
    Visualize the optimized point configuration
    
    Args:
        program_path: Path to the program file
        output_file: Path to save the visualization image
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
        print(f"Error importing program: {str(e)}")
        return
    
    # Get number of points from evaluate.py
    try:
        eval_path = os.path.join(os.path.dirname(program_path), "..", "evaluate.py")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_content = f.read()
            
            # Extract NUM_POINTS
            import re
            match = re.search(r"NUM_POINTS\s*=\s*(\d+)", eval_content)
            if match:
                num_points = int(match.group(1))
            else:
                num_points = 16  # Default
        else:
            num_points = 16  # Default
    except Exception:
        num_points = 16  # Default
    
    # Generate points with the best algorithm
    try:
        points = optimize_points(num_points, iterations=200)
        ratio = calculate_max_min_ratio(points)
    except Exception as e:
        print(f"Error generating points: {str(e)}")
        return
    
    # Visualize points
    n = points.shape[0]
    
    # Calculate distances
    min_dist = float('inf')
    min_pair = None
    max_dist = 0
    max_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            
            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)
            
            if dist > max_dist:
                max_dist = dist
                max_pair = (i, j)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=100)
    
    # Plot indices
    for i in range(n):
        plt.text(points[i, 0], points[i, 1], str(i), fontsize=12)
    
    # Plot minimum distance edge
    if min_pair is not None:
        i, j = min_pair
        plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'g-', linewidth=2, label=f'Min: {min_dist:.4f}')
    
    # Plot maximum distance edge
    if max_pair is not None:
        i, j = max_pair
        plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'r-', linewidth=2, label=f'Max: {max_dist:.4f}')
    
    # Plot all edges for visualization
    for i in range(n):
        for j in range(i + 1, n):
            plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'gray', linewidth=0.5, alpha=0.3)
    
    # Set plot properties
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Optimized Point Configuration (n={n})\nMax/Min Ratio = {ratio:.6f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize point configuration")
    parser.add_argument("program_path", help="Path to the program file")
    parser.add_argument("--output", "-o", help="Path to save the visualization image")
    
    args = parser.parse_args()
    
    # Visualize solution
    visualize_solution(args.program_path, args.output)
