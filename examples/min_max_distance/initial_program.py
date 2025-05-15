"""
Initial implementation for minimizing max/min distance ratio

This program generates points in 2D and calculates the ratio between
the maximum and minimum pairwise distances.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# EVOLVE-BLOCK-START
def generate_points(num_points: int, seed: int = 42) -> np.ndarray:
    """
    Generate points in 2D space
    
    Args:
        num_points: Number of points to generate
        seed: Random seed
        
    Returns:
        Array of shape (num_points, 2) with point coordinates
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate points randomly on a unit circle
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.ones(num_points)  # Uniform radius
    
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Combine into array
    points = np.column_stack((x, y))
    
    return points


def optimize_points(num_points: int, iterations: int = 1000) -> np.ndarray:
    """
    Optimize the placement of points to minimize max/min distance ratio
    
    Args:
        num_points: Number of points
        iterations: Number of optimization iterations
        
    Returns:
        Optimized array of shape (num_points, 2) with point coordinates
    """
    # Start with points on a circle
    points = generate_points(num_points)
    
    # Simple optimization: try small random perturbations
    best_points = points.copy()
    best_ratio = calculate_max_min_ratio(best_points)
    
    for i in range(iterations):
        # Apply small random perturbation
        perturbed_points = points + np.random.normal(0, 0.01, points.shape)
        
        # Calculate new ratio
        ratio = calculate_max_min_ratio(perturbed_points)
        
        # Update if better
        if ratio < best_ratio:
            best_ratio = ratio
            best_points = perturbed_points.copy()
            points = perturbed_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END


def calculate_max_min_ratio(points: np.ndarray) -> float:
    """
    Calculate the ratio between maximum and minimum pairwise distances
    
    Args:
        points: Array of shape (n, 2) with point coordinates
        
    Returns:
        Ratio between maximum and minimum distances
    """
    # Calculate pairwise distances
    n = points.shape[0]
    distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            distances.append(dist)
    
    # Find maximum and minimum distances
    max_dist = max(distances)
    min_dist = min(distances)
    
    # Handle edge case of min_dist = 0
    if min_dist < 1e-10:
        return float('inf')
    
    # Calculate ratio
    ratio = max_dist / min_dist
    
    return ratio


def visualize_points(points: np.ndarray, title: str = 'Point Configuration'):
    """
    Visualize the points and their distances
    
    Args:
        points: Array of shape (n, 2) with point coordinates
        title: Plot title
    """
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
    
    # Calculate ratio
    ratio = max_dist / min_dist
    
    # Create plot
    plt.figure(figsize=(10, 10))
    
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
    
    # Set plot properties
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'{title}\nRatio = {ratio:.4f}')
    plt.legend()
    
    return ratio


if __name__ == "__main__":
    # Generate and optimize points
    num_points = 16
    optimized_points = optimize_points(num_points)
    
    # Calculate and display the ratio
    ratio = calculate_max_min_ratio(optimized_points)
    print(f"Max/min distance ratio: {ratio:.6f}")
    
    # Visualize the points
    visualize_points(optimized_points)
    plt.savefig('optimized_points.png')
    plt.show()
