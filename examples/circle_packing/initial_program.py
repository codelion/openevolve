# EVOLVE-BLOCK-START
"""Circle packing optimization for OpenEvolve"""
import numpy as np


def pack_circles(n=26, iterations=10000):
    """
    Place n circles in a unit square to maximize sum of radii

    Args:
        n: Number of circles to pack
        iterations: Number of optimization iterations

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
        sum_of_radii: Sum of all radii (the optimization objective)
    """
    # Initialize with random centers
    centers = np.random.rand(n, 2)

    # Greedy algorithm to assign radii
    radii = compute_radii(centers)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)

    for i in range(iterations):
        # Randomly perturb a circle's position
        new_centers = centers.copy()
        idx = np.random.randint(0, n)
        new_centers[idx] += 0.01 * (np.random.rand(2) - 0.5)

        # Keep centers inside the unit square
        new_centers = np.clip(new_centers, 0, 1)

        # Compute new radii
        new_radii = compute_radii(new_centers)
        new_sum = np.sum(new_radii)

        # Update if better
        if new_sum > best_sum:
            best_centers = new_centers.copy()
            best_radii = new_radii.copy()
            best_sum = new_sum
            centers = new_centers.copy()
            radii = new_radii.copy()

    return best_centers, best_radii, best_sum


def compute_radii(centers):
    """
    Compute maximum possible radii for circles at given centers

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Initialize radii based on distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Adjust radii to avoid overlaps
    for i in range(n):
        for j in range(i + 1, n):
            # Distance between centers
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # Maximum radius sum to avoid overlap
            max_sum = dist

            # If current radii would cause overlap, scale them down
            if radii[i] + radii[j] > max_sum:
                scale = max_sum / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing(n=26):
    """Run the circle packing algorithm with specified n"""
    centers, radii, sum_radii = pack_circles(n)
    return centers, radii, sum_radii

def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha='center', va='center')
    
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)
    # AlphaEvolve improved this to 2.635 for n=26
