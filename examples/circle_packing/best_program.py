# EVOLVE-BLOCK-START
"""Advanced circle packing for n=26 circles in a unit square"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    using mathematical principles and optimization techniques.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26

    # Initial guess: Strategic placement with some randomness
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Heuristic placement for better initial guess: place larger circles in center
    radii[:] = np.linspace(0.12, 0.05, n)  # Linear distribution of radii

    # Initial placement: approximate hexagonal grid
    grid_x = int(np.sqrt(n))
    grid_y = int(n / grid_x)

    x_coords = np.linspace(0.15, 0.85, grid_x)
    y_coords = np.linspace(0.15, 0.85, grid_y)

    count = 0
    for i in range(grid_x):
        for j in range(grid_y):
            if count < n:
                centers[count] = [x_coords[i] + 0.05 * (j % 2), y_coords[j]]
                count += 1

    # Place remaining circles randomly
    while count < n:
        centers[count] = np.random.rand(2) * 0.7 + 0.15
        count += 1

    # Objective function: Negative sum of radii (to maximize)
    def objective(x):
        centers = x[: 2 * n].reshape(n, 2)
        radii = x[2 * n :]
        return -np.sum(radii)

    # Constraint: No overlaps and circles stay within the unit square
    def constraint(x):
        centers = x[: 2 * n].reshape(n, 2)
        radii = x[2 * n :]

        # Overlap constraint
        overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                overlap_constraints.append(dist - (radii[i] + radii[j]))

        # Boundary constraints
        boundary_constraints = []
        for i in range(n):
            boundary_constraints.append(centers[i, 0] - radii[i])  # x >= radius
            boundary_constraints.append(1 - centers[i, 0] - radii[i])  # x <= 1 - radius
            boundary_constraints.append(centers[i, 1] - radii[i])  # y >= radius
            boundary_constraints.append(1 - centers[i, 1] - radii[i])  # y <= 1 - radius

        return np.array(overlap_constraints + boundary_constraints)

    # Initial guess vector
    x0 = np.concatenate([centers.flatten(), radii])

    # Bounds: Circles stay within the unit square and radii are positive
    bounds = [(0, 1)] * (2 * n) + [(0.03, 0.2)] * n  # radii are positive, up to 0.2

    # Constraints dictionary
    constraints = {"type": "ineq", "fun": constraint}

    # Optimization using SLSQP
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-8},
    )

    # Extract optimized centers and radii
    optimized_centers = result.x[: 2 * n].reshape(n, 2)
    optimized_radii = result.x[2 * n :]

    # Ensure radii are not negative (numerical stability)
    optimized_radii = np.maximum(optimized_radii, 0.001)

    # Calculate the sum of radii
    sum_radii = np.sum(optimized_radii)

    return optimized_centers, optimized_radii, sum_radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)
