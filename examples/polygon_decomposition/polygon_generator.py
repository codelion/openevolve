"""
Rectilinear Polygon Generator

Generates random rectilinear polygons using a simplified approach based on
randomly placing rectangles that touch/overlap, then extracting the outer boundary.

This creates valid rectilinear polygons with varying complexity suitable for
testing rectangle decomposition algorithms.
"""

import random
from typing import List, Tuple, Set
import numpy as np


class RectilinearPolygonGenerator:
    """
    Generates rectilinear polygons by placing random rectangles and
    extracting the boundary.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_polygon(
        self,
        num_rectangles: int = 5,
        grid_size: int = 20,
        min_rect_size: int = 3,
        max_rect_size: int = 8,
    ) -> List[Tuple[int, int]]:
        """
        Generate a rectilinear polygon by placing rectangles and extracting boundary.

        Args:
            num_rectangles: Number of rectangles to use
            grid_size: Size of the grid (coordinates will be in [0, grid_size])
            min_rect_size: Minimum rectangle dimension
            max_rect_size: Maximum rectangle dimension

        Returns:
            List of (x, y) vertices defining the polygon boundary (clockwise)
        """
        # Create a grid to track occupied cells
        grid = np.zeros((grid_size + 1, grid_size + 1), dtype=bool)

        # Place first rectangle at center
        first_w = random.randint(min_rect_size, max_rect_size)
        first_h = random.randint(min_rect_size, max_rect_size)
        first_x = (grid_size - first_w) // 2
        first_y = (grid_size - first_h) // 2

        rectangles = [(first_x, first_y, first_x + first_w, first_y + first_h)]
        grid[first_y : first_y + first_h, first_x : first_x + first_w] = True

        # Place additional rectangles adjacent to existing ones
        for _ in range(num_rectangles - 1):
            # Pick a random existing rectangle
            base_rect = random.choice(rectangles)
            x1, y1, x2, y2 = base_rect

            # Generate new rectangle
            w = random.randint(min_rect_size, max_rect_size)
            h = random.randint(min_rect_size, max_rect_size)

            # Try to place adjacent to one of the four sides
            positions = []

            # Right side
            if x2 + w <= grid_size:
                positions.append((x2, y1, x2 + w, y1 + h))

            # Left side
            if x1 - w >= 0:
                positions.append((x1 - w, y1, x1, y1 + h))

            # Top side
            if y2 + h <= grid_size:
                positions.append((x1, y2, x1 + w, y2 + h))

            # Bottom side
            if y1 - h >= 0:
                positions.append((x1, y1 - h, x1 + w, y1))

            if positions:
                new_rect = random.choice(positions)
                x1, y1, x2, y2 = new_rect

                # Ensure it's within bounds
                if 0 <= x1 < x2 <= grid_size and 0 <= y1 < y2 <= grid_size:
                    rectangles.append(new_rect)
                    grid[y1:y2, x1:x2] = True

        # Extract boundary from the grid
        vertices = self._extract_boundary(grid, grid_size)

        return vertices

    def _extract_boundary(
        self, grid: np.ndarray, grid_size: int
    ) -> List[Tuple[int, int]]:
        """
        Extract the outer boundary of the filled region as a rectilinear polygon.

        Args:
            grid: Boolean grid of occupied cells
            grid_size: Size of the grid

        Returns:
            List of vertices in clockwise order
        """
        # Find all edges (transitions from filled to empty)
        edges = []

        # Horizontal edges
        for y in range(grid_size + 1):
            for x in range(grid_size):
                # Check vertical transition
                above = grid[y, x] if y < grid_size else False
                below = grid[y + 1, x] if y + 1 <= grid_size else False

                if above and not below:
                    # Bottom edge of cell
                    edges.append(((x, y + 1), (x + 1, y + 1), "horizontal"))
                elif below and not above:
                    # Top edge of cell
                    edges.append(((x, y), (x + 1, y), "horizontal"))

        # Vertical edges
        for y in range(grid_size):
            for x in range(grid_size + 1):
                # Check horizontal transition
                left = grid[y, x - 1] if x > 0 else False
                right = grid[y, x] if x < grid_size else False

                if right and not left:
                    # Left edge of cell
                    edges.append(((x, y), (x, y + 1), "vertical"))
                elif left and not right:
                    # Right edge of cell
                    edges.append(((x, y), (x, y + 1), "vertical"))

        # Trace the outer boundary
        if not edges:
            # Default to a simple square if no edges found
            return [(1, 1), (grid_size - 1, 1), (grid_size - 1, grid_size - 1), (1, grid_size - 1)]

        # Build adjacency for boundary tracing
        # Start from leftmost, bottommost edge
        boundary = self._trace_boundary(edges)

        return boundary

    def _trace_boundary(self, edges: List[Tuple]) -> List[Tuple[int, int]]:
        """
        Trace the boundary from a list of edges.

        Args:
            edges: List of ((x1, y1), (x2, y2), direction) tuples

        Returns:
            List of vertices forming the boundary
        """
        if not edges:
            return [(0, 0), (10, 0), (10, 10), (0, 10)]

        # Build adjacency map
        edge_map = {}
        for start, end, direction in edges:
            if start not in edge_map:
                edge_map[start] = []
            edge_map[start].append(end)

        # Find starting point (leftmost, then bottommost)
        start_point = min(edge_map.keys(), key=lambda p: (p[0], p[1]))

        # Trace boundary
        boundary = [start_point]
        current = start_point
        visited = set()

        max_iterations = len(edges) * 2  # Prevent infinite loops
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            if current in visited:
                # Check if we've returned to start
                if current == start_point and len(boundary) > 2:
                    break

            visited.add(current)

            # Find next point
            if current in edge_map:
                neighbors = edge_map[current]

                # Choose next based on right-hand rule (clockwise traversal)
                if neighbors:
                    # Pick the first unvisited neighbor, or any if all visited
                    next_point = None
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            next_point = neighbor
                            break

                    if next_point is None:
                        next_point = neighbors[0]

                    if next_point != current:
                        boundary.append(next_point)
                        current = next_point
                    else:
                        break
                else:
                    break
            else:
                break

        # Remove duplicate consecutive points
        cleaned = [boundary[0]]
        for i in range(1, len(boundary)):
            if boundary[i] != cleaned[-1]:
                cleaned.append(boundary[i])

        # Remove the last point if it's the same as the first
        if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned.pop()

        return cleaned

    def generate_test_set(
        self, num_polygons: int = 10, difficulty: str = "medium"
    ) -> List[List[Tuple[int, int]]]:
        """
        Generate a set of test polygons with varying complexity.

        Args:
            num_polygons: Number of polygons to generate
            difficulty: "easy", "medium", or "hard"

        Returns:
            List of polygon vertex lists
        """
        if difficulty == "easy":
            params = {
                "num_rectangles": (3, 5),
                "grid_size": 15,
                "min_rect_size": 3,
                "max_rect_size": 6,
            }
        elif difficulty == "hard":
            params = {
                "num_rectangles": (8, 12),
                "grid_size": 30,
                "min_rect_size": 2,
                "max_rect_size": 8,
            }
        else:  # medium
            params = {
                "num_rectangles": (5, 8),
                "grid_size": 20,
                "min_rect_size": 3,
                "max_rect_size": 7,
            }

        polygons = []
        for _ in range(num_polygons):
            num_rects = random.randint(*params["num_rectangles"])
            polygon = self.generate_polygon(
                num_rectangles=num_rects,
                grid_size=params["grid_size"],
                min_rect_size=params["min_rect_size"],
                max_rect_size=params["max_rect_size"],
            )
            polygons.append(polygon)

        return polygons


def visualize_polygon(vertices: List[Tuple[int, int]], rectangles: List = None):
    """
    Visualize a rectilinear polygon and optionally its decomposition.

    Args:
        vertices: Polygon vertices
        rectangles: Optional list of Rectangle namedtuples from decomposition
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MPLRectangle, Polygon
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw polygon
    polygon = Polygon(vertices, fill=False, edgecolor="blue", linewidth=2)
    ax.add_patch(polygon)

    # Draw rectangles if provided
    if rectangles:
        for rect in rectangles:
            width = rect.x_max - rect.x_min
            height = rect.y_max - rect.y_min
            mpl_rect = MPLRectangle(
                (rect.x_min, rect.y_min),
                width,
                height,
                fill=True,
                facecolor="lightblue",
                edgecolor="red",
                linewidth=1,
                alpha=0.3,
            )
            ax.add_patch(mpl_rect)

    # Set axis properties
    if vertices:
        xs, ys = zip(*vertices)
        margin = 2
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Rectilinear Polygon (vertices: {len(vertices)})")

    plt.show()


if __name__ == "__main__":
    # Test the generator
    generator = RectilinearPolygonGenerator(seed=42)

    print("Generating test polygon...")
    polygon = generator.generate_polygon(num_rectangles=6, grid_size=20)

    print(f"Generated polygon with {len(polygon)} vertices:")
    print(f"Vertices: {polygon[:5]}... (showing first 5)")

    # Visualize
    print("\nVisualizing...")
    visualize_polygon(polygon)

    # Generate test set
    print("\nGenerating test set...")
    test_set = generator.generate_test_set(num_polygons=5, difficulty="medium")
    print(f"Generated {len(test_set)} test polygons")
    for i, poly in enumerate(test_set):
        print(f"  Polygon {i}: {len(poly)} vertices")
