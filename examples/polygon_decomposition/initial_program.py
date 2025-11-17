# EVOLVE-BLOCK-START
"""
Rectilinear Polygon Decomposition using CP-SAT

Decomposes rectilinear polygons into rectangles, minimizing:
1. Number of rectangles with sides < threshold (primary)
2. Total number of rectangles (secondary)
"""

from collections import namedtuple
from typing import List, Tuple, Optional
from matplotlib.path import Path
from ortools.sat.python import cp_model

Rectangle = namedtuple("Rectangle", ["x_min", "y_min", "x_max", "y_max"])


class SweepLineDecomposer:
    """
    Decomposes a rectilinear polygon into maximal rectangles using sweep line algorithm.
    """

    def __init__(self, polygon_vertices: List[Tuple[int, int]]):
        """
        Initialize the decomposer with polygon vertices.

        Args:
            polygon_vertices: List of (x, y) tuples defining the polygon
        """
        self.polygon_vertices = polygon_vertices
        self.polygon_path = Path(polygon_vertices)

        # Extract unique coordinates
        self.x_coords = sorted(set(v[0] for v in polygon_vertices))
        self.y_coords = sorted(set(v[1] for v in polygon_vertices))

    def decompose_sweep_line(self) -> List[Rectangle]:
        """
        Performs full sweep line decomposition to get all primitive rectangles.
        This creates the finest possible rectangular decomposition.
        """
        # Create grid from all x and y coordinates
        x_lines = sorted(set(v[0] for v in self.polygon_vertices))
        y_lines = sorted(set(v[1] for v in self.polygon_vertices))

        primitive_rectangles = []

        # Check each potential rectangle in the grid
        for i in range(len(x_lines) - 1):
            for j in range(len(y_lines) - 1):
                x_min = x_lines[i]
                x_max = x_lines[i + 1]
                y_min = y_lines[j]
                y_max = y_lines[j + 1]

                # Check if this rectangle is inside the polygon
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                if self.polygon_path.contains_point((center_x, center_y)):
                    rect = Rectangle(x_min, y_min, x_max, y_max)
                    primitive_rectangles.append(rect)

        return primitive_rectangles

    def find_maximal_rectangles_with_min_size(
        self, l_min: int, primitive_rects: Optional[List[Rectangle]] = None,
        time_limit: float = 300.0
    ) -> List[Rectangle]:
        """
        Produce a full partition of primitives into axis-aligned rectangles.

        Uses CP-SAT to enforce that each primitive is covered exactly once.

        Objective (lexicographic):
        1) Minimize number of rectangles whose width < l_min OR height < l_min
        2) Minimize total number of rectangles
        3) Maximize coverage (maximize sum of primitive counts)

        Args:
            l_min: Minimum acceptable side length
            primitive_rects: Pre-computed primitive rectangles (optional)
            time_limit: Solver time limit in seconds

        Returns:
            List[Rectangle] — all rectangles in the selected partition
        """
        if primitive_rects is None:
            primitive_rects = self.decompose_sweep_line()

        if not primitive_rects:
            return []

        num = len(primitive_rects)

        def area(r: Rectangle) -> float:
            return (r.x_max - r.x_min) * (r.y_max - r.y_min)

        # Build adjacency graph (edge-touching on full edge)
        adj = [[] for _ in range(num)]
        for i in range(num):
            r1 = primitive_rects[i]
            for j in range(i + 1, num):
                r2 = primitive_rects[j]
                # Horizontal adjacency
                horz = (
                    r1.y_min == r2.y_min
                    and r1.y_max == r2.y_max
                    and (
                        abs(r1.x_max - r2.x_min) < 1e-9
                        or abs(r2.x_max - r1.x_min) < 1e-9
                    )
                )
                # Vertical adjacency
                vert = (
                    r1.x_min == r2.x_min
                    and r1.x_max == r2.x_max
                    and (
                        abs(r1.y_max - r2.y_min) < 1e-9
                        or abs(r2.y_max - r1.y_min) < 1e-9
                    )
                )
                if horz or vert:
                    adj[i].append(j)
                    adj[j].append(i)

        prim_areas = [area(r) for r in primitive_rects]

        # Enumerate candidate merged rectangles
        seen_sets = set()
        candidates = []

        def bbox_of_indices(indices):
            xs = []
            ys = []
            for ii in indices:
                r = primitive_rects[ii]
                xs.append(r.x_min)
                xs.append(r.x_max)
                ys.append(r.y_min)
                ys.append(r.y_max)
            return Rectangle(min(xs), min(ys), max(xs), max(ys))

        def is_solid(indices, bbox):
            total = sum(prim_areas[ii] for ii in indices)
            return abs(total - area(bbox)) < 1e-6

        # BFS/DFS to find all connected solid rectangles
        for seed in range(num):
            start = frozenset([seed])
            stack = [start]
            while stack:
                curr = stack.pop()
                if curr in seen_sets:
                    continue
                seen_sets.add(curr)

                bbox = bbox_of_indices(curr)
                # If union exactly fills bbox -> candidate
                if is_solid(curr, bbox):
                    candidates.append(
                        {"indices": curr, "rect": bbox, "size": len(curr)}
                    )

                # Expand
                neighs = set()
                for ii in curr:
                    for n in adj[ii]:
                        if n not in curr:
                            neighs.add(n)
                for n in neighs:
                    newset = frozenset(set(curr) | {n})
                    if newset not in seen_sets:
                        stack.append(newset)

        if not candidates:
            return []

        # Remove duplicate candidates
        unique = {}
        for c in candidates:
            key = tuple(sorted(c["indices"]))
            if key not in unique or unique[key]["size"] < c["size"]:
                unique[key] = c
        candidates = list(unique.values())

        # Sort by size (heuristic)
        candidates.sort(key=lambda c: c["size"], reverse=True)
        m = len(candidates)

        # Precompute which primitives each candidate covers
        cand_primitives = [set(c["indices"]) for c in candidates]

        # Precompute whether candidate violates l_min
        bad = []
        for c in candidates:
            bbox = c["rect"]
            w = bbox.x_max - bbox.x_min
            h = bbox.y_max - bbox.y_min
            bad.append(1 if (w + 1e-9 < l_min or h + 1e-9 < l_min) else 0)

        # Build CP-SAT model
        model = cp_model.CpModel()
        cand_vars = [model.NewBoolVar(f"cand_{i}") for i in range(m)]

        # Full partition constraint: each primitive covered exactly once
        for prim_idx in range(num):
            covering_candidates = [
                i for i in range(m) if prim_idx in cand_primitives[i]
            ]
            if not covering_candidates:
                # Create singleton candidate for safety
                singleton_key = (prim_idx,)
                bbox = primitive_rects[prim_idx]
                new_idx = len(candidates)
                candidates.append(
                    {
                        "indices": frozenset([prim_idx]),
                        "rect": primitive_rects[prim_idx],
                        "size": 1,
                    }
                )
                cand_primitives.append({prim_idx})
                w = bbox.x_max - bbox.x_min
                h = bbox.y_max - bbox.y_min
                bad.append(1 if (w + 1e-9 < l_min or h + 1e-9 < l_min) else 0)
                var = model.NewBoolVar(f"cand_{new_idx}")
                cand_vars.append(var)
                covering_candidates = [new_idx]

            # Exactly-one constraint
            model.Add(sum(cand_vars[i] for i in covering_candidates) == 1)

        # Objective: lexicographic
        sizes = [c["size"] for c in candidates]
        total_bad = sum(bad[i] * cand_vars[i] for i in range(len(cand_vars)))
        total_selected = sum(cand_vars[i] for i in range(len(cand_vars)))
        total_covered = sum(sizes[i] * cand_vars[i] for i in range(len(cand_vars)))

        # Lexicographic weights
        big1 = (num + 1) * (num + 1)  # Primary: minimize bad rectangles
        big2 = num + 1  # Secondary: minimize total rectangles
        # Tertiary: maximize coverage (negative term)
        model.Minimize(big1 * total_bad + big2 * total_selected - total_covered)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        # Optional parallelization
        # solver.parameters.num_search_workers = 4
        status = solver.Solve(model)

        result_rects = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i in range(len(cand_vars)):
                try:
                    val = solver.Value(cand_vars[i])
                except Exception:
                    val = 0
                if val == 1:
                    result_rects.append(candidates[i]["rect"])

        # Ensure full coverage (add missing singletons if needed)
        selected_indices = set()
        for i in range(len(candidates)):
            try:
                if (
                    solver.StatusName(status) in ("OPTIMAL", "FEASIBLE")
                    and solver.Value(cand_vars[i]) == 1
                ):
                    selected_indices |= cand_primitives[i]
            except Exception:
                continue

        if len(selected_indices) < num:
            missing = set(range(num)) - selected_indices
            for prim_idx in missing:
                result_rects.append(primitive_rects[prim_idx])

        # Sort for determinism
        result_rects.sort(key=lambda r: (r.x_min, r.y_min, r.x_max, r.y_max))
        return result_rects


# EVOLVE-BLOCK-END


def decompose_polygon(
    polygon_vertices: List[Tuple[int, int]], l_min: int = 5, time_limit: float = 60.0
) -> Tuple[List[Rectangle], dict]:
    """
    Main entry point for polygon decomposition.

    Args:
        polygon_vertices: List of (x, y) vertices defining the polygon
        l_min: Minimum acceptable side length
        time_limit: Time limit for solver in seconds

    Returns:
        Tuple of (rectangles, stats) where stats contains:
        - num_rectangles: Total number of rectangles
        - num_violations: Number of rectangles with sides < l_min
        - success: Whether decomposition succeeded
    """
    try:
        decomposer = SweepLineDecomposer(polygon_vertices)
        rectangles = decomposer.find_maximal_rectangles_with_min_size(
            l_min, time_limit=time_limit
        )

        # Calculate violations
        num_violations = 0
        for rect in rectangles:
            w = rect.x_max - rect.x_min
            h = rect.y_max - rect.y_min
            if w < l_min or h < l_min:
                num_violations += 1

        stats = {
            "num_rectangles": len(rectangles),
            "num_violations": num_violations,
            "success": len(rectangles) > 0,
        }

        return rectangles, stats

    except Exception as e:
        # Return failure stats
        stats = {"num_rectangles": 0, "num_violations": 0, "success": False, "error": str(e)}
        return [], stats


if __name__ == "__main__":
    # Test with a simple polygon
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from polygon_generator import RectilinearPolygonGenerator, visualize_polygon

    # Generate test polygon
    generator = RectilinearPolygonGenerator(seed=42)
    polygon = generator.generate_polygon(num_rectangles=5, grid_size=20)

    print(f"Testing decomposition on polygon with {len(polygon)} vertices")
    print(f"Vertices: {polygon}")

    # Decompose
    l_min = 3
    rectangles, stats = decompose_polygon(polygon, l_min=l_min, time_limit=30.0)

    print(f"\nDecomposition results:")
    print(f"  Success: {stats['success']}")
    print(f"  Total rectangles: {stats['num_rectangles']}")
    print(f"  Violations (sides < {l_min}): {stats['num_violations']}")

    # Visualize
    if rectangles:
        print("\nVisualizing result...")
        visualize_polygon(polygon, rectangles)
