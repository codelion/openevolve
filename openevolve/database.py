"""
Program database for OpenEvolve
"""

import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from openevolve.config import DatabaseConfig
from openevolve.utils.code_utils import calculate_edit_distance

logger = logging.getLogger(__name__)


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        return cls(**data)


class ProgramDatabase:
    """
    Database for storing and sampling programs during evolution

    The database implements a combination of MAP-Elites algorithm and
    island-based population model to maintain diversity during evolution.
    It also tracks the absolute best program separately to ensure it's never lost.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

        # In-memory program storage
        self.programs: Dict[str, Program] = {}

        # Feature grid for MAP-Elites
        self.feature_map: Dict[str, str] = {}
        self.feature_bins = config.feature_bins

        # Island populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Archive of elite programs
        self.archive: Set[str] = set()

        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        logger.info(f"Initialized program database with {len(self.programs)} programs")

    def add(self, program: Program) -> str:
        """
        Add a program to the database

        Args:
            program: Program to add

        Returns:
            Program ID
        """
        # Store the program
        self.programs[program.id] = program

        # Calculate feature coordinates for MAP-Elites
        feature_coords = self._calculate_feature_coords(program)

        # Add to feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords)
        if feature_key not in self.feature_map or self._is_better(
            program, self.programs[self.feature_map[feature_key]]
        ):
            self.feature_map[feature_key] = program.id

        # Add to an island (randomly)
        island_idx = random.randint(0, len(self.islands) - 1)
        self.islands[island_idx].add(program.id)

        # Update archive
        self._update_archive(program)

        # Update the absolute best program tracking
        self._update_best_program(program)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id} to database")
        return program.id

    def get(self, program_id: str) -> Optional[Program]:
        """
        Get a program by ID

        Args:
            program_id: Program ID

        Returns:
            Program or None if not found
        """
        return self.programs.get(program_id)

    def sample(self) -> Tuple[Program, List[Program]]:
        """
        Sample a program and inspirations for the next evolution step

        Returns:
            Tuple of (parent_program, inspiration_programs)
        """
        # Select parent program
        parent = self._sample_parent()

        # Select inspirations
        inspirations = self._sample_inspirations(parent, n=5)

        logger.debug(f"Sampled parent {parent.id} and {len(inspirations)} inspirations")
        return parent, inspirations

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """
        Get the best program based on a metric

        Args:
            metric: Metric to use for ranking (uses combined_score or average if None)

        Returns:
            Best program or None if database is empty
        """
        if not self.programs:
            return None

        # If no specific metric and we have a tracked best program, return it
        if metric is None and self.best_program_id and self.best_program_id in self.programs:
            logger.debug(f"Using tracked best program: {self.best_program_id}")
            return self.programs[self.best_program_id]

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        elif self.programs and all("combined_score" in p.metrics for p in self.programs.values()):
            # Sort by combined_score if it exists (preferred method)
            sorted_programs = sorted(
                self.programs.values(), key=lambda p: p.metrics["combined_score"], reverse=True
            )
            if sorted_programs:
                logger.debug(f"Found best program by combined_score: {sorted_programs[0].id}")
        else:
            # Sort by average of all metrics as fallback
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: sum(p.metrics.values()) / max(1, len(p.metrics)),
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by average metrics: {sorted_programs[0].id}")

        # Update the best program tracking if we found a better program
        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(f"Updated best program tracking from {old_id} to {self.best_program_id}")

            # Also log the scores to help understand the update
            if (
                old_id
                and old_id in self.programs
                and "combined_score" in self.programs[old_id].metrics
                and "combined_score" in self.programs[self.best_program_id].metrics
            ):
                old_score = self.programs[old_id].metrics["combined_score"]
                new_score = self.programs[self.best_program_id].metrics["combined_score"]
                logger.info(
                    f"Score change: {old_score:.4f} → {new_score:.4f} ({new_score-old_score:+.4f})"
                )

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(self, n: int = 10, metric: Optional[str] = None) -> List[Program]:
        """
        Get the top N programs based on a metric

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking (uses average if None)

        Returns:
            List of top programs
        """
        if not self.programs:
            return []

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            # Sort by average of all metrics
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: sum(p.metrics.values()) / max(1, len(p.metrics)),
                reverse=True,
            )

        return sorted_programs[:n]

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            self._save_program(program, save_path)

        # Save metadata
        metadata = {
            "feature_map": self.feature_map,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved database with {len(self.programs)} programs to {save_path}")

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.feature_map = metadata.get("feature_map", {})
            self.islands = [set(island) for island in metadata.get("islands", [])]
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

    def _save_program(self, program: Program, base_path: Optional[str] = None) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_path = os.path.join(programs_dir, f"{program.id}.json")
        with open(program_path, "w") as f:
            json.dump(program.to_dict(), f)

    def _calculate_feature_coords(self, program: Program) -> List[int]:
        """
        Calculate feature coordinates for the MAP-Elites grid

        Args:
            program: Program to calculate features for

        Returns:
            List of feature coordinates
        """
        coords = []

        for dim in self.config.feature_dimensions:
            if dim == "complexity":
                # Use code length as complexity measure
                complexity = len(program.code)
                bin_idx = min(int(complexity / 1000 * self.feature_bins), self.feature_bins - 1)
                coords.append(bin_idx)
            elif dim == "diversity":
                # Use average edit distance to other programs
                if len(self.programs) < 5:
                    bin_idx = 0
                else:
                    sample_programs = random.sample(
                        list(self.programs.values()), min(5, len(self.programs))
                    )
                    avg_distance = sum(
                        calculate_edit_distance(program.code, other.code)
                        for other in sample_programs
                    ) / len(sample_programs)
                    bin_idx = min(
                        int(avg_distance / 1000 * self.feature_bins), self.feature_bins - 1
                    )
                coords.append(bin_idx)
            elif dim == "score":
                # Use average of metrics
                if not program.metrics:
                    bin_idx = 0
                else:
                    avg_score = sum(program.metrics.values()) / len(program.metrics)
                    bin_idx = min(int(avg_score * self.feature_bins), self.feature_bins - 1)
                coords.append(bin_idx)
            elif dim in program.metrics:
                # Use specific metric
                score = program.metrics[dim]
                bin_idx = min(int(score * self.feature_bins), self.feature_bins - 1)
                coords.append(bin_idx)
            else:
                # Default to middle bin if feature not found
                coords.append(self.feature_bins // 2)

        return coords

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        """
        Convert feature coordinates to a string key

        Args:
            coords: Feature coordinates

        Returns:
            String key
        """
        return "-".join(str(c) for c in coords)

    def _is_better(self, program1: Program, program2: Program) -> bool:
        """
        Determine if program1 is better than program2

        Args:
            program1: First program
            program2: Second program

        Returns:
            True if program1 is better than program2
        """
        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # If only one has metrics, it's better
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        # Check for combined_score first (this is the preferred metric)
        if "combined_score" in program1.metrics and "combined_score" in program2.metrics:
            return program1.metrics["combined_score"] > program2.metrics["combined_score"]

        # Fallback to average of all metrics
        avg1 = sum(program1.metrics.values()) / len(program1.metrics)
        avg2 = sum(program2.metrics.values()) / len(program2.metrics)

        return avg1 > avg2

    def _update_archive(self, program: Program) -> None:
        """
        Update the archive of elite programs

        Args:
            program: Program to consider for archive
        """
        # If archive not full, add program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Otherwise, find worst program in archive
        archive_programs = [self.programs[pid] for pid in self.archive]
        worst_program = min(
            archive_programs, key=lambda p: sum(p.metrics.values()) / max(1, len(p.metrics))
        )

        # Replace if new program is better
        if self._is_better(program, worst_program):
            self.archive.remove(worst_program.id)
            self.archive.add(program.id)

    def _update_best_program(self, program: Program) -> None:
        """
        Update the absolute best program tracking

        Args:
            program: Program to consider as the new best
        """
        # If we don't have a best program yet, this becomes the best
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return

        # Compare with current best program
        current_best = self.programs[self.best_program_id]

        # Update if the new program is better
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            # Log the change
            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    f"New best program {program.id} replaces {old_id} (combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _sample_parent(self) -> Program:
        """
        Sample a parent program for the next evolution step

        Returns:
            Parent program
        """
        # Decide between exploitation and exploration
        if random.random() < self.config.exploitation_ratio and self.archive:
            # Exploitation: Use elite program from archive
            parent_id = random.choice(list(self.archive))
            return self.programs[parent_id]

        # Exploration: Sample from an island
        island_idx = random.randint(0, len(self.islands) - 1)

        if not self.islands[island_idx]:
            # If island is empty, use best program
            return self.get_best_program() or next(iter(self.programs.values()))

        parent_id = random.choice(list(self.islands[island_idx]))
        return self.programs[parent_id]

    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]:
        """
        Sample inspiration programs for the next evolution step

        Args:
            parent: Parent program
            n: Number of inspirations to sample

        Returns:
            List of inspiration programs
        """
        inspirations = []

        # Always include the absolute best program if available and different from parent
        if self.best_program_id is not None and self.best_program_id != parent.id:
            best_program = self.programs[self.best_program_id]
            inspirations.append(best_program)
            logger.debug(f"Including best program {self.best_program_id} in inspirations")

        # Add top programs as inspirations
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_programs = self.get_top_programs(n=top_n)
        for program in top_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)

        # Add diverse programs
        if len(self.programs) > n and len(inspirations) < n:
            # Sample from different feature cells
            feature_coords = self._calculate_feature_coords(parent)

            # Get programs from nearby feature cells
            nearby_programs = []
            for _ in range(n - len(inspirations)):
                # Perturb coordinates
                perturbed_coords = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-1, 1)))
                    for c in feature_coords
                ]

                # Try to get program from this cell
                cell_key = self._feature_coords_to_key(perturbed_coords)
                if cell_key in self.feature_map:
                    program_id = self.feature_map[cell_key]
                    if program_id != parent.id and program_id not in [p.id for p in inspirations]:
                        nearby_programs.append(self.programs[program_id])

            # If we need more, add random programs
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)
                all_ids = set(self.programs.keys())
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_ids = list(all_ids - excluded_ids)

                if available_ids:
                    random_ids = random.sample(available_ids, min(remaining, len(available_ids)))
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        return inspirations[:n]
