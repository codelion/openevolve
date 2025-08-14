"""
Program database for OpenEvolve
"""

import base64
import json
import logging
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

# FileLock removed - no longer needed with threaded parallel processing
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from openevolve.config import DatabaseConfig
from openevolve.utils.code_utils import calculate_edit_distance
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


def _safe_sum_metrics(metrics: Dict[str, Any]) -> float:
    """Safely sum only numeric metric values, ignoring strings and other types"""
    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) if numeric_values else 0.0


def _safe_avg_metrics(metrics: Dict[str, Any]) -> float:
    """Safely calculate average of only numeric metric values"""
    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) / max(1, len(numeric_values)) if numeric_values else 0.0


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
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Artifact storage
    artifacts_json: Optional[str] = None  # JSON-serialized small artifacts
    artifact_dir: Optional[str] = None  # Path to large artifact files

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


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

        # Handle both int and dict types for feature_bins
        if isinstance(config.feature_bins, int):
            self.feature_bins = max(
                config.feature_bins,
                int(pow(config.archive_size, 1 / len(config.feature_dimensions)) + 0.99),
            )
        else:
            # If dict, keep as is (we'll use feature_bins_per_dim instead)
            self.feature_bins = 10  # Default fallback for backward compatibility

        # Island populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Island management attributes
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)  # Default to 10
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)  # Default to 0.1

        # Archive of elite programs
        self.archive: Set[str] = set()

        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Track best program per island for proper island-based evolution
        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        # Prompt log
        self.prompts_by_program: Dict[str, Dict[str, Dict[str, str]]] = None

        # Set random seed for reproducible sampling if specified
        if config.random_seed is not None:
            import random

            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

        # Diversity caching infrastructure
        self.diversity_cache: Dict[int, Dict[str, Union[float, float]]] = (
            {}
        )  # hash -> {"value": float, "timestamp": float}
        self.diversity_cache_size: int = 1000  # LRU cache size
        self.diversity_reference_set: List[str] = (
            []
        )  # Reference program codes for consistent diversity
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)

        # Feature scaling infrastructure
        self.feature_stats: Dict[str, Dict[str, Union[float, float, List[float]]]] = {}
        self.feature_scaling_method: str = "minmax"  # Options: minmax, zscore, percentile

        # Per-dimension bins support
        if hasattr(config, "feature_bins") and isinstance(config.feature_bins, dict):
            self.feature_bins_per_dim = config.feature_bins
        else:
            # Backward compatibility - use same bins for all dimensions
            self.feature_bins_per_dim = {
                dim: self.feature_bins for dim in config.feature_dimensions
            }

        logger.info(f"Initialized program database with {len(self.programs)} programs")

    def add(
        self, program: Program, iteration: int = None, target_island: Optional[int] = None
    ) -> str:
        """
        Add a program to the database

        Args:
            program: Program to add
            iteration: Current iteration (defaults to last_iteration)
            target_island: Specific island to add to (uses current_island if None)

        Returns:
            Program ID
        """
        # Store the program
        # If iteration is provided, update the program's iteration_found
        if iteration is not None:
            program.iteration_found = iteration
            # Update last_iteration if needed
            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program

        # Calculate feature coordinates for MAP-Elites
        feature_coords = self._calculate_feature_coords(program)

        # Add to feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords)
        should_replace = feature_key not in self.feature_map

        if not should_replace:
            # Check if the existing program still exists before comparing
            existing_program_id = self.feature_map[feature_key]
            if existing_program_id not in self.programs:
                # Stale reference, replace it
                should_replace = True
                logger.debug(
                    f"Replacing stale program reference {existing_program_id} in feature map"
                )
            else:
                # Program exists, compare fitness
                should_replace = self._is_better(program, self.programs[existing_program_id])

        if should_replace:
            # Log significant MAP-Elites events
            coords_dict = {
                self.config.feature_dimensions[i]: feature_coords[i]
                for i in range(len(feature_coords))
            }

            if feature_key not in self.feature_map:
                # New cell occupation
                logger.info("New MAP-Elites cell occupied: %s", coords_dict)
                # Check coverage milestone
                total_possible_cells = self.feature_bins ** len(self.config.feature_dimensions)
                coverage = (len(self.feature_map) + 1) / total_possible_cells
                if coverage in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    logger.info(
                        "MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                        coverage * 100,
                        len(self.feature_map) + 1,
                        total_possible_cells,
                    )
            else:
                # Cell replacement - existing program being replaced
                existing_program_id = self.feature_map[feature_key]
                if existing_program_id in self.programs:
                    existing_program = self.programs[existing_program_id]
                    new_fitness = safe_numeric_average(program.metrics)
                    existing_fitness = safe_numeric_average(existing_program.metrics)
                    logger.info(
                        "MAP-Elites cell improved: %s (fitness: %.3f -> %.3f)",
                        coords_dict,
                        existing_fitness,
                        new_fitness,
                    )

                    # use MAP-Elites to manage archive
                    if existing_program_id in self.archive:
                        self.archive.discard(existing_program_id)
                        self.archive.add(program.id)

            self.feature_map[feature_key] = program.id

        # Add to specific island (not random!)
        island_idx = target_island if target_island is not None else self.current_island
        island_idx = island_idx % len(self.islands)  # Ensure valid island
        self.islands[island_idx].add(program.id)

        # Track which island this program belongs to
        program.metadata["island"] = island_idx

        # Update archive
        self._update_archive(program)

        # Enforce population size limit BEFORE updating best program tracking
        # This ensures newly added programs aren't immediately removed
        self._enforce_population_limit(exclude_program_id=program.id)

        # Update the absolute best program tracking (after population enforcement)
        self._update_best_program(program)

        # Update island-specific best program tracking
        self._update_island_best_program(program, island_idx)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id} to island {island_idx}")

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
        if metric is None and self.best_program_id:
            if self.best_program_id in self.programs:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.programs[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

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
            # Sort by average of all numeric metrics as fallback
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: safe_numeric_average(p.metrics),
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

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:
        """
        Get the top N programs based on a metric

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking (uses average if None)
            island_idx: If specified, only return programs from this island

        Returns:
            List of top programs
        """
        # Validate island_idx parameter
        if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands)):
            raise IndexError(f"Island index {island_idx} is out of range (0-{len(self.islands)-1})")

        if not self.programs:
            return []

        # Get candidate programs
        if island_idx is not None:
            # Island-specific query
            island_programs = [
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            ]
            candidates = island_programs
        else:
            # Global query
            candidates = list(self.programs.values())

        if not candidates:
            return []

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            # Sort by average of all numeric metrics
            sorted_programs = sorted(
                candidates,
                key=lambda p: safe_numeric_average(p.metrics),
                reverse=True,
            )

        return sorted_programs[:n]

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Perform artifact cleanup before saving
        self._cleanup_old_artifacts(save_path)

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and program.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            "feature_map": self.feature_map,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "island_best_programs": self.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
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

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        saved_islands = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.feature_map = metadata.get("feature_map", {})
            saved_islands = metadata.get("islands", [])
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.island_best_programs = metadata.get(
                "island_best_programs", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get("island_generations", [0] * len(saved_islands))
            self.last_migration_generation = metadata.get("last_migration_generation", 0)

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")

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

        # Reconstruct island assignments from metadata
        self._reconstruct_islands(saved_islands)

        # Ensure island_generations list has correct length
        if len(self.island_generations) != len(self.islands):
            self.island_generations = [0] * len(self.islands)

        # Ensure island_best_programs list has correct length
        if len(self.island_best_programs) != len(self.islands):
            self.island_best_programs = [None] * len(self.islands)

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

        # Log the reconstructed island status
        self.log_island_status()

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:
        """
        Reconstruct island assignments from saved metadata

        Args:
            saved_islands: List of island program ID lists from metadata
        """
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.config.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        missing_programs = []
        restored_programs = 0

        # Restore island assignments
        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue

            for program_id in program_ids:
                if program_id in self.programs:
                    # Program exists, add to island
                    self.islands[island_idx].add(program_id)
                    # Set island metadata on the program
                    self.programs[program_id].metadata["island"] = island_idx
                    restored_programs += 1
                else:
                    # Program missing, track it
                    missing_programs.append((island_idx, program_id))

        # Clean up archive - remove missing programs
        original_archive_size = len(self.archive)
        self.archive = {pid for pid in self.archive if pid in self.programs}

        # Clean up feature_map - remove missing programs
        feature_keys_to_remove = []
        for key, program_id in self.feature_map.items():
            if program_id not in self.programs:
                feature_keys_to_remove.append(key)
        for key in feature_keys_to_remove:
            del self.feature_map[key]

        # Clean up island best programs - remove stale references
        self._cleanup_stale_island_bests()

        # Check best program
        if self.best_program_id and self.best_program_id not in self.programs:
            logger.warning(f"Best program {self.best_program_id} not found, will recalculate")
            self.best_program_id = None

        # Log reconstruction results
        if missing_programs:
            logger.warning(
                f"Found {len(missing_programs)} missing programs during island reconstruction:"
            )
            for island_idx, program_id in missing_programs[:5]:  # Show first 5
                logger.warning(f"  Island {island_idx}: {program_id}")
            if len(missing_programs) > 5:
                logger.warning(f"  ... and {len(missing_programs) - 5} more")

        if original_archive_size > len(self.archive):
            logger.info(
                f"Removed {original_archive_size - len(self.archive)} missing programs from archive"
            )

        if feature_keys_to_remove:
            logger.info(f"Removed {len(feature_keys_to_remove)} missing programs from feature map")

        logger.info(f"Reconstructed islands: restored {restored_programs} programs to islands")

        # If we have programs but no island assignments, distribute them
        if self.programs and sum(len(island) for island in self.islands) == 0:
            logger.info("No island assignments found, distributing programs across islands")
            self._distribute_programs_to_islands()

    def _distribute_programs_to_islands(self) -> None:
        """
        Distribute loaded programs across islands when no island metadata exists
        """
        program_ids = list(self.programs.keys())

        # Distribute programs round-robin across islands
        for i, program_id in enumerate(program_ids):
            island_idx = i % len(self.islands)
            self.islands[island_idx].add(program_id)
            self.programs[program_id].metadata["island"] = island_idx

        logger.info(f"Distributed {len(program_ids)} programs across {len(self.islands)} islands")

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program, in the format {template_key: { 'system': str, 'user': str }}
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

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
                bin_idx = self._calculate_complexity_bin(complexity)
                coords.append(bin_idx)
            elif dim == "diversity":
                # Use cached diversity calculation with reference set
                if len(self.programs) < 2:
                    bin_idx = 0
                else:
                    diversity = self._get_cached_diversity(program)
                    bin_idx = self._calculate_diversity_bin(diversity)
                coords.append(bin_idx)
            elif dim == "score":
                # Use average of numeric metrics
                if not program.metrics:
                    bin_idx = 0
                else:
                    avg_score = safe_numeric_average(program.metrics)
                    # Update stats and scale
                    self._update_feature_stats("score", avg_score)
                    scaled_value = self._scale_feature_value("score", avg_score)
                    num_bins = self.feature_bins_per_dim.get("score", self.feature_bins)
                    bin_idx = int(scaled_value * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif dim in program.metrics:
                # Use specific metric
                score = program.metrics[dim]
                # Update stats and scale
                self._update_feature_stats(dim, score)
                scaled_value = self._scale_feature_value(dim, score)
                num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            else:
                # Feature not found - this is an error
                raise ValueError(
                    f"Feature dimension '{dim}' specified in config but not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Either remove '{dim}' from feature_dimensions or ensure your evaluator returns it."
                )
        # Only log coordinates at debug level for troubleshooting
        logger.debug(
            "MAP-Elites coords: %s",
            str({self.config.feature_dimensions[i]: coords[i] for i in range(len(coords))}),
        )
        return coords

    def _calculate_complexity_bin(self, complexity: int) -> int:
        """
        Calculate the bin index for a given complexity value using feature scaling.

        Args:
            complexity: The complexity value (code length)

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics
        self._update_feature_stats("complexity", float(complexity))

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("complexity", float(complexity))

        # Get number of bins for this dimension
        num_bins = self.feature_bins_per_dim.get("complexity", self.feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _calculate_diversity_bin(self, diversity: float) -> int:
        """
        Calculate the bin index for a given diversity value using feature scaling.

        Args:
            diversity: The average fast code diversity to other programs

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics
        self._update_feature_stats("diversity", diversity)

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("diversity", diversity)

        # Get number of bins for this dimension
        num_bins = self.feature_bins_per_dim.get("diversity", self.feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

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

        # Fallback to average of all numeric metrics
        avg1 = safe_numeric_average(program1.metrics)
        avg2 = safe_numeric_average(program2.metrics)

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

        # Clean up stale references and get valid archive programs
        valid_archive_programs = []
        stale_ids = []

        for pid in self.archive:
            if pid in self.programs:
                valid_archive_programs.append(self.programs[pid])
            else:
                stale_ids.append(pid)

        # Remove stale references from archive
        for stale_id in stale_ids:
            self.archive.discard(stale_id)
            logger.debug(f"Removing stale program {stale_id} from archive")

        # If archive is now not full after cleanup, just add the new program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Find worst program among valid programs
        if valid_archive_programs:
            worst_program = min(
                valid_archive_programs, key=lambda p: safe_numeric_average(p.metrics)
            )

            # Replace if new program is better
            if self._is_better(program, worst_program):
                self.archive.remove(worst_program.id)
                self.archive.add(program.id)
        else:
            # No valid programs in archive, just add the new one
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

        # Compare with current best program (if it still exists)
        if self.best_program_id not in self.programs:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

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

    def _update_island_best_program(self, program: Program, island_idx: int) -> None:
        """
        Update the best program tracking for a specific island

        Args:
            program: Program to consider as the new best for the island
            island_idx: Island index
        """
        # Ensure island_idx is valid
        if island_idx >= len(self.island_best_programs):
            logger.warning(f"Invalid island index {island_idx}, skipping island best update")
            return

        # If island doesn't have a best program yet, this becomes the best
        current_island_best_id = self.island_best_programs[island_idx]
        if current_island_best_id is None:
            self.island_best_programs[island_idx] = program.id
            logger.debug(f"Set initial best program for island {island_idx} to {program.id}")
            return

        # Check if current best still exists
        if current_island_best_id not in self.programs:
            logger.warning(
                f"Island {island_idx} best program {current_island_best_id} no longer exists, updating to {program.id}"
            )
            self.island_best_programs[island_idx] = program.id
            return

        current_island_best = self.programs[current_island_best_id]

        # Update if the new program is better
        if self._is_better(program, current_island_best):
            old_id = current_island_best_id
            self.island_best_programs[island_idx] = program.id

            # Log the change
            if (
                "combined_score" in program.metrics
                and "combined_score" in current_island_best.metrics
            ):
                old_score = current_island_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id} "
                    f"(combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id}"
                )

    def _sample_parent(self) -> Program:
        """
        Sample a parent program from the current island for the next evolution step

        Returns:
            Parent program from current island
        """
        # Use exploration_ratio and exploitation_ratio to decide sampling strategy
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            # EXPLORATION: Sample from current island (diverse sampling)
            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite programs)
            return self._sample_exploitation_parent()
        else:
            # RANDOM: Sample from any program (remaining probability)
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """
        Sample a parent for exploration (from current island)
        """
        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:
            # If current island is empty, initialize with best program or random program
            if self.best_program_id and self.best_program_id in self.programs:
                # Create a copy of best program for the empty island (don't reuse same ID)
                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Initialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                # Use any available program
                return next(iter(self.programs.values()))

        # Clean up stale references and sample from current island
        valid_programs = [pid for pid in current_island_programs if pid in self.programs]

        # Remove stale program IDs from island
        if len(valid_programs) < len(current_island_programs):
            stale_ids = current_island_programs - set(valid_programs)
            logger.debug(
                f"Removing {len(stale_ids)} stale program IDs from island {self.current_island}"
            )
            for stale_id in stale_ids:
                self.islands[self.current_island].discard(stale_id)

        # If no valid programs after cleanup, reinitialize island
        if not valid_programs:
            logger.warning(
                f"Island {self.current_island} has no valid programs after cleanup, reinitializing"
            )
            if self.best_program_id and self.best_program_id in self.programs:
                # Create a copy of best program for the empty island (don't reuse same ID)
                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Reinitialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                return next(iter(self.programs.values()))

        # Sample from valid programs
        parent_id = random.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:
        """
        Sample a parent for exploitation (from archive/elite programs)
        """
        if not self.archive:
            # Fallback to exploration if no archive
            return self._sample_exploration_parent()

        # Clean up stale references in archive
        valid_archive = [pid for pid in self.archive if pid in self.programs]

        # Remove stale program IDs from archive
        if len(valid_archive) < len(self.archive):
            stale_ids = self.archive - set(valid_archive)
            logger.debug(f"Removing {len(stale_ids)} stale program IDs from archive")
            for stale_id in stale_ids:
                self.archive.discard(stale_id)

        # If no valid archive programs, fallback to exploration
        if not valid_archive:
            logger.warning(
                "Archive has no valid programs after cleanup, falling back to exploration"
            )
            return self._sample_exploration_parent()

        # Prefer programs from current island in archive
        archive_programs_in_island = [
            pid
            for pid in valid_archive
            if self.programs[pid].metadata.get("island") == self.current_island
        ]

        if archive_programs_in_island:
            parent_id = random.choice(archive_programs_in_island)
            return self.programs[parent_id]
        else:
            # Fall back to any valid archive program if current island has none
            parent_id = random.choice(valid_archive)
            return self.programs[parent_id]

    def _sample_random_parent(self) -> Program:
        """
        Sample a completely random parent from all programs
        """
        if not self.programs:
            raise ValueError("No programs available for sampling")

        # Sample randomly from all programs
        program_id = random.choice(list(self.programs.keys()))
        return self.programs[program_id]

    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]:
        """
        Sample inspiration programs for the next evolution step.

        For proper island-based evolution, inspirations are sampled ONLY from the
        current island, maintaining genetic isolation between islands.

        Args:
            parent: Parent program
            n: Number of inspirations to sample

        Returns:
            List of inspiration programs from the current island
        """
        inspirations = []

        # Get the parent's island (should be current_island)
        parent_island = parent.metadata.get("island", self.current_island)

        # Get all programs from the current island
        island_program_ids = list(self.islands[parent_island])
        island_programs = [self.programs[pid] for pid in island_program_ids if pid in self.programs]

        if not island_programs:
            logger.warning(f"Island {parent_island} has no programs for inspiration sampling")
            return []

        # Include the island's best program if available and different from parent
        island_best_id = self.island_best_programs[parent_island]
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.programs
        ):
            island_best = self.programs[island_best_id]
            inspirations.append(island_best)
            logger.debug(
                f"Including island {parent_island} best program {island_best_id} in inspirations"
            )
        elif island_best_id is not None and island_best_id not in self.programs:
            # Clean up stale island best reference
            logger.warning(
                f"Island {parent_island} best program {island_best_id} no longer exists, clearing reference"
            )
            self.island_best_programs[parent_island] = None

        # Add top programs from the island as inspirations
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island_programs = self.get_top_programs(n=top_n, island_idx=parent_island)
        for program in top_island_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)

        # Add diverse programs from within the island
        if len(island_programs) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)

            # Try to sample from different feature cells within the island
            feature_coords = self._calculate_feature_coords(parent)
            nearby_programs = []

            # Create a mapping of feature cells to island programs for efficient lookup
            island_feature_map = {}
            for prog_id in island_program_ids:
                if prog_id in self.programs:
                    prog = self.programs[prog_id]
                    prog_coords = self._calculate_feature_coords(prog)
                    cell_key = self._feature_coords_to_key(prog_coords)
                    island_feature_map[cell_key] = prog_id

            # Try to find programs from nearby feature cells within the island
            for _ in range(remaining_slots * 3):  # Try more times to find nearby programs
                # Perturb coordinates
                perturbed_coords = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-2, 2)))
                    for c in feature_coords
                ]

                cell_key = self._feature_coords_to_key(perturbed_coords)
                if cell_key in island_feature_map:
                    program_id = island_feature_map[cell_key]
                    if (
                        program_id != parent.id
                        and program_id not in [p.id for p in inspirations]
                        and program_id not in [p.id for p in nearby_programs]
                        and program_id in self.programs
                    ):
                        nearby_programs.append(self.programs[program_id])
                        if len(nearby_programs) >= remaining_slots:
                            break

            # If we still need more, add random programs from the island
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)

                # Get available programs from the island
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_island_ids = [
                    pid
                    for pid in island_program_ids
                    if pid not in excluded_ids and pid in self.programs
                ]

                if available_island_ids:
                    random_ids = random.sample(
                        available_island_ids, min(remaining, len(available_island_ids))
                    )
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        # Log island isolation info
        logger.debug(
            f"Sampled {len(inspirations)} inspirations from island {parent_island} "
            f"(island has {len(island_programs)} programs total)"
        )

        return inspirations[:n]

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:
        """
        Enforce the population size limit by removing worst programs if needed

        Args:
            exclude_program_id: Program ID to never remove (e.g., newly added program)
        """
        if len(self.programs) <= self.config.population_size:
            return

        # Calculate how many programs to remove
        num_to_remove = len(self.programs) - self.config.population_size

        logger.info(
            f"Population size ({len(self.programs)}) exceeds limit ({self.config.population_size}), removing {num_to_remove} programs"
        )

        # Get programs sorted by fitness (worst first)
        all_programs = list(self.programs.values())

        # Sort by average metric (worst first)
        sorted_programs = sorted(
            all_programs,
            key=lambda p: safe_numeric_average(p.metrics),
        )

        # Remove worst programs, but never remove the best program or excluded program
        programs_to_remove = []
        protected_ids = {self.best_program_id, exclude_program_id} - {None}

        for program in sorted_programs:
            if len(programs_to_remove) >= num_to_remove:
                break
            # Don't remove the best program or excluded program
            if program.id not in protected_ids:
                programs_to_remove.append(program)

        # If we still need to remove more and only have protected programs,
        # remove from the remaining programs anyway (but keep the protected ones)
        if len(programs_to_remove) < num_to_remove:
            remaining_programs = [
                p
                for p in sorted_programs
                if p not in programs_to_remove and p.id not in protected_ids
            ]
            additional_removals = remaining_programs[: num_to_remove - len(programs_to_remove)]
            programs_to_remove.extend(additional_removals)

        # Remove the selected programs
        for program in programs_to_remove:
            program_id = program.id

            # Remove from main programs dict
            if program_id in self.programs:
                del self.programs[program_id]

            # Remove from feature map
            keys_to_remove = []
            for key, pid in self.feature_map.items():
                if pid == program_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.feature_map[key]

            # Remove from islands
            for island in self.islands:
                island.discard(program_id)

            # Remove from archive
            self.archive.discard(program_id)

            logger.debug(f"Removed program {program_id} due to population limit")

        logger.info(f"Population size after cleanup: {len(self.programs)}")

        # Clean up any stale island best program references after removal
        self._cleanup_stale_island_bests()

    # Island management methods
    def set_current_island(self, island_idx: int) -> None:
        """Set which island is currently being evolved"""
        self.current_island = island_idx % len(self.islands)
        logger.debug(f"Switched to evolving island {self.current_island}")

    def next_island(self) -> int:
        """Move to the next island in round-robin fashion"""
        self.current_island = (self.current_island + 1) % len(self.islands)
        logger.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Increment generation counter for an island"""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1
        logger.debug(f"Island {idx} generation incremented to {self.island_generations[idx]}")

    def should_migrate(self) -> bool:
        """Check if migration should occur based on generation counters"""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval

    def migrate_programs(self) -> None:
        """
        Perform migration between islands

        This should be called periodically to share good solutions between islands
        """
        if len(self.islands) < 2:
            return

        logger.info("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue

            # Select top programs from this island for migration
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]
            if not island_programs:
                continue

            # Sort by fitness (using combined_score or average metrics)
            island_programs.sort(
                key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
                reverse=True,
            )

            # Select top programs for migration
            num_to_migrate = max(1, int(len(island_programs) * self.migration_rate))
            migrants = island_programs[:num_to_migrate]

            # Migrate to adjacent islands (ring topology)
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]

            for migrant in migrants:
                # Prevent re-migration of already migrated programs to avoid exponential duplication.
                # Analysis of actual evolution runs shows this causes severe issues:
                # - Program cb5d07f2 had 183 descendant copies by iteration 850
                # - Program 5645fbd2 had 31 descendant copies
                # - IDs grow exponentially: program_migrant_2_migrant_3_migrant_4_migrant_0...
                #
                # This is particularly problematic for OpenEvolve's MAP-Elites + Island hybrid architecture:
                # 1. All copies have identical code → same complexity/diversity/performance scores
                # 2. They all map to the SAME MAP-Elites cell → only 1 survives, rest discarded
                # 3. Wastes computation evaluating hundreds of identical programs
                # 4. Reduces actual diversity as islands fill with duplicates
                #
                # By preventing already-migrated programs from migrating again, we ensure:
                # - Each program migrates at most once per lineage
                # - True diversity is maintained between islands
                # - Computational resources aren't wasted on duplicates
                # - Aligns with MAP-Elites' one-program-per-cell principle
                if migrant.metadata.get("migrant", False):
                    continue

                for target_island in target_islands:
                    # Create a copy for migration (to avoid removing from source)
                    migrant_copy = Program(
                        id=f"{migrant.id}_migrant_{target_island}",
                        code=migrant.code,
                        language=migrant.language,
                        parent_id=migrant.id,
                        generation=migrant.generation,
                        metrics=migrant.metrics.copy(),
                        metadata={**migrant.metadata, "island": target_island, "migrant": True},
                    )

                    # Add to target island
                    self.islands[target_island].add(migrant_copy.id)
                    self.programs[migrant_copy.id] = migrant_copy

                    # Update island-specific best program if migrant is better
                    self._update_island_best_program(migrant_copy, target_island)

                    # Log migration with MAP-Elites coordinates
                    feature_coords = self._calculate_feature_coords(migrant_copy)
                    coords_dict = {
                        self.config.feature_dimensions[j]: feature_coords[j]
                        for j in range(len(feature_coords))
                    }
                    logger.info(
                        "Program migrated to island %d at MAP-Elites coords: %s",
                        target_island,
                        coords_dict,
                    )

        # Update last migration generation
        self.last_migration_generation = max(self.island_generations)
        logger.info(f"Migration completed at generation {self.last_migration_generation}")

        # Validate migration results
        self._validate_migration_results()

    def _validate_migration_results(self) -> None:
        """
        Validate migration didn't create inconsistencies

        Checks that:
        1. Program island metadata matches actual island assignment
        2. No programs are assigned to multiple islands
        3. All island best programs exist and are in correct islands
        """
        seen_program_ids = set()

        for i, island in enumerate(self.islands):
            for program_id in island:
                # Check for duplicate assignments
                if program_id in seen_program_ids:
                    logger.error(f"Program {program_id} assigned to multiple islands")
                    continue
                seen_program_ids.add(program_id)

                # Check program exists
                if program_id not in self.programs:
                    logger.warning(f"Island {i} contains nonexistent program {program_id}")
                    continue

                # Check metadata consistency
                program = self.programs[program_id]
                stored_island = program.metadata.get("island")
                if stored_island != i:
                    logger.warning(
                        f"Island mismatch for program {program_id}: "
                        f"in island {i} but metadata says {stored_island}"
                    )

        # Validate island best programs
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                if best_id not in self.programs:
                    logger.warning(f"Island {i} best program {best_id} does not exist")
                elif best_id not in self.islands[i]:
                    logger.warning(f"Island {i} best program {best_id} not in island")

    def _cleanup_stale_island_bests(self) -> None:
        """
        Remove stale island best program references

        Cleans up references to programs that no longer exist in the database
        or are not actually in their assigned islands.
        """
        cleaned_count = 0

        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                should_clear = False

                # Check if program still exists
                if best_id not in self.programs:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (program deleted)"
                    )
                    should_clear = True
                # Check if program is still in the island
                elif best_id not in self.islands[i]:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.island_best_programs[i] = None
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale island best program references")

            # Recalculate best programs for islands that were cleared
            for i, best_id in enumerate(self.island_best_programs):
                if best_id is None and len(self.islands[i]) > 0:
                    # Find new best program for this island
                    island_programs = [
                        self.programs[pid] for pid in self.islands[i] if pid in self.programs
                    ]
                    if island_programs:
                        # Sort by fitness and update
                        best_program = max(
                            island_programs,
                            key=lambda p: p.metrics.get(
                                "combined_score", safe_numeric_average(p.metrics)
                            ),
                        )
                        self.island_best_programs[i] = best_program.id
                        logger.debug(f"Recalculated island {i} best program: {best_program.id}")

    def get_island_stats(self) -> List[dict]:
        """Get statistics for each island"""
        stats = []

        for i, island in enumerate(self.islands):
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]

            if island_programs:
                scores = [
                    p.metrics.get("combined_score", safe_numeric_average(p.metrics))
                    for p in island_programs
                ]

                best_score = max(scores) if scores else 0.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_programs),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "diversity": diversity,
                    "generation": self.island_generations[i],
                    "is_current": i == self.current_island,
                }
            )

        return stats

    def _calculate_island_diversity(self, programs: List[Program]) -> float:
        """Calculate diversity within an island (deterministic version)"""
        if len(programs) < 2:
            return 0.0

        total_diversity = 0
        comparisons = 0

        # Use deterministic sampling instead of random.sample() to ensure consistent results
        sample_size = min(5, len(programs))  # Reduced from 10 to 5

        # Sort programs by ID for deterministic ordering
        sorted_programs = sorted(programs, key=lambda p: p.id)

        # Take first N programs instead of random sampling
        sample_programs = sorted_programs[:sample_size]

        # Limit total comparisons for performance
        max_comparisons = 6  # Maximum comparisons to prevent long delays

        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                if comparisons >= max_comparisons:
                    break

                # Use fast approximation instead of expensive edit distance
                diversity = self._fast_code_diversity(prog1.code, prog2.code)
                total_diversity += diversity
                comparisons += 1

            if comparisons >= max_comparisons:
                break

        return total_diversity / max(1, comparisons)

    def _fast_code_diversity(self, code1: str, code2: str) -> float:
        """
        Fast approximation of code diversity using simple metrics

        Returns diversity score (higher = more diverse)
        """
        if code1 == code2:
            return 0.0

        # Length difference (scaled to reasonable range)
        len1, len2 = len(code1), len(code2)
        length_diff = abs(len1 - len2)

        # Line count difference
        lines1 = code1.count("\n")
        lines2 = code2.count("\n")
        line_diff = abs(lines1 - lines2)

        # Simple character set difference
        chars1 = set(code1)
        chars2 = set(code2)
        char_diff = len(chars1.symmetric_difference(chars2))

        # Combine metrics (scaled to match original edit distance range)
        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5

        return diversity

    def _get_cached_diversity(self, program: Program) -> float:
        """
        Get diversity score for a program using cache and reference set

        Args:
            program: The program to calculate diversity for

        Returns:
            Diversity score (cached or newly computed)
        """
        code_hash = hash(program.code)

        # Check cache first
        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]

        # Update reference set if needed
        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()

        # Compute diversity against reference set
        diversity_scores = []
        for ref_code in self.diversity_reference_set:
            if ref_code != program.code:  # Don't compare with itself
                diversity_scores.append(self._fast_code_diversity(program.code, ref_code))

        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )

        # Cache the result with LRU eviction
        self._cache_diversity_value(code_hash, diversity)

        return diversity

    def _update_diversity_reference_set(self) -> None:
        """Update the reference set for diversity calculation"""
        if len(self.programs) == 0:
            return

        # Select diverse programs for reference set
        all_programs = list(self.programs.values())

        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.code for p in all_programs]
        else:
            # Select programs with maximum diversity
            selected = []
            remaining = all_programs.copy()

            # Start with a random program
            first_idx = random.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(first_idx))

            # Greedily add programs that maximize diversity to selected set
            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1
                best_idx = -1

                for i, candidate in enumerate(remaining):
                    # Calculate minimum diversity to selected programs
                    min_div = float("inf")
                    for selected_prog in selected:
                        div = self._fast_code_diversity(candidate.code, selected_prog.code)
                        min_div = min(min_div, div)

                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i

                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))

            self.diversity_reference_set = [p.code for p in selected]

        logger.debug(
            f"Updated diversity reference set with {len(self.diversity_reference_set)} programs"
        )

    def _cache_diversity_value(self, code_hash: int, diversity: float) -> None:
        """Cache a diversity value with LRU eviction"""
        # Check if cache is full
        if len(self.diversity_cache) >= self.diversity_cache_size:
            # Remove oldest entry
            oldest_hash = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_hash]

        # Add new entry
        self.diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}

    def _invalidate_diversity_cache(self) -> None:
        """Invalidate the diversity cache when programs change significantly"""
        self.diversity_cache.clear()
        self.diversity_reference_set = []
        logger.debug("Diversity cache invalidated")

    def _update_feature_stats(self, feature_name: str, value: float) -> None:
        """
        Update statistics for a feature dimension

        Args:
            feature_name: Name of the feature dimension
            value: New value to incorporate into stats
        """
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],  # Keep recent values for percentile calculation if needed
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

        # Keep recent values for more sophisticated scaling methods
        stats["values"].append(value)
        if len(stats["values"]) > 1000:  # Limit memory usage
            stats["values"] = stats["values"][-1000:]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        """
        Scale a feature value according to the configured scaling method

        Args:
            feature_name: Name of the feature dimension
            value: Raw feature value

        Returns:
            Scaled value in range [0, 1]
        """
        if feature_name not in self.feature_stats:
            # No stats yet, return normalized by a reasonable default
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]

        if self.feature_scaling_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = stats["min"]
            max_val = stats["max"]

            if max_val == min_val:
                return 0.5  # All values are the same

            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))  # Ensure in [0, 1]

        elif self.feature_scaling_method == "percentile":
            # Use percentile ranking
            values = stats["values"]
            if not values:
                return 0.5

            # Count how many values are less than or equal to this value
            count = sum(1 for v in values if v <= value)
            percentile = count / len(values)
            return percentile

        else:
            # Default to min-max if unknown method
            return self._scale_feature_value_minmax(feature_name, value)

    def _scale_feature_value_minmax(self, feature_name: str, value: float) -> float:
        """Helper for min-max scaling"""
        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]

        if max_val == min_val:
            return 0.5

        scaled = (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, scaled))

    def log_island_status(self) -> None:
        """Log current status of all islands"""
        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            island_idx = stat["island"]
            island_best_id = (
                self.island_best_programs[island_idx]
                if island_idx < len(self.island_best_programs)
                else None
            )
            best_indicator = f" (best: {island_best_id})" if island_best_id else ""
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}{best_indicator}"
            )

    # Artifact storage and retrieval methods

    def store_artifacts(self, program_id: str, artifacts: Dict[str, Union[str, bytes]]) -> None:
        """
        Store artifacts for a program

        Args:
            program_id: ID of the program
            artifacts: Dictionary of artifact name to content
        """
        if not artifacts:
            return

        program = self.get(program_id)
        if not program:
            logger.warning(f"Cannot store artifacts: program {program_id} not found")
            return

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"
        if not artifacts_enabled:
            logger.debug("Artifacts disabled, skipping storage")
            return

        # Split artifacts by size
        small_artifacts = {}
        large_artifacts = {}
        size_threshold = getattr(self.config, "artifact_size_threshold", 32 * 1024)  # 32KB default

        for key, value in artifacts.items():
            size = self._get_artifact_size(value)
            if size <= size_threshold:
                small_artifacts[key] = value
            else:
                large_artifacts[key] = value

        # Store small artifacts as JSON
        if small_artifacts:
            program.artifacts_json = json.dumps(small_artifacts, default=self._artifact_serializer)
            logger.debug(f"Stored {len(small_artifacts)} small artifacts for program {program_id}")

        # Store large artifacts to disk
        if large_artifacts:
            artifact_dir = self._create_artifact_dir(program_id)
            program.artifact_dir = artifact_dir
            for key, value in large_artifacts.items():
                self._write_artifact_file(artifact_dir, key, value)
            logger.debug(f"Stored {len(large_artifacts)} large artifacts for program {program_id}")

    def get_artifacts(self, program_id: str) -> Dict[str, Union[str, bytes]]:
        """
        Retrieve all artifacts for a program

        Args:
            program_id: ID of the program

        Returns:
            Dictionary of artifact name to content
        """
        program = self.get(program_id)
        if not program:
            return {}

        artifacts = {}

        # Load small artifacts from JSON
        if program.artifacts_json:
            try:
                small_artifacts = json.loads(program.artifacts_json)
                artifacts.update(small_artifacts)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode artifacts JSON for program {program_id}: {e}")

        # Load large artifacts from disk
        if program.artifact_dir and os.path.exists(program.artifact_dir):
            disk_artifacts = self._load_artifact_dir(program.artifact_dir)
            artifacts.update(disk_artifacts)

        return artifacts

    def _get_artifact_size(self, value: Union[str, bytes]) -> int:
        """Get size of an artifact value in bytes"""
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def _artifact_serializer(self, obj):
        """JSON serializer for artifacts that handles bytes"""
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("utf-8")}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _artifact_deserializer(self, dct):
        """JSON deserializer for artifacts that handles bytes"""
        if "__bytes__" in dct:
            return base64.b64decode(dct["__bytes__"])
        return dct

    def _create_artifact_dir(self, program_id: str) -> str:
        """Create artifact directory for a program"""
        base_path = getattr(self.config, "artifacts_base_path", None)
        if not base_path:
            base_path = (
                os.path.join(self.config.db_path or ".", "artifacts")
                if self.config.db_path
                else "./artifacts"
            )

        artifact_dir = os.path.join(base_path, program_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _cleanup_old_artifacts(self, checkpoint_path: str) -> None:
        """
        Remove artifact directories older than the configured retention period.

        Args:
            checkpoint_path: The path of the current checkpoint being saved, which
                             contains the artifacts folder to be cleaned.
        """
        if not self.config.cleanup_old_artifacts:
            return

        artifacts_base_path = os.path.join(checkpoint_path, "artifacts")

        if not os.path.isdir(artifacts_base_path):
            return

        now = time.time()
        retention_seconds = self.config.artifact_retention_days * 24 * 60 * 60
        deleted_count = 0

        logger.debug(f"Starting artifact cleanup in {artifacts_base_path}...")

        for dirname in os.listdir(artifacts_base_path):
            dirpath = os.path.join(artifacts_base_path, dirname)
            if os.path.isdir(dirpath):
                try:
                    dir_mod_time = os.path.getmtime(dirpath)
                    if (now - dir_mod_time) > retention_seconds:
                        shutil.rmtree(dirpath)
                        deleted_count += 1
                        logger.debug(f"Removed old artifact directory: {dirpath}")
                except FileNotFoundError:
                    # Can happen in race conditions; ignore.
                    continue
                except Exception as e:
                    logger.error(f"Error removing artifact directory {dirpath}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old artifact directories.")

    def _write_artifact_file(self, artifact_dir: str, key: str, value: Union[str, bytes]) -> None:
        """Write an artifact to a file"""
        # Sanitize filename
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"

        file_path = os.path.join(artifact_dir, safe_key)

        try:
            if isinstance(value, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(value)
            elif isinstance(value, bytes):
                with open(file_path, "wb") as f:
                    f.write(value)
            else:
                # Convert to string and write
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(value))
        except Exception as e:
            logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")

    def _load_artifact_dir(self, artifact_dir: str) -> Dict[str, Union[str, bytes]]:
        """Load artifacts from a directory"""
        artifacts = {}

        try:
            for filename in os.listdir(artifact_dir):
                file_path = os.path.join(artifact_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        # Try to read as text first
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except UnicodeDecodeError:
                        # If text fails, read as binary
                        with open(file_path, "rb") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except Exception as e:
                        logger.warning(f"Failed to read artifact file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list artifact directory {artifact_dir}: {e}")

        return artifacts

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:
        """
        Log a prompt for a program.
        Only logs if self.config.log_prompts is True.

        Args:
        program_id: ID of the program to log the prompt for
        template_key: Key for the prompt template
        prompt: Prompts in the format {template_key: { 'system': str, 'user': str }}.
        responses: Optional list of responses to the prompt, if available.
        """

        if not self.config.log_prompts:
            return

        if responses is None:
            responses = []
        prompt["responses"] = responses

        if self.prompts_by_program is None:
            self.prompts_by_program = {}

        if program_id not in self.prompts_by_program:
            self.prompts_by_program[program_id] = {}
        self.prompts_by_program[program_id][template_key] = prompt
