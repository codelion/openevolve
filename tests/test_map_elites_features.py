"""
Tests for MAP-Elites feature enhancements in openevolve.database
"""

import unittest
from unittest.mock import MagicMock, patch
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestMapElitesFeatures(unittest.TestCase):
    """Tests for MAP-Elites feature enhancements"""

    def setUp(self):
        """Set up test database with enhanced features"""
        config = Config()
        config.database.in_memory = True
        config.database.feature_dimensions = ["complexity", "diversity"]
        config.database.feature_bins = 10
        self.db = ProgramDatabase(config.database)

    def test_diversity_caching(self):
        """Test diversity score caching functionality"""
        # Add some reference programs
        for i in range(5):
            program = Program(
                id=f"ref_{i}",
                code=f"def func_{i}():\n    return {i} * 2",
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            self.db.add(program)

        # Create a test program
        test_program = Program(
            id="test", code="def test():\n    return 42", language="python", metrics={"score": 0.7}
        )

        # First call should compute diversity
        diversity1 = self.db._get_cached_diversity(test_program)

        # Second call should use cache
        diversity2 = self.db._get_cached_diversity(test_program)

        # Should be the same value
        self.assertEqual(diversity1, diversity2)

        # Check cache was populated
        code_hash = hash(test_program.code)
        self.assertIn(code_hash, self.db.diversity_cache)
        self.assertEqual(self.db.diversity_cache[code_hash]["value"], diversity1)

    def test_diversity_reference_set_update(self):
        """Test diversity reference set updates"""
        # Initially empty
        self.assertEqual(len(self.db.diversity_reference_set), 0)

        # Add programs
        for i in range(30):
            program = Program(
                id=f"prog_{i}",
                code=f"def func_{i}():\n    " + "x = 1\n" * i,  # Varying complexity
                language="python",
                metrics={"score": 0.5},
            )
            self.db.add(program)

        # Update reference set
        self.db._update_diversity_reference_set()

        # Should have up to diversity_reference_size programs
        expected_size = min(30, self.db.diversity_reference_size)
        self.assertEqual(len(self.db.diversity_reference_set), expected_size)

        # Reference set should contain diverse programs (no duplicates)
        self.assertEqual(len(set(self.db.diversity_reference_set)), expected_size)

    def test_feature_scaling_minmax(self):
        """Test min-max feature scaling"""
        # Add programs with different complexities
        complexities = [100, 300, 500, 700, 900]
        for i, complexity in enumerate(complexities):
            program = Program(
                id=f"scale_{i}", code="x" * complexity, language="python", metrics={"score": 0.5}
            )
            self.db.add(program)

        # Test scaling
        # Min value (100) should scale to 0
        scaled_min = self.db._scale_feature_value("complexity", 100.0)
        self.assertAlmostEqual(scaled_min, 0.0, places=5)

        # Max value (900) should scale to 1
        scaled_max = self.db._scale_feature_value("complexity", 900.0)
        self.assertAlmostEqual(scaled_max, 1.0, places=5)

        # Middle value (500) should scale to 0.5
        scaled_mid = self.db._scale_feature_value("complexity", 500.0)
        self.assertAlmostEqual(scaled_mid, 0.5, places=5)

    def test_feature_stats_update(self):
        """Test feature statistics are updated correctly"""
        # Initially no stats
        self.assertEqual(len(self.db.feature_stats), 0)

        # Add programs and check stats are updated
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for val in values:
            self.db._update_feature_stats("test_feature", val)

        # Check stats
        stats = self.db.feature_stats["test_feature"]
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)
        self.assertEqual(len(stats["values"]), 5)
        self.assertEqual(stats["values"], values)

    def test_per_dimension_bins(self):
        """Test per-dimension bin configuration"""
        # Create database with per-dimension bins
        config = Config()
        config.database.in_memory = True
        config.database.feature_dimensions = ["complexity", "diversity", "score"]
        config.database.feature_bins = {"complexity": 20, "diversity": 10, "score": 5}
        db = ProgramDatabase(config.database)

        # Check per-dimension bins were set correctly
        self.assertEqual(db.feature_bins_per_dim["complexity"], 20)
        self.assertEqual(db.feature_bins_per_dim["diversity"], 10)
        self.assertEqual(db.feature_bins_per_dim["score"], 5)

        # Add a program and check binning
        program = Program(
            id="test_bins",
            code="def test():\n    return 42",
            language="python",
            metrics={"score": 0.8},
        )
        db.add(program)

        coords = db._calculate_feature_coords(program)

        # Each coordinate should be within its dimension's range
        self.assertLess(coords[0], 20)  # complexity
        self.assertLess(coords[1], 10)  # diversity
        self.assertLess(coords[2], 5)  # score

    def test_default_feature_dimensions(self):
        """Test default feature dimensions are complexity and diversity"""
        config = Config()
        # Don't set feature_dimensions, use defaults
        self.assertEqual(config.database.feature_dimensions, ["complexity", "diversity"])

    def test_diversity_cache_lru_eviction(self):
        """Test LRU eviction in diversity cache"""
        # Set small cache size
        self.db.diversity_cache_size = 3

        # Add reference programs
        for i in range(3):
            program = Program(
                id=f"ref_{i}",
                code=f"def func_{i}(): pass",
                language="python",
                metrics={"score": 0.5},
            )
            self.db.add(program)

        # Fill cache
        programs = []
        for i in range(5):
            program = Program(
                id=f"cache_test_{i}",
                code=f"def test_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5},
            )
            programs.append(program)
            self.db._get_cached_diversity(program)

        # Cache should only have last 3 entries
        self.assertLessEqual(len(self.db.diversity_cache), 3)

        # First 2 programs should be evicted
        self.assertNotIn(hash(programs[0].code), self.db.diversity_cache)
        self.assertNotIn(hash(programs[1].code), self.db.diversity_cache)

        # Last 3 should be in cache
        self.assertIn(hash(programs[2].code), self.db.diversity_cache)
        self.assertIn(hash(programs[3].code), self.db.diversity_cache)
        self.assertIn(hash(programs[4].code), self.db.diversity_cache)

    def test_feature_scaling_with_identical_values(self):
        """Test feature scaling when all values are identical"""
        # Add programs with same complexity
        for i in range(3):
            program = Program(
                id=f"same_{i}",
                code="x" * 100,  # Same length
                language="python",
                metrics={"score": 0.5},
            )
            self.db.add(program)

        # Scaling should return 0.5 for all values when min==max
        scaled = self.db._scale_feature_value("complexity", 100.0)
        self.assertEqual(scaled, 0.5)

    def test_feature_coordinates_with_new_defaults(self):
        """Test feature coordinate calculation with new default dimensions"""
        # Create fresh database with default config
        config = Config()
        config.database.in_memory = True
        db = ProgramDatabase(config.database)

        # Default dimensions should be complexity and diversity
        self.assertEqual(db.config.feature_dimensions, ["complexity", "diversity"])

        # Add some programs
        for i in range(5):
            program = Program(
                id=f"default_test_{i}",
                code=f"def func_{i}():\n    " + "pass\n" * i,
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            db.add(program)

        # Test program
        test_program = Program(
            id="test_defaults",
            code="def test():\n    return 42",
            language="python",
            metrics={"score": 0.7},
        )

        coords = db._calculate_feature_coords(test_program)

        # Should have 2 coordinates (complexity, diversity)
        self.assertEqual(len(coords), 2)

        # Both should be valid bin indices
        for coord in coords:
            self.assertGreaterEqual(coord, 0)
            self.assertLess(coord, db.feature_bins)

    def test_missing_feature_dimension_error(self):
        """Test that missing feature dimensions raise appropriate errors"""
        config = Config()
        config.database.in_memory = True
        config.database.feature_dimensions = ["complexity", "nonexistent_metric"]
        db = ProgramDatabase(config.database)

        # Add a program without the required metric
        program = Program(
            id="test_error",
            code="def test(): pass",
            language="python",
            metrics={"score": 0.5},  # Missing 'nonexistent_metric'
        )

        # Should raise ValueError when calculating feature coordinates
        with self.assertRaises(ValueError) as context:
            db.add(program)

        # Check error message
        self.assertIn("nonexistent_metric", str(context.exception))
        self.assertIn("not found in program metrics", str(context.exception))
        self.assertIn("score", str(context.exception))  # Should show available metrics


if __name__ == "__main__":
    unittest.main()
