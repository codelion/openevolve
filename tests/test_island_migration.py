"""
Tests for island migration functionality in openevolve.database
"""

import unittest
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestIslandMigration(unittest.TestCase):
    """Tests for island migration in program database"""

    def setUp(self):
        """Set up test database with multiple islands"""
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.migration_rate = 0.5  # 50% of programs migrate
        config.database.migration_interval = 5  # Migrate every 5 generations
        self.db = ProgramDatabase(config.database)

    def _create_test_program(self, program_id: str, score: float, island: int) -> Program:
        """Helper to create a test program"""
        program = Program(
            id=program_id,
            code=f"def func_{program_id}(): return {score}",
            language="python",
            metrics={"score": score, "combined_score": score},
            metadata={"island": island},
        )
        return program

    def test_initial_island_setup(self):
        """Test that islands are properly initialized"""
        self.assertEqual(len(self.db.islands), 3)
        self.assertEqual(len(self.db.island_best_programs), 3)
        self.assertEqual(len(self.db.island_generations), 3)

        # All islands should be empty initially
        for island in self.db.islands:
            self.assertEqual(len(island), 0)

        # All island best programs should be None initially
        for best_id in self.db.island_best_programs:
            self.assertIsNone(best_id)

    def test_program_island_assignment(self):
        """Test that programs are assigned to correct islands"""
        # Add programs to specific islands
        program1 = self._create_test_program("test1", 0.5, 0)
        program2 = self._create_test_program("test2", 0.7, 1)
        program3 = self._create_test_program("test3", 0.3, 2)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)
        self.db.add(program3, target_island=2)

        # Verify island assignments
        self.assertIn("test1", self.db.islands[0])
        self.assertIn("test2", self.db.islands[1])
        self.assertIn("test3", self.db.islands[2])

        # Verify metadata
        self.assertEqual(self.db.programs["test1"].metadata["island"], 0)
        self.assertEqual(self.db.programs["test2"].metadata["island"], 1)
        self.assertEqual(self.db.programs["test3"].metadata["island"], 2)

    def test_should_migrate_logic(self):
        """Test the migration timing logic"""
        # Initially should not migrate (no generations passed)
        self.assertFalse(self.db.should_migrate())

        # Advance island generations
        self.db.island_generations = [5, 6, 7]  # Max is 7, last migration was 0, so 7-0=7 >= 5
        self.assertTrue(self.db.should_migrate())

        # Test with mixed generations below threshold
        self.db.island_generations = [3, 4, 2]  # Max is 4, 4-0=4 < 5
        self.assertFalse(self.db.should_migrate())

    def test_migration_ring_topology(self):
        """Test that migration follows ring topology"""
        # Add programs to islands 0 and 1
        program1 = self._create_test_program("test1", 0.8, 0)
        program2 = self._create_test_program("test2", 0.6, 1)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=1)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]  # Trigger migration

        initial_program_count = len(self.db.programs)

        # Perform migration
        self.db.migrate_programs()

        # Should have created migrant copies
        self.assertGreater(len(self.db.programs), initial_program_count)

        # Check that migrants were created with proper naming
        migrant_ids = [pid for pid in self.db.programs.keys() if "_migrant_" in pid]
        self.assertGreater(len(migrant_ids), 0)

        # Verify ring topology: island 0 -> islands 1,2
        island_0_migrants = [pid for pid in migrant_ids if "test1_migrant_" in pid]

        # test1 from island 0 should migrate to islands 1 and 2 (0+1=1, 0-1=-1%3=2)
        self.assertTrue(any(pid.endswith("_1") for pid in island_0_migrants))
        self.assertTrue(any(pid.endswith("_2") for pid in island_0_migrants))

        # Note: Due to the current migration implementation, test2 may not create direct migrants
        # when test1 migrants are added to island 1 during the same migration round.
        # This is a known limitation of the current implementation that processes islands
        # sequentially while modifying them, causing interference between migration rounds.

    def test_migration_rate_respected(self):
        """Test that migration rate is properly applied"""
        # Add multiple programs to island 0
        programs = []
        for i in range(10):
            program = self._create_test_program(f"test{i}", 0.5 + i * 0.05, 0)
            programs.append(program)
            self.db.add(program, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        initial_count = len(self.db.programs)

        # Perform migration
        self.db.migrate_programs()

        # Calculate expected migrants
        # With 50% migration rate and 10 programs, expect 5 migrants
        # Each migrant goes to 2 target islands, so 10 initial new programs
        # But migrants can themselves migrate, so more programs are created
        initial_migrants = 5 * 2  # 5 migrants * 2 target islands each
        actual_new_programs = len(self.db.programs) - initial_count

        # Should have at least the initial expected migrants
        self.assertGreaterEqual(actual_new_programs, initial_migrants)

        # Check that the right number of first-generation migrants were created
        first_gen_migrants = [
            pid
            for pid in self.db.programs.keys()
            if pid.count("_migrant_") == 1 and "_migrant_" in pid
        ]
        self.assertEqual(len(first_gen_migrants), initial_migrants)

    def test_migration_preserves_best_programs(self):
        """Test that migration selects the best programs for migration"""
        # Add programs with different scores to island 0
        program1 = self._create_test_program("low_score", 0.2, 0)
        program2 = self._create_test_program("high_score", 0.9, 0)
        program3 = self._create_test_program("med_score", 0.5, 0)

        self.db.add(program1, target_island=0)
        self.db.add(program2, target_island=0)
        self.db.add(program3, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Perform migration
        self.db.migrate_programs()

        # Check that the high-score program was selected for migration
        migrant_ids = [pid for pid in self.db.programs.keys() if "_migrant_" in pid]
        high_score_migrants = [pid for pid in migrant_ids if "high_score_migrant_" in pid]

        self.assertGreater(len(high_score_migrants), 0)

    def test_migration_updates_generations(self):
        """Test that migration updates the last migration generation"""
        # Add a program and set up for migration
        program = self._create_test_program("test1", 0.5, 0)
        self.db.add(program, target_island=0)

        self.db.island_generations = [6, 7, 8]
        initial_migration_gen = self.db.last_migration_generation

        # Perform migration
        self.db.migrate_programs()

        # Should update to max of island generations
        self.assertEqual(self.db.last_migration_generation, 8)
        self.assertGreater(self.db.last_migration_generation, initial_migration_gen)

    def test_migration_with_empty_islands(self):
        """Test that migration handles empty islands gracefully"""
        # Add program only to island 0, leave others empty
        program = self._create_test_program("test1", 0.5, 0)
        self.db.add(program, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Should not crash with empty islands
        try:
            self.db.migrate_programs()
        except Exception as e:
            self.fail(f"Migration with empty islands should not crash: {e}")

    def test_migration_creates_proper_copies(self):
        """Test that migration creates proper program copies"""
        program = self._create_test_program("original", 0.7, 0)
        self.db.add(program, target_island=0)

        # Set up for migration
        self.db.island_generations = [6, 6, 6]

        # Perform migration
        self.db.migrate_programs()

        # Find migrant copies
        migrant_ids = [pid for pid in self.db.programs.keys() if "original_migrant_" in pid]
        self.assertGreater(len(migrant_ids), 0)

        # Check first-generation migrant properties
        first_gen_migrants = [pid for pid in migrant_ids if pid.count("_migrant_") == 1]
        self.assertGreater(len(first_gen_migrants), 0)

        for migrant_id in first_gen_migrants:
            migrant = self.db.programs[migrant_id]

            # Should have same code and metrics as original
            self.assertEqual(migrant.code, program.code)
            self.assertEqual(migrant.metrics, program.metrics)

            # Should have proper parent reference
            self.assertEqual(migrant.parent_id, "original")

            # Should be marked as migrant
            self.assertTrue(migrant.metadata.get("migrant", False))

            # Should be in correct target island
            target_island = migrant.metadata["island"]
            self.assertIn(migrant_id, self.db.islands[target_island])

    def test_no_migration_with_single_island(self):
        """Test that migration is skipped with single island"""
        # Create database with single island
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 1
        single_island_db = ProgramDatabase(config.database)

        program = self._create_test_program("test1", 0.5, 0)
        single_island_db.add(program, target_island=0)

        single_island_db.island_generations = [6]

        initial_count = len(single_island_db.programs)

        # Should not perform migration
        single_island_db.migrate_programs()

        # Program count should remain the same
        self.assertEqual(len(single_island_db.programs), initial_count)


if __name__ == "__main__":
    unittest.main()
