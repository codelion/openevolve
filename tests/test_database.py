"""
Tests for ProgramDatabase in openevolve.database
"""

import unittest
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestProgramDatabase(unittest.TestCase):
    """Tests for program database"""

    def setUp(self):
        """Set up test database"""
        config = Config()
        config.database.in_memory = True
        self.db = ProgramDatabase(config.database)

    def test_add_and_get(self):
        """Test adding and retrieving a program"""
        program = Program(
            id="test1",
            code="def test(): pass",
            language="python",
            metrics={"score": 0.5},
        )

        self.db.add(program)

        retrieved = self.db.get("test1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "test1")
        self.assertEqual(retrieved.code, "def test(): pass")
        self.assertEqual(retrieved.metrics["score"], 0.5)

    def test_get_best_program(self):
        """Test getting the best program"""
        program1 = Program(
            id="test1",
            code="def test1(): pass",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="test2",
            code="def test2(): pass",
            language="python",
            metrics={"score": 0.7},
        )

        self.db.add(program1)
        self.db.add(program2)

        best = self.db.get_best_program()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "test2")

    def test_sample(self):
        """Test sampling from the database"""
        program1 = Program(
            id="test1",
            code="def test1(): pass",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="test2",
            code="def test2(): pass",
            language="python",
            metrics={"score": 0.7},
        )

        self.db.add(program1)
        self.db.add(program2)

        parent, inspirations = self.db.sample()

        self.assertIsNotNone(parent)
        self.assertIn(parent.id, ["test1", "test2"])


if __name__ == "__main__":
    unittest.main()
