"""
Basic tests for OpenEvolve components
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import apply_diff, extract_diffs


class TestCodeUtils(unittest.TestCase):
    """Tests for code utilities"""

    def test_extract_diffs(self):
        """Test extracting diffs from a response"""
        diff_text = """
        Let's improve this code:

        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        Another change:

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        diffs = extract_diffs(diff_text)
        self.assertEqual(len(diffs), 2)
        self.assertEqual(
            diffs[0][0],
            """        def hello():
            print("Hello")""",
        )
        self.assertEqual(
            diffs[0][1],
            """        def hello():
            print("Hello, World!")""",
        )
        self.assertEqual(diffs[1][0], "        x = 1")
        self.assertEqual(diffs[1][1], "        x = 2")

    def test_apply_diff(self):
        """Test applying diffs to code"""
        original_code = """
        def hello():
            print("Hello")

        x = 1
        y = 2
        """

        diff_text = """
        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        expected_code = """
        def hello():
            print("Hello, World!")

        x = 2
        y = 2
        """

        result = apply_diff(original_code, diff_text)

        # Normalize whitespace for comparison
        self.assertEqual(
            result,
            expected_code,
        )


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


class TestPromptSampler(unittest.TestCase):
    """Tests for prompt sampler"""

    def setUp(self):
        """Set up test prompt sampler"""
        config = Config()
        self.prompt_sampler = PromptSampler(config.prompt)

    def test_build_prompt(self):
        """Test building a prompt"""
        current_program = "def test(): pass"
        parent_program = "def test(): pass"
        program_metrics = {"score": 0.5}
        previous_programs = [
            {
                "id": "prev1",
                "code": "def prev1(): pass",
                "metrics": {"score": 0.4},
            }
        ]
        top_programs = [
            {
                "id": "top1",
                "code": "def top1(): pass",
                "metrics": {"score": 0.6},
            }
        ]

        prompt = self.prompt_sampler.build_prompt(
            current_program=current_program,
            parent_program=parent_program,
            program_metrics=program_metrics,
            previous_programs=previous_programs,
            top_programs=top_programs,
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("def test(): pass", prompt["user"])
        self.assertIn("score: 0.5", prompt["user"])


if __name__ == "__main__":
    unittest.main()
