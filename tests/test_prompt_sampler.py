"""
Tests for PromptSampler in openevolve.prompt.sampler
"""

import unittest
from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler


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
