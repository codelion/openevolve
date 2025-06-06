"""
Tests for LLMEnsemble in openevolve.llm.ensemble
"""

import unittest
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig


class TestLLMEnsemble(unittest.TestCase):
    def test_weighted_sampling(self):
        models = [
            LLMModelConfig(name="a", weight=0.0),
            LLMModelConfig(name="b", weight=1.0),
        ]
        ensemble = LLMEnsemble(models)
        # Should always sample model 'b'
        for _ in range(10):
            self.assertEqual(ensemble._sample_model().model, "b")

        models = [
            LLMModelConfig(name="a", weight=0.3),
            LLMModelConfig(name="b", weight=0.3),
            LLMModelConfig(name="c", weight=0.3),
        ]
        ensemble = LLMEnsemble(models)
        # Should sample both models. Track sampled models in a set
        sampled_models = set()
        for _ in range(1000):
            sampled_models.add(ensemble._sample_model().model)
            # Cancel once we have both models
            if len(sampled_models) == len(models):
                break
        self.assertEqual(len(sampled_models), len(models))


if __name__ == "__main__":
    unittest.main()
