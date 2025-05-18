"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs for generating diverse code modifications"""

    def __init__(self, config: LLMConfig):
        self.config = config

        # Initialize primary and secondary models
        self.primary_model = OpenAILLM(config, model=config.primary_model)
        self.secondary_model = OpenAILLM(config, model=config.secondary_model)

        # Model weights for sampling
        self._weights = [
            config.primary_model_weight,
            config.secondary_model_weight,
        ]

        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

        logger.info(
            f"Initialized LLM ensemble with models: "
            f"{config.primary_model} (weight: {self._weights[0]:.2f}), "
            f"{config.secondary_model} (weight: {self._weights[1]:.2f})"
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        models = [self.primary_model, self.secondary_model]
        index = random.choices(range(len(models)), weights=self._weights, k=1)[0]
        return models[index]

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
