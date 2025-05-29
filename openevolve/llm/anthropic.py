"""
Anthropic Claude API interface for LLMs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """LLM interface using Anthropic's Claude API"""

    def __init__(
        self,
        config: LLMConfig,
        model: Optional[str] = None,
    ):
        self.config = config
        self.model = model or config.primary_model

        # Set up API client
        self.client = anthropic.Anthropic(
            api_key=config.api_key,
            base_url=config.api_base,
        )

        logger.info(f"Initialized Anthropic LLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.config.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages for Claude format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        # Set up generation parameters
        params = {
            "model": self.model,
            "system": system_message,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.config.retries)
        retry_delay = kwargs.get("retry_delay", self.config.retry_delay)
        timeout = kwargs.get("timeout", self.config.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.client.messages.create(**params))

        # Extract the response content
        return response.content[0].text
