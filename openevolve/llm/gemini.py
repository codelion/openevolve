"""
Gemini API interface for LLMs
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class GeminiLLM(LLMInterface):
    """LLM interface using Google Gemini API"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = model_cfg.api_key

        # Set up Gemini client
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Initialized Gemini LLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Format the conversation for Gemini
        # Gemini doesn't have explicit system messages, so we prepend it to the first user message
        formatted_content = []
        
        # Add system message as the first part of the conversation
        if system_message:
            formatted_content.append(f"System: {system_message}\n\n")
        
        # Add conversation history
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_content.append(f"User: {content}")
            elif role == "assistant":
                formatted_content.append(f"Assistant: {content}")
            else:
                # Handle any other roles as user messages
                formatted_content.append(f"{role}: {content}")
        
        # Join all parts into a single prompt
        full_prompt = "\n\n".join(formatted_content)

        # Set up generation parameters
        generation_config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._call_api(full_prompt, generation_config), 
                    timeout=timeout
                )
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

    async def _call_api(self, prompt: str, generation_config: types.GenerateContentConfig) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=generation_config
            )
        )
        return response.text