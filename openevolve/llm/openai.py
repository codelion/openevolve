"""
OpenAI API interface for LLMs
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

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
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.api_type = getattr(model_cfg, "api_type", "auto")

        # Set up API client
        # OpenAI client requires max_retries to be int, not None
        max_retries = self.retries if self.retries is not None else 0
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=max_retries,
        )

        # Determine which API to use (Responses API vs Chat Completions)
        self.use_responses_api = self._should_use_responses_api()

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    def _should_use_responses_api(self) -> bool:
        """
        Determine if the Responses API should be used instead of Chat Completions.
        
        The Responses API is only available on OpenAI's official endpoints.
        For other providers (OpenRouter, Google AI Studio, local servers, etc.),
        we must use the Chat Completions API for compatibility.
        
        Returns:
            True if Responses API should be used, False for Chat Completions
        """
        # Normalize api_type (None defaults to "auto")
        api_type = self.api_type if self.api_type is not None else "auto"
        
        # Check for explicit override
        if api_type == "responses":
            return True
        if api_type == "chat_completions":
            return False
        
        # Auto-detect based on API base URL
        if not self.api_base:
            return False
        
        api_lower = self.api_base.lower()
        
        # Only use Responses API for official OpenAI endpoints
        return (
            api_lower.startswith("https://api.openai.com") or
            api_lower.startswith("https://eu.api.openai.com") or
            api_lower.startswith("https://apac.api.openai.com")
        )

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
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model based on model name pattern
        # This works for all endpoints (OpenAI, Azure, OptiLLM, OpenRouter, etc.)
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES) 

        if is_openai_reasoning_model:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

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
        """Make the actual API call, dispatching to appropriate API"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()

        if self.use_responses_api:
            response = await loop.run_in_executor(
                None, lambda: self._call_responses_api(params)
            )
            response_text = response.output_text
        else:
            response = await loop.run_in_executor(
                None, lambda: self.client.chat.completions.create(**params)
            )
            response_text = response.choices[0].message.content

        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response_text}")
        return response_text

    def _call_responses_api(self, chat_params: Dict[str, Any]) -> Any:
        """
        Convert Chat Completions params to Responses API format and make the call.
        
        The Responses API uses a different parameter structure:
        - 'messages' -> 'input' (can be array of messages)
        - System message in 'messages' -> 'instructions' parameter
        - 'max_tokens'/'max_completion_tokens' -> 'max_output_tokens'
        - 'reasoning_effort' -> 'reasoning: {"effort": ...}'
        
        Args:
            chat_params: Parameters in Chat Completions format
            
        Returns:
            Response object from client.responses.create()
        """
        messages = chat_params["messages"]

        # Extract system message as instructions, keep other messages as input
        instructions = None
        input_messages = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                input_messages.append(msg)

        # Build Responses API params
        resp_params = {
            "model": chat_params["model"],
            "input": input_messages,
        }

        if instructions:
            resp_params["instructions"] = instructions

        # Map token limits (Responses API uses max_output_tokens)
        if "max_completion_tokens" in chat_params:
            resp_params["max_output_tokens"] = chat_params["max_completion_tokens"]
        elif "max_tokens" in chat_params:
            resp_params["max_output_tokens"] = chat_params["max_tokens"]

        # Map sampling parameters
        if "temperature" in chat_params:
            resp_params["temperature"] = chat_params["temperature"]
        if "top_p" in chat_params:
            resp_params["top_p"] = chat_params["top_p"]
        if "seed" in chat_params:
            resp_params["seed"] = chat_params["seed"]

        # Map reasoning_effort to nested format for Responses API
        if "reasoning_effort" in chat_params:
            resp_params["reasoning"] = {"effort": chat_params["reasoning_effort"]}

        # Disable conversation storage (not needed for OpenEvolve's use case)
        resp_params["store"] = False

        return self.client.responses.create(**resp_params)
