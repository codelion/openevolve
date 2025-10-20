"""
Adapted from SakanaAI/ShinkaEvolve (Apache-2.0 License)
Original source: https://github.com/SakanaAI/ShinkaEvolve/blob/main/shinka/llm/embedding.py
"""

import os
import logging
from typing import List, Optional, Union

import openai

logger = logging.getLogger(__name__)

M = 1_000_000

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

AZURE_EMBEDDING_MODELS = [
    "azure-text-embedding-3-small",
    "azure-text-embedding-3-large",
]

OPENAI_EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.02 / M,
    "text-embedding-3-large": 0.13 / M,
}

class EmbeddingClient:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the EmbeddingClient.

        Args:
            model (str): The OpenAI embedding model name to use.
        """
        self.client, self.model = self._get_client_model(model_name)
        self._disabled_reason: Optional[str] = None
    
    def _get_client_model(self, model_name: str) -> tuple[openai.OpenAI, str]:
        if model_name in OPENAI_EMBEDDING_MODELS:
            client = openai.OpenAI()
            model_to_use = model_name
        elif model_name in AZURE_EMBEDDING_MODELS:
            # get rid of the azure- prefix
            model_to_use = model_name.split("azure-")[-1]
            client = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            )
        else:
            raise ValueError(f"Invalid embedding model: {model_name}")

        return client, model_to_use

    def get_embedding(
        self, code: Union[str, List[str]]
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Computes the text embedding for a code string.

        Args:
            code (str, list[str]): The code as a string or list
                of strings.

        Returns:
            list: Embedding vector for the code or None if an error
                occurs.
        """
        if self._disabled_reason:
            logger.debug(
                "Embedding client disabled; reusing cached reason: %s",
                self._disabled_reason,
            )
            return None

        if isinstance(code, str):
            code = [code]
            single_code = True
        else:
            single_code = False
        try:
            response = self.client.embeddings.create(
                model=self.model, input=code, encoding_format="float"
            )
            # Extract embedding from response
            if single_code:
                return response.data[0].embedding
            else:
                return [d.embedding for d in response.data]
        except Exception as e:  # pragma: no cover - depends on external API
            error_message = str(e)
            logger.error("Embedding request failed: %s", error_message)

            if "unsupported_country_region_territory" in error_message:
                self._disabled_reason = (
                    "OpenAI embeddings endpoint is unavailable from this region."
                )
                logger.error(
                    "Disabling embedding client after unsupported region response; "
                    "set a supported embedding provider or disable novelty checks."
                )

            return None
