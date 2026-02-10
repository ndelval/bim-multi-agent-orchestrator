"""
OpenAI LLM client implementation for BIM-IR agent.
"""

import logging
from typing import Optional

from openai import OpenAI, OpenAIError, APITimeoutError

from .base import LLMClient, LLMError, LLMTimeoutError, LLMAPIError

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        timeout: int = 30
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4-turbo)
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        response_format: Optional[str] = None
    ) -> str:
        """
        Generate text completion using OpenAI API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            response_format: Optional format ("json" for JSON mode)

        Returns:
            Generated text response

        Raises:
            LLMTimeoutError: If request times out
            LLMAPIError: If API returns an error
        """
        try:
            logger.debug(f"Generating with model={self._model}, temp={temperature}, max_tokens={max_tokens}")

            # Build request parameters
            params = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add JSON mode if requested
            if response_format == "json":
                params["response_format"] = {"type": "json_object"}

            # Make API call
            response = self.client.chat.completions.create(**params)

            # Extract content
            content = response.choices[0].message.content

            logger.debug(f"Generated {len(content)} characters")
            return content

        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}")
            raise LLMTimeoutError(f"Request timed out: {e}") from e

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMAPIError(f"API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI client: {e}")
            raise LLMError(f"Unexpected error: {e}") from e

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model
