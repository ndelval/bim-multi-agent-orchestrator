"""
Base LLM client abstraction for BIM-IR agent.

Provides a common interface for different LLM providers (OpenAI, Anthropic, local models).
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        response_format: Optional[str] = None
    ) -> str:
        """
        Generate text completion from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response
            response_format: Optional format specification (e.g., "json")

        Returns:
            Generated text response

        Raises:
            LLMError: If generation fails
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
    pass


class LLMAPIError(LLMError):
    """LLM API returned an error."""
    pass


class LLMInvalidResponseError(LLMError):
    """LLM returned an invalid or unparseable response."""
    pass
