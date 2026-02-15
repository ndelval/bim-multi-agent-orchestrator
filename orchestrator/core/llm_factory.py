"""
LLM Factory for creating provider-agnostic chat model instances.

Supports OpenAI (required), Anthropic (optional), and Cohere (optional).
All returned instances implement the LangChain BaseChatModel interface,
allowing agents to swap providers without changing agent logic.

Reference: prompting.txt best practice -
"A ModelClient class that normalizes calls to OpenAI/Anthropic/Bedrock,
while agents only see a generate(messages, system_prompt, max_tokens) function."
"""

import importlib
import logging
from typing import Optional, Any, Dict, List

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Default models per provider
_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "cohere": "command-r-plus",
}

# Provider registry: provider_name -> (langchain_package, class_name)
_PROVIDER_REGISTRY: Dict[str, tuple] = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "cohere": ("langchain_cohere", "ChatCohere"),
}


class LLMFactory:
    """
    Factory for creating LangChain BaseChatModel instances from provider configuration.

    All LLM creation in the orchestrator should go through this factory to maintain
    provider-agnostic agent logic. Optional providers (Anthropic, Cohere) are imported
    at runtime via importlib so their packages are not required at install time.

    Usage:
        llm = LLMFactory.create(provider="openai", model="gpt-4o-mini", temperature=0.1)
        llm = LLMFactory.create(provider="anthropic", model="claude-sonnet-4-20250514")
        llm = LLMFactory.create()  # defaults to openai/gpt-4o-mini
    """

    @staticmethod
    def create(
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs: Any,
    ):
        """
        Create a LangChain chat model instance.

        Args:
            provider: LLM provider name ("openai", "anthropic", "cohere")
            model: Model identifier. If None, uses provider's default model.
            temperature: Sampling temperature (default 0.1)
            **kwargs: Additional provider-specific parameters (e.g., max_tokens)

        Returns:
            BaseChatModel instance from the specified provider

        Raises:
            ConfigurationError: If provider is unsupported or its package is not installed
        """
        provider = provider.lower().strip()

        if provider not in _PROVIDER_REGISTRY:
            supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
            raise ConfigurationError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported providers: {supported}",
                recovery_hint=f"Use one of: {supported}",
            )

        module_name, class_name = _PROVIDER_REGISTRY[provider]
        resolved_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        # Dynamically import the provider package
        try:
            module = importlib.import_module(module_name)
            chat_class = getattr(module, class_name)
        except ImportError:
            raise ConfigurationError(
                f"Provider '{provider}' requires package '{module_name}' "
                f"which is not installed.",
                recovery_hint=f"Install it with: pip install {module_name}",
            )

        # Instantiate the chat model
        try:
            llm = chat_class(model=resolved_model, temperature=temperature, **kwargs)
            logger.info(
                "Created %s (provider=%s, model=%s, temperature=%s)",
                class_name,
                provider,
                resolved_model,
                temperature,
            )
            return llm
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create LLM instance for provider '{provider}', "
                f"model '{resolved_model}': {str(e)}",
                recovery_hint=(
                    f"Check that the API key environment variable is set "
                    f"and model '{resolved_model}' is valid for provider '{provider}'"
                ),
            )

    @staticmethod
    def get_default_model(provider: str = "openai") -> str:
        """Get the default model name for a provider.

        Args:
            provider: Provider name

        Returns:
            Default model identifier string
        """
        return _DEFAULT_MODELS.get(provider.lower(), "gpt-4o-mini")

    @staticmethod
    def list_providers() -> List[str]:
        """List all supported provider names.

        Returns:
            Sorted list of provider name strings
        """
        return sorted(_PROVIDER_REGISTRY.keys())
