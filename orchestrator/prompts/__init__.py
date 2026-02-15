"""Centralized prompt template registry (BP-MCP-05, BP-PROMPT-08).

All LLM-facing prompt templates are registered here and retrieved
via ``get_prompt(name, **kwargs)`` from any call site.
"""

from .registry import PromptRegistry, get_prompt

# Import submodules to trigger template auto-registration
from . import planning, routing, summarization, system  # noqa: F401

__all__ = ["PromptRegistry", "get_prompt"]
