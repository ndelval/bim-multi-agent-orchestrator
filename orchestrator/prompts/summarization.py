"""Summarization prompt template (BP-MCP-05, BP-PROMPT-08).

Template: ``summarization.compress``
Placeholder: ``{conversation}``
"""

from .registry import PromptRegistry

_SUMMARIZATION_COMPRESS = (
    "Summarize this conversation history into key facts, decisions, "
    "and state changes. Keep under 500 tokens.\n"
    "Format as:\n"
    "KeyFacts: [...]\n"
    "Decisions: [...]\n"
    "CurrentState: {{...}}\n\n"
    "Conversation:\n{conversation}"
)

PromptRegistry.register("summarization.compress", _SUMMARIZATION_COMPRESS)
