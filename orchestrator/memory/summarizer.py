"""
Conversation history summarizer for context windowing (BP-COST-07).

Compresses older messages into a structured summary to prevent
unbounded context growth and reduce token costs.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Attempt BaseMessage import; fall back to a protocol-compatible type hint
try:
    from langchain.schema import BaseMessage
except ImportError:
    BaseMessage = Any  # type: ignore[assignment,misc]


class Summarizer:
    """LLM-based conversation history summarizer.

    Uses LLMFactory to create a cheap, fast model that compresses
    a list of BaseMessage objects into a structured summary string.

    The LLM is created lazily on the first call to avoid import-time
    side-effects and unnecessary API-key validation.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._llm: Optional[Any] = None  # Created lazily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        """Return the LLM instance, creating it on first access."""
        if self._llm is None:
            from orchestrator.core.llm_factory import LLMFactory

            self._llm = LLMFactory.create(
                provider=self._provider,
                model=self._model,
                temperature=self._temperature,
                max_tokens=600,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_old_messages(self, old_messages: List[Any]) -> str:
        """Summarize a list of messages into a structured summary string.

        Args:
            old_messages: List of BaseMessage instances to summarize.

        Returns:
            A structured summary string, or empty string if input is empty.
        """
        if not old_messages:
            return ""

        # Short histories: format directly without an LLM call
        if len(old_messages) <= 2:
            return "; ".join(
                f"{type(m).__name__}: {m.content[:200]}" for m in old_messages
            )

        # Format conversation for the prompt
        conversation = "\n".join(
            f"{type(m).__name__}: {m.content}" for m in old_messages
        )
        from orchestrator.prompts import get_prompt

        prompt = get_prompt("summarization.compress", conversation=conversation)

        llm = self._get_llm()
        result = llm.invoke(prompt)

        # Extract content from AIMessage or plain string
        content = result.content if hasattr(result, "content") else str(result)
        logger.debug(
            "Summarized %d messages into %d chars", len(old_messages), len(content)
        )
        return content
