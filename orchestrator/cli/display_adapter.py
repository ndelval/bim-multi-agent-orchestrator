"""
Display adapter pattern for unified orchestrator UI output.

This module provides a clean abstraction for different display backends,
enabling easy switching between Rich UI, plain console, JSON output, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DisplayAdapter(ABC):
    """
    Abstract base class for display adapters.

    Provides unified interface for different output formats:
    - RichDisplayAdapter: Beautiful Rich-based UI
    - ConsoleDisplayAdapter: Plain text output
    - JSONDisplayAdapter: Structured JSON output (future)
    - HTMLDisplayAdapter: Web-based output (future)
    """

    @abstractmethod
    def clear(self) -> None:
        """Clear display for new workflow execution."""
        pass

    @abstractmethod
    def show_header(self, user_prompt: str, backend: str = "LangGraph") -> None:
        """
        Show workflow header with user prompt.

        Args:
            user_prompt: User's input prompt
            backend: Backend type
        """
        pass

    @abstractmethod
    def show_router_decision(
        self,
        decision: str,
        confidence: str,
        rationale: str,
        latency: float,
        tokens: int = 0
    ) -> None:
        """
        Show router decision.

        Args:
            decision: Decision type (quick/research/analysis)
            confidence: Confidence level
            rationale: Decision rationale
            latency: Decision latency in seconds
            tokens: Tokens used
        """
        pass

    @abstractmethod
    def show_agent_start(self, agent_name: str, status: str) -> None:
        """
        Show agent start notification.

        Args:
            agent_name: Name of agent starting
            status: Status message
        """
        pass

    @abstractmethod
    def show_agent_complete(self, agent_name: str, summary: str) -> None:
        """
        Show agent completion notification.

        Args:
            agent_name: Name of completed agent
            summary: Completion summary
        """
        pass

    @abstractmethod
    def show_final_answer(self, answer: str) -> None:
        """
        Show final assistant response.

        Args:
            answer: Final answer text
        """
        pass

    @abstractmethod
    def show_error(
        self,
        error_message: str,
        recovery_path: Optional[str] = None
    ) -> None:
        """
        Show error with optional recovery information.

        Args:
            error_message: Error message
            recovery_path: Optional recovery path description
        """
        pass

    def show_metrics_dashboard(self) -> None:
        """Show performance and quality metrics (optional)."""
        pass  # Optional - not all adapters need this

    def show_timeline(self) -> None:
        """Show execution timeline (optional)."""
        pass  # Optional - not all adapters need this


class RichDisplayAdapter(DisplayAdapter):
    """
    Display adapter using Rich library for beautiful terminal UI.

    Wraps RichWorkflowDisplay to provide DisplayAdapter interface.
    """

    def __init__(self, rich_display=None):
        """
        Initialize Rich display adapter.

        Args:
            rich_display: Optional RichWorkflowDisplay instance
        """
        if rich_display is None:
            from .rich_display import RichWorkflowDisplay
            rich_display = RichWorkflowDisplay()

        self.rich_display = rich_display
        self._console = rich_display.console
        self._final_answer_shown = False  # Track display state to prevent duplicates

    def clear(self) -> None:
        """Clear display for new workflow execution."""
        self.rich_display.clear()
        self._final_answer_shown = False  # Reset display flag for new workflow

    def show_header(self, user_prompt: str, backend: str = "LangGraph") -> None:
        """Show workflow header with user prompt."""
        self.rich_display.show_header(user_prompt, backend)

    def show_router_decision(
        self,
        decision: str,
        confidence: str,
        rationale: str,
        latency: float,
        tokens: int = 0
    ) -> None:
        """Show router decision."""
        self.rich_display.show_router_decision(
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            latency=latency,
            tokens=tokens
        )

    def show_agent_start(self, agent_name: str, status: str) -> None:
        """Show agent start notification."""
        from .events import emit_agent_start
        emit_agent_start(agent_name, status)

    def show_agent_complete(self, agent_name: str, summary: str) -> None:
        """Show agent completion notification."""
        from .events import emit_agent_complete
        emit_agent_complete(agent_name, summary)

    def show_final_answer(self, answer: str) -> None:
        """Show final assistant response (only once to prevent duplicates)."""
        if self._final_answer_shown:
            logger.debug("Final answer already displayed, skipping duplicate display")
            return

        from .events import emit_final_answer
        emit_final_answer(answer)
        self.rich_display.show_final_output(answer)
        self._final_answer_shown = True  # Mark as shown to prevent duplicates

    def show_error(
        self,
        error_message: str,
        recovery_path: Optional[str] = None
    ) -> None:
        """Show error with optional recovery information."""
        self.rich_display.show_error(error_message, recovery_path)

    def show_metrics_dashboard(self) -> None:
        """Show performance and quality metrics."""
        self.rich_display.show_metrics_dashboard()

    def show_timeline(self) -> None:
        """Show execution timeline."""
        self.rich_display.show_timeline()


class ConsoleDisplayAdapter(DisplayAdapter):
    """
    Simple console display adapter for plain text output.

    Provides basic text-based output without Rich formatting.
    Useful for headless environments or non-terminal output.
    """

    def __init__(self):
        """Initialize console display adapter."""
        self._last_prompt = ""

    def clear(self) -> None:
        """Clear display for new workflow execution."""
        print("\n" + "="*80 + "\n")

    def show_header(self, user_prompt: str, backend: str = "LangGraph") -> None:
        """Show workflow header with user prompt."""
        self._last_prompt = user_prompt
        print("="*80)
        print("Multi-Agent Orchestrator CLI")
        print(f"Backend: {backend}")
        print("="*80)
        print(f"User Query: {user_prompt}")
        print("="*80 + "\n")

    def show_router_decision(
        self,
        decision: str,
        confidence: str,
        rationale: str,
        latency: float,
        tokens: int = 0
    ) -> None:
        """Show router decision."""
        print("--- Router Decision ---")
        print(f"Decision: {decision}")
        print(f"Confidence: {confidence}")
        print(f"Latency: {latency:.2f}s")
        if tokens > 0:
            print(f"Tokens: {tokens}")
        print(f"Rationale: {rationale[:200]}")
        print()

    def show_agent_start(self, agent_name: str, status: str) -> None:
        """Show agent start notification."""
        print(f"[{agent_name}] Starting: {status}")

    def show_agent_complete(self, agent_name: str, summary: str) -> None:
        """Show agent completion notification."""
        print(f"[{agent_name}] Complete: {summary}")

    def show_final_answer(self, answer: str) -> None:
        """Show final assistant response."""
        print("\n" + "="*80)
        print("FINAL ANSWER")
        print("="*80)
        print(answer)
        print("="*80 + "\n")

    def show_error(
        self,
        error_message: str,
        recovery_path: Optional[str] = None
    ) -> None:
        """Show error with optional recovery information."""
        print("\n" + "!"*80)
        print(f"ERROR: {error_message}")
        if recovery_path:
            print(f"Recovery: {recovery_path}")
        print("!"*80 + "\n")


# Factory function for creating display adapters
def create_display_adapter(
    adapter_type: str = "rich",
    **kwargs
) -> DisplayAdapter:
    """
    Factory function for creating display adapters.

    Args:
        adapter_type: Type of adapter ("rich", "console", "json")
        **kwargs: Additional arguments for adapter initialization

    Returns:
        DisplayAdapter instance

    Raises:
        ValueError: If adapter_type is unknown
    """
    if adapter_type == "rich":
        try:
            return RichDisplayAdapter(**kwargs)
        except ImportError:
            logger.warning("Rich not available, falling back to console adapter")
            return ConsoleDisplayAdapter()

    elif adapter_type == "console":
        return ConsoleDisplayAdapter()

    else:
        raise ValueError(
            f"Unknown display adapter type: {adapter_type}. "
            f"Available types: rich, console"
        )


__all__ = [
    "DisplayAdapter",
    "RichDisplayAdapter",
    "ConsoleDisplayAdapter",
    "create_display_adapter"
]
