"""
Custom exceptions for the orchestrator system with recovery hints.

All exceptions now support optional recovery_hint and category attributes
for integration with the ErrorHandler system.
"""

from typing import Optional


class OrchestratorError(Exception):
    """
    Base exception for all orchestrator-related errors.

    Supports recovery hints and error categorization for centralized
    error handling via ErrorHandler.

    Attributes:
        recovery_hint: Optional user-friendly recovery suggestion
        category: Optional error category for automatic classification
    """

    def __init__(
        self,
        message: str,
        recovery_hint: Optional[str] = None,
        category: Optional[str] = None,
    ):
        """
        Initialize orchestrator error.

        Args:
            message: Error message
            recovery_hint: Optional recovery suggestion
            category: Optional error category
        """
        super().__init__(message)
        self.recovery_hint = recovery_hint or self._default_recovery_hint()
        self.category = category

    def _default_recovery_hint(self) -> str:
        """Get default recovery hint for this error type."""
        return "Check logs for detailed error information"

    def is_retryable(self) -> bool:
        """
        Whether this error is retryable.

        Override in subclasses to specify retry behavior.
        Default: False (most errors require user intervention)

        Returns:
            True if error is retryable, False otherwise
        """
        return False


class ConfigurationError(OrchestratorError):
    """Raised when there's an error in configuration validation or loading."""

    def _default_recovery_hint(self) -> str:
        return "Review configuration file and fix invalid settings"

    def is_retryable(self) -> bool:
        return False  # Configuration errors require user intervention


class AgentCreationError(OrchestratorError):
    """Raised when there's an error creating or configuring agents."""

    def _default_recovery_hint(self) -> str:
        return "Check agent configuration and ensure all required fields are valid"

    def is_retryable(self) -> bool:
        return True  # May be transient LLM initialization issues


class TaskExecutionError(OrchestratorError):
    """Raised when there's an error during task execution."""

    def _default_recovery_hint(self) -> str:
        return (
            "Verify task definition and agent configuration. Check input data validity"
        )

    def is_retryable(self) -> bool:
        return True  # May be transient execution issues


class MemoryError(OrchestratorError):
    """Raised when there's an error with memory operations."""

    def _default_recovery_hint(self) -> str:
        return "Check memory provider configuration and resource availability"

    def is_retryable(self) -> bool:
        return False  # Memory/resource errors need system intervention


class WorkflowError(OrchestratorError):
    """Raised when there's an error in workflow execution or routing."""

    def _default_recovery_hint(self) -> str:
        return "Review workflow definition and verify agent sequence is valid"

    def is_retryable(self) -> bool:
        return True  # May be transient workflow state issues


class DependencyError(WorkflowError):
    """Raised when there are issues with task dependencies."""

    def _default_recovery_hint(self) -> str:
        return "Check task dependency graph for circular references or missing tasks"

    def is_retryable(self) -> bool:
        return False  # Dependency errors are structural


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def _default_recovery_hint(self) -> str:
        return "Fix validation errors in configuration or input data"

    def is_retryable(self) -> bool:
        return False  # Validation errors need data correction


class ProviderError(MemoryError):
    """Raised when a memory provider encounters an error."""

    def _default_recovery_hint(self) -> str:
        return "Check memory provider connection and credentials. Verify provider is running"

    def is_retryable(self) -> bool:
        return True  # Provider errors may be transient connectivity issues


class TemplateError(OrchestratorError):
    """Raised when there's an error with agent or task templates."""

    def _default_recovery_hint(self) -> str:
        return "Verify template exists and has valid format. Check template registry"

    def is_retryable(self) -> bool:
        return False  # Template errors are configuration issues


class AgentExecutionError(OrchestratorError):
    """Raised when an agent fails during task execution.

    Attributes:
        agent_name: Name of the agent that failed
        original_error: The underlying exception that caused the failure
    """

    def __init__(
        self,
        agent_name: str,
        original_error: Exception,
        recovery_hint: Optional[str] = None,
        category: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.original_error = original_error
        message = f"Agent '{agent_name}' execution failed: {original_error}"
        super().__init__(message, recovery_hint=recovery_hint, category=category)

    def _default_recovery_hint(self) -> str:
        return f"Check agent '{self.agent_name}' configuration, LLM availability, and input data"

    def is_retryable(self) -> bool:
        return True  # Agent execution failures are often transient (LLM timeouts, rate limits)


class GraphCreationError(OrchestratorError):
    """Raised when there's an error creating LangGraph StateGraphs."""

    def _default_recovery_hint(self) -> str:
        return "Check graph specification and ensure all nodes and edges are valid"

    def is_retryable(self) -> bool:
        return False  # Graph structure errors need correction


class BudgetExceededError(OrchestratorError):
    """Raised when token or cost budget is exceeded.

    Attributes:
        agent_name: Name of the agent that triggered the budget limit
        current_usage: Current usage value (tokens or cost)
        budget_limit: The configured limit that was exceeded
        budget_type: Type of budget exceeded ("tokens" or "cost")
    """

    def __init__(
        self,
        message: str,
        agent_name: str = "",
        current_usage: float = 0.0,
        budget_limit: float = 0.0,
        budget_type: str = "tokens",
        recovery_hint: Optional[str] = None,
        category: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.current_usage = current_usage
        self.budget_limit = budget_limit
        self.budget_type = budget_type
        super().__init__(
            message,
            recovery_hint=recovery_hint
            or f"Increase {budget_type} budget or reduce task complexity",
            category=category or "budget",
        )

    def _default_recovery_hint(self) -> str:
        return f"Increase {self.budget_type} budget or reduce task complexity"

    def is_retryable(self) -> bool:
        return False  # Budget exceeded requires configuration change
