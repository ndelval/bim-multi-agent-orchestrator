"""
Custom exceptions for the orchestrator system.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator-related errors."""
    pass


class ConfigurationError(OrchestratorError):
    """Raised when there's an error in configuration validation or loading."""
    pass


class AgentCreationError(OrchestratorError):
    """Raised when there's an error creating or configuring agents."""
    pass


class TaskExecutionError(OrchestratorError):
    """Raised when there's an error during task execution."""
    pass


class MemoryError(OrchestratorError):
    """Raised when there's an error with memory operations."""
    pass


class WorkflowError(OrchestratorError):
    """Raised when there's an error in workflow execution or routing."""
    pass


class DependencyError(WorkflowError):
    """Raised when there are issues with task dependencies."""
    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class ProviderError(MemoryError):
    """Raised when a memory provider encounters an error."""
    pass


class TemplateError(OrchestratorError):
    """Raised when there's an error with agent or task templates."""
    pass


class GraphCreationError(OrchestratorError):
    """Raised when there's an error creating LangGraph StateGraphs."""
    pass