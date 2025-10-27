"""
Refactored OrchestratorState implementation with improved type safety and maintainability.

This is a reference implementation showing the refactoring improvements.
To integrate: review, test, then replace OrchestratorState in langchain_integration.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Type, Literal
from typing_extensions import TypedDict
import traceback
import logging

from langchain_core.messages import BaseMessage
from orchestrator.core.constants import EXECUTION_PATH_DISPLAY_LIMIT
from orchestrator.core.exceptions import (
    ValidationError,
    TaskExecutionError,
    MemoryError,
    OrchestratorError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Structured Type Definitions
# ============================================================================

class RouterDecision(TypedDict, total=False):
    """Structured router decision output with type constraints."""
    route: Literal["quick", "research", "analysis", "standards"]
    confidence: float
    reasoning: str
    assigned_agents: List[str]


class Assignment(TypedDict):
    """Task assignment structure for agent coordination."""
    agent_name: str
    task_description: str
    dependencies: List[str]
    priority: int


@dataclass(frozen=True)
class ExecutionError:
    """
    Structured error information for workflow failures.

    Immutable dataclass that captures complete error context for debugging
    and recovery decisions.
    """
    error_type: Type[Exception]
    error_message: str
    node_name: Optional[str] = None
    agent_name: Optional[str] = None
    stack_trace: Optional[str] = None
    is_recoverable: bool = False
    recovery_hint: Optional[str] = None

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        is_recoverable: bool = False
    ) -> "ExecutionError":
        """
        Create ExecutionError from caught exception.

        Args:
            exc: The caught exception
            node_name: Name of graph node where error occurred
            agent_name: Name of agent that raised error
            is_recoverable: Whether automatic recovery is possible

        Returns:
            ExecutionError with captured context
        """
        return cls(
            error_type=type(exc),
            error_message=str(exc),
            node_name=node_name,
            agent_name=agent_name,
            stack_trace=traceback.format_exc(),
            is_recoverable=is_recoverable,
            recovery_hint=cls._suggest_recovery(exc)
        )

    @staticmethod
    def _suggest_recovery(exc: Exception) -> Optional[str]:
        """Suggest recovery actions based on error type."""
        if isinstance(exc, TaskExecutionError):
            return "Retry with modified task parameters or reduced scope"
        elif isinstance(exc, MemoryError):
            return "Check memory provider connectivity and configuration"
        elif isinstance(exc, ValidationError):
            return "Review input parameters and state consistency"
        return "Review error details and check system logs"

    def __str__(self) -> str:
        """Human-readable error description."""
        parts = [f"{self.error_type.__name__}: {self.error_message}"]
        if self.node_name:
            parts.append(f"Node: {self.node_name}")
        if self.agent_name:
            parts.append(f"Agent: {self.agent_name}")
        if self.recovery_hint:
            parts.append(f"Recovery: {self.recovery_hint}")
        return " | ".join(parts)


# ============================================================================
# Refactored State Schema
# ============================================================================

@dataclass
class OrchestratorState:
    """
    State schema for LangGraph StateGraph orchestration.

    This state object is passed between nodes in a LangGraph workflow,
    tracking execution progress, agent outputs, and error conditions.

    Key improvements over previous implementation:
    - Uses field(default_factory=...) for all mutable defaults (type safety)
    - Includes all production-used fields (no dynamic attribute creation)
    - Structured error handling with ExecutionError
    - Type-safe routing with Literal types
    - Validation logic in __post_init__
    - Helper methods for common operations (DRY principle)
    """

    # ========================================================================
    # Required Input Fields
    # ========================================================================

    messages: List[BaseMessage]
    """LangChain message history for conversation context."""

    user_prompt: str
    """Original user query that initiated the workflow."""

    # ========================================================================
    # Output Fields
    # ========================================================================

    final_output: Optional[str] = None
    """Final response to be returned to user."""

    # ========================================================================
    # Routing and Decision Making
    # ========================================================================

    current_route: Optional[Literal["quick", "research", "analysis", "standards"]] = None
    """Current execution route determined by router agent."""

    router_decision: Optional[RouterDecision] = None
    """Structured router decision with confidence and reasoning."""

    assignments: List[Assignment] = field(default_factory=list)
    """Task assignments for agent coordination."""

    # ========================================================================
    # Agent Execution Tracking
    # ========================================================================

    current_agent: Optional[str] = None
    """Name of currently executing agent."""

    agent_outputs: Dict[str, str] = field(default_factory=dict)
    """Mapping of agent names to their execution results."""

    completed_agents: List[str] = field(default_factory=list)
    """List of agents that have completed execution."""

    # ========================================================================
    # Graph Execution Tracking
    # ========================================================================

    current_node: Optional[str] = None
    """Name of currently executing graph node."""

    execution_path: List[str] = field(default_factory=list)
    """Ordered list of executed node names (execution trace)."""

    node_outputs: Dict[str, str] = field(default_factory=dict)
    """Mapping of node names to their execution results."""

    condition_results: Dict[str, bool] = field(default_factory=dict)
    """Results of condition node evaluations for edge routing."""

    parallel_execution_active: bool = False
    """Flag indicating whether parallel execution is in progress."""

    # ========================================================================
    # Memory and Context
    # ========================================================================

    recall_items: List[str] = field(default_factory=list)
    """Items retrieved from memory system for context."""

    memory_context: Optional[str] = None
    """Formatted memory context for LLM prompts."""

    # ========================================================================
    # Execution Control
    # ========================================================================

    max_iterations: int = 10
    """Maximum number of agent execution iterations allowed."""

    current_iteration: int = 0
    """Current iteration count (increments with each agent execution)."""

    error_state: Optional[ExecutionError] = None
    """Structured error information if workflow failed."""

    # ========================================================================
    # Validation and Initialization
    # ========================================================================

    def __post_init__(self):
        """
        Validate state consistency after initialization.

        Raises:
            ValidationError: If state violates consistency constraints
        """
        self._validate_iteration_bounds()
        self._validate_route_consistency()
        self._validate_required_fields()

    def _validate_iteration_bounds(self) -> None:
        """Ensure iteration counts are within valid bounds."""
        if self.current_iteration < 0:
            raise ValidationError(
                f"current_iteration cannot be negative: {self.current_iteration}"
            )

        if self.current_iteration > self.max_iterations:
            raise ValidationError(
                f"current_iteration ({self.current_iteration}) exceeds "
                f"max_iterations ({self.max_iterations})"
            )

        if self.max_iterations <= 0:
            raise ValidationError(
                f"max_iterations must be positive: {self.max_iterations}"
            )

    def _validate_route_consistency(self) -> None:
        """Ensure routing state is consistent."""
        if self.current_route and not self.router_decision:
            logger.warning(
                f"current_route set to '{self.current_route}' without router_decision"
            )

        if self.router_decision:
            decision_route = self.router_decision.get("route")
            if decision_route != self.current_route:
                raise ValidationError(
                    f"Inconsistent routing: current_route='{self.current_route}', "
                    f"router_decision route='{decision_route}'"
                )

    def _validate_required_fields(self) -> None:
        """Validate required fields are present and non-empty."""
        if not self.messages:
            raise ValidationError("messages list cannot be empty")

        if not self.user_prompt or not self.user_prompt.strip():
            raise ValidationError("user_prompt is required and cannot be empty")

    # ========================================================================
    # Helper Methods (DRY Principle)
    # ========================================================================

    def record_node_execution(
        self,
        node_name: str,
        output: str,
        update_current: bool = True
    ) -> None:
        """
        Record execution of a graph node.

        Centralizes the pattern of updating current_node, execution_path,
        and node_outputs to ensure consistency.

        Args:
            node_name: Name of the executed node
            output: Result of node execution
            update_current: Whether to update current_node tracking
        """
        if update_current:
            self.current_node = node_name
        self.execution_path.append(node_name)
        self.node_outputs[node_name] = output
        logger.debug(f"Recorded execution: {node_name} -> {output[:50]}...")

    def record_agent_completion(
        self,
        agent_name: str,
        result: str
    ) -> None:
        """
        Record successful agent execution.

        Updates agent_outputs, completed_agents, and increments iteration counter.

        Args:
            agent_name: Name of the completed agent
            result: Agent execution result
        """
        self.agent_outputs[agent_name] = result
        self.completed_agents.append(agent_name)
        self.current_iteration += 1
        logger.info(
            f"Agent {agent_name} completed (iteration {self.current_iteration}/{self.max_iterations})"
        )

    def record_error(
        self,
        exc: Exception,
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        is_recoverable: bool = False
    ) -> None:
        """
        Record workflow error with structured context.

        Args:
            exc: The caught exception
            node_name: Name of node where error occurred
            agent_name: Name of agent that raised error
            is_recoverable: Whether automatic recovery is possible
        """
        self.error_state = ExecutionError.from_exception(
            exc=exc,
            node_name=node_name or self.current_node,
            agent_name=agent_name or self.current_agent,
            is_recoverable=is_recoverable
        )
        logger.error(f"Workflow error recorded: {self.error_state}")

    def is_iteration_limit_reached(self) -> bool:
        """Check if maximum iterations reached."""
        return self.current_iteration >= self.max_iterations

    def has_error(self) -> bool:
        """Check if workflow is in error state."""
        return self.error_state is not None

    def can_recover_from_error(self) -> bool:
        """Check if current error allows automatic recovery."""
        return self.error_state is not None and self.error_state.is_recoverable

    def get_last_agent_output(self) -> Optional[str]:
        """Get most recent agent output."""
        if not self.completed_agents:
            return None
        last_agent = self.completed_agents[-1]
        return self.agent_outputs.get(last_agent)

    def get_execution_summary(self) -> Dict[str, any]:
        """
        Generate execution summary for logging/debugging.

        Returns:
            Dictionary with key execution metrics
        """
        return {
            "iteration": f"{self.current_iteration}/{self.max_iterations}",
            "completed_agents": len(self.completed_agents),
            "execution_path": " -> ".join(self.execution_path[-EXECUTION_PATH_DISPLAY_LIMIT:]),  # Last N nodes
            "has_error": self.has_error(),
            "current_route": self.current_route,
            "parallel_active": self.parallel_execution_active
        }

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        summary = self.get_execution_summary()
        return (
            f"OrchestratorState(iteration={summary['iteration']}, "
            f"route={summary['current_route']}, "
            f"agents={summary['completed_agents']}, "
            f"error={summary['has_error']})"
        )