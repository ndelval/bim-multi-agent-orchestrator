"""
Value objects for orchestrator domain models.

This module provides immutable, validated data structures that encapsulate
related data and reduce parameter passing across the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence

from .config import MemoryConfig


@dataclass(frozen=True)
class RouterDecision:
    """
    Value object for router decision results.

    Encapsulates all routing decision data in a type-safe,
    immutable structure. Replaces error-prone dict/tuple passing.

    Attributes:
        decision: Route type (quick, research, analysis, planning, standards)
        confidence: Confidence level (High, Medium, Low)
        reasoning: Decision rationale and explanation
        latency: Decision time in seconds
        assigned_agents: List of agents assigned to this route
        tokens: Number of tokens used in decision
        payload: Additional metadata from router

    Raises:
        ValueError: If decision type is invalid or latency is negative
    """
    decision: str
    confidence: str
    reasoning: str
    latency: float
    assigned_agents: List[str] = field(default_factory=list)
    tokens: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate router decision data."""
        valid_decisions = ["quick", "research", "analysis", "planning", "standards"]
        if self.decision.lower() not in valid_decisions:
            raise ValueError(
                f"Invalid decision: '{self.decision}'. "
                f"Must be one of: {', '.join(valid_decisions)}"
            )

        if self.latency < 0:
            raise ValueError(f"Latency cannot be negative: {self.latency}")

        if self.tokens < 0:
            raise ValueError(f"Token count cannot be negative: {self.tokens}")

    def get_payload_value(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieve value from payload dictionary.

        Args:
            key: Payload key to retrieve
            default: Default value if key not found

        Returns:
            Value from payload or default
        """
        return self.payload.get(key, default)


@dataclass(frozen=True)
class ExecutionContext:
    """
    Value object for workflow execution context.

    Reduces parameter passing by encapsulating related workflow
    configuration data in a single, validated structure.

    Attributes:
        prompt: User query or task description
        user_id: Unique user identifier
        verbose: Verbosity level (0=silent, 1=normal, 2=debug)
        max_iterations: Maximum workflow iterations
        recall_items: Memory items to recall for context
        assignments: Task assignments for workflow
        base_memory_config: Memory configuration for workflow

    Raises:
        ValueError: If max_iterations is invalid or verbose level is out of range
    """
    prompt: str
    user_id: str = "default_user"
    verbose: int = 1
    max_iterations: int = 6
    recall_items: Sequence[str] = field(default_factory=list)
    assignments: Optional[List[Dict[str, Any]]] = None
    base_memory_config: Optional[MemoryConfig] = None

    def __post_init__(self):
        """Validate execution context data."""
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got: {self.max_iterations}"
            )

        if not (0 <= self.verbose <= 2):
            raise ValueError(
                f"verbose must be 0-2, got: {self.verbose}"
            )

        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt cannot be empty")

    def with_assignments(self, assignments: List[Dict[str, Any]]) -> "ExecutionContext":
        """
        Create new ExecutionContext with updated assignments.

        Since ExecutionContext is immutable, this creates a new instance
        with the specified assignments while preserving other fields.

        Args:
            assignments: New task assignments

        Returns:
            New ExecutionContext with updated assignments
        """
        return ExecutionContext(
            prompt=self.prompt,
            user_id=self.user_id,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            recall_items=self.recall_items,
            assignments=assignments,
            base_memory_config=self.base_memory_config
        )

    def with_recall_items(self, recall_items: Sequence[str]) -> "ExecutionContext":
        """
        Create new ExecutionContext with updated recall items.

        Args:
            recall_items: New memory recall items

        Returns:
            New ExecutionContext with updated recall items
        """
        return ExecutionContext(
            prompt=self.prompt,
            user_id=self.user_id,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            recall_items=recall_items,
            assignments=self.assignments,
            base_memory_config=self.base_memory_config
        )


__all__ = ["RouterDecision", "ExecutionContext"]
