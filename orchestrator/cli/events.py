"""
Event streaming system for real-time CLI UI updates.

This module provides a centralized event emitter for tracking orchestrator
execution events and streaming them to UI components in real-time.
"""

from __future__ import annotations

import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types for orchestrator execution."""

    # Workflow lifecycle
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"

    # Router events
    ROUTER_START = "router_start"
    ROUTER_DECISION = "router_decision"
    ROUTER_COMPLETE = "router_complete"

    # Planning events
    PLANNING_START = "planning_start"
    TOT_GENERATION = "tot_generation"
    TOT_EVALUATION = "tot_evaluation"
    TOT_SELECTION = "tot_selection"
    PLANNING_COMPLETE = "planning_complete"

    # Graph compilation
    GRAPH_VALIDATION = "graph_validation"
    GRAPH_COMPILATION_START = "graph_compilation_start"
    NODE_COMPILATION = "node_compilation"
    EDGE_COMPILATION = "edge_compilation"
    GRAPH_COMPILATION_COMPLETE = "graph_compilation_complete"

    # Agent execution
    EXECUTION_START = "execution_start"
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_COMPLETE = "tool_complete"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    FINAL_ANSWER = "final_answer"
    EXECUTION_COMPLETE = "execution_complete"

    # Error handling
    ERROR = "error"
    ERROR_RECOVERY = "error_recovery"
    FALLBACK_TRIGGERED = "fallback_triggered"


@dataclass
class ExecutionEvent:
    """Base class for execution events."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class EventEmitter:
    """
    Centralized event emitter for orchestrator execution.

    Provides callback-based event streaming for real-time UI updates.
    """

    def __init__(self):
        """Initialize event emitter."""
        self._callbacks: List[Callable[[ExecutionEvent], None]] = []
        self._event_history: List[ExecutionEvent] = []
        self._max_history = 1000  # Prevent memory leaks
        self._enabled = True

    def register_callback(self, callback: Callable[[ExecutionEvent], None]) -> None:
        """
        Register a callback to receive events.

        Args:
            callback: Function that receives ExecutionEvent objects
        """
        self._callbacks.append(callback)
        logger.debug(f"Registered event callback: {callback.__name__}")

    def unregister_callback(self, callback: Callable[[ExecutionEvent], None]) -> None:
        """
        Unregister a callback.

        Args:
            callback: Previously registered callback function
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Unregistered event callback: {callback.__name__}")

    def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event_type: Type of event
            data: Optional event data
        """
        if not self._enabled:
            return

        event = ExecutionEvent(
            event_type=event_type,
            data=data or {}
        )

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback {callback.__name__}: {e}")

    def get_history(self,
                   event_type: Optional[EventType] = None,
                   limit: Optional[int] = None) -> List[ExecutionEvent]:
        """
        Get event history, optionally filtered by type.

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events in chronological order
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if limit:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def export_json(self, filepath: str) -> None:
        """
        Export event history to JSON file.

        Args:
            filepath: Path to output file
        """
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in self._event_history], f, indent=2)


# Global singleton instance
_global_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter singleton."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter()
    return _global_emitter


def reset_event_emitter() -> None:
    """Reset the global event emitter (useful for testing)."""
    global _global_emitter
    _global_emitter = None


# Convenience functions for emitting common events

def emit_workflow_start(prompt: str, user_id: str) -> None:
    """Emit workflow start event."""
    get_event_emitter().emit(EventType.WORKFLOW_START, {
        "prompt": prompt,
        "user_id": user_id
    })


def emit_router_decision(decision: str, confidence: str, rationale: str,
                         latency: float, tokens: Optional[int] = None) -> None:
    """Emit router decision event."""
    get_event_emitter().emit(EventType.ROUTER_DECISION, {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "latency": latency,
        "tokens": tokens
    })


def emit_tot_generation(step: int, candidates: int, method: str) -> None:
    """Emit ToT thought generation event."""
    get_event_emitter().emit(EventType.TOT_GENERATION, {
        "step": step,
        "candidates": candidates,
        "method": method
    })


def emit_tot_evaluation(step: int, scores: List[float]) -> None:
    """Emit ToT evaluation event."""
    get_event_emitter().emit(EventType.TOT_EVALUATION, {
        "step": step,
        "scores": scores,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "max_score": max(scores) if scores else 0
    })


def emit_tot_selection(step: int, kept: int, pruned: int) -> None:
    """Emit ToT selection event."""
    get_event_emitter().emit(EventType.TOT_SELECTION, {
        "step": step,
        "kept": kept,
        "pruned": pruned
    })


def emit_planning_complete(method: str, quality_score: float,
                          assignments: int, duration: float) -> None:
    """Emit planning complete event."""
    get_event_emitter().emit(EventType.PLANNING_COMPLETE, {
        "method": method,
        "quality_score": quality_score,
        "assignments": assignments,
        "duration": duration
    })


def emit_node_start(node_name: str, agent_name: str, tools: List[str]) -> None:
    """Emit node start event."""
    get_event_emitter().emit(EventType.NODE_START, {
        "node_name": node_name,
        "agent_name": agent_name,
        "tools": tools
    })


def emit_node_complete(node_name: str, agent_name: str,
                       duration: float, output_length: int) -> None:
    """Emit node complete event."""
    get_event_emitter().emit(EventType.NODE_COMPLETE, {
        "node_name": node_name,
        "agent_name": agent_name,
        "duration": duration,
        "output_length": output_length
    })


def emit_tool_invocation(tool_name: str, agent_name: str, args: Dict[str, Any]) -> None:
    """Emit tool invocation event."""
    get_event_emitter().emit(EventType.TOOL_INVOCATION, {
        "tool_name": tool_name,
        "agent_name": agent_name,
        "args": args
    })


def emit_tool_complete(tool_name: str, agent_name: str,
                      results_count: int, duration: float) -> None:
    """Emit tool complete event."""
    get_event_emitter().emit(EventType.TOOL_COMPLETE, {
        "tool_name": tool_name,
        "agent_name": agent_name,
        "results_count": results_count,
        "duration": duration
    })


def emit_agent_start(agent_name: str, status: str) -> None:
    """Emit agent start event (compat helper for legacy CLI)."""
    get_event_emitter().emit(EventType.AGENT_START, {
        "agent_name": agent_name,
        "status": status,
    })


def emit_agent_progress(agent_name: str, message: str,
                        progress: Optional[float] = None) -> None:
    """Emit agent progress update."""
    payload = {
        "agent_name": agent_name,
        "message": message,
    }
    if progress is not None:
        payload["progress"] = progress
    get_event_emitter().emit(EventType.AGENT_PROGRESS, payload)


def emit_agent_complete(agent_name: str, summary: str) -> None:
    """Emit agent completion event."""
    get_event_emitter().emit(EventType.AGENT_COMPLETE, {
        "agent_name": agent_name,
        "summary": summary,
    })


def emit_final_answer(answer: str) -> None:
    """Emit final answer event for UI display."""
    get_event_emitter().emit(EventType.FINAL_ANSWER, {
        "answer": answer,
    })


def emit_error_recovery(error_type: str, recovery_path: str, impact: str) -> None:
    """Emit error recovery event."""
    get_event_emitter().emit(EventType.ERROR_RECOVERY, {
        "error_type": error_type,
        "recovery_path": recovery_path,
        "impact": impact
    })


def emit_workflow_complete(duration: float, success: bool) -> None:
    """Emit workflow complete event."""
    get_event_emitter().emit(EventType.WORKFLOW_COMPLETE, {
        "duration": duration,
        "success": success
    })
