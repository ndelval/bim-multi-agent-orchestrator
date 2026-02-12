"""
Lifecycle management for the Orchestrator.

This module handles callback management and event emission,
following the Single Responsibility Principle by separating
lifecycle concerns from initialization and execution logic.
"""

import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages lifecycle events and callbacks for the orchestrator.

    This class handles registration and emission of workflow and task events,
    providing a centralized mechanism for lifecycle management.
    """

    def __init__(self):
        """Initialize the lifecycle manager."""
        # Workflow callbacks
        self.on_workflow_start: Optional[Callable[[], None]] = None
        self.on_workflow_complete: Optional[Callable[[Any], None]] = None

        # Task callbacks
        self.on_task_start: Optional[Callable[[str, Any], None]] = None
        self.on_task_complete: Optional[Callable[[str, Any], None]] = None

        # Error callback
        self.on_error: Optional[Callable[[Exception], None]] = None

    def register_workflow_start_callback(self, callback: Callable[[], None]) -> None:
        """
        Register callback for workflow start event.

        Args:
            callback: Function to call when workflow starts
        """
        self.on_workflow_start = callback
        logger.debug("Registered workflow start callback")

    def register_workflow_complete_callback(
        self, callback: Callable[[Any], None]
    ) -> None:
        """
        Register callback for workflow completion event.

        Args:
            callback: Function to call when workflow completes
        """
        self.on_workflow_complete = callback
        logger.debug("Registered workflow complete callback")

    def register_task_start_callback(
        self, callback: Callable[[str, Any], None]
    ) -> None:
        """
        Register callback for task start event.

        Args:
            callback: Function to call when a task starts
        """
        self.on_task_start = callback
        logger.debug("Registered task start callback")

    def register_task_complete_callback(
        self, callback: Callable[[str, Any], None]
    ) -> None:
        """
        Register callback for task completion event.

        Args:
            callback: Function to call when a task completes
        """
        self.on_task_complete = callback
        logger.debug("Registered task complete callback")

    def register_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Register callback for error events.

        Args:
            callback: Function to call when an error occurs
        """
        self.on_error = callback
        logger.debug("Registered error callback")

    def emit_workflow_start(self) -> None:
        """Emit workflow start event."""
        if self.on_workflow_start:
            try:
                self.on_workflow_start()
                logger.debug("Emitted workflow start event")
            except Exception as e:
                logger.error(f"Error in workflow start callback: {e}")

    def emit_workflow_complete(self, metrics: Any) -> None:
        """
        Emit workflow completion event.

        Args:
            metrics: Workflow execution metrics
        """
        if self.on_workflow_complete:
            try:
                self.on_workflow_complete(metrics)
                logger.debug("Emitted workflow complete event")
            except Exception as e:
                logger.error(f"Error in workflow complete callback: {e}")

        # Log metrics regardless of callback
        logger.info(
            f"Workflow completed - Duration: {metrics.total_duration:.2f}s, "
            f"Tasks: {metrics.completed_tasks}/{metrics.total_tasks}"
        )

    def emit_task_start(self, task_name: str, execution: Any) -> None:
        """
        Emit task start event.

        Args:
            task_name: Name of the task
            execution: Task execution context
        """
        if self.on_task_start:
            try:
                self.on_task_start(task_name, execution)
                logger.debug(f"Emitted task start event: {task_name}")
            except Exception as e:
                logger.error(f"Error in task start callback: {e}")
        else:
            logger.debug(f"Task started: {task_name}")

    def emit_task_complete(self, task_name: str, execution: Any) -> None:
        """
        Emit task completion event.

        Args:
            task_name: Name of the task
            execution: Task execution context
        """
        if self.on_task_complete:
            try:
                self.on_task_complete(task_name, execution)
                logger.debug(f"Emitted task complete event: {task_name}")
            except Exception as e:
                logger.error(f"Error in task complete callback: {e}")
        else:
            logger.debug(f"Task completed: {task_name}")

    def emit_task_fail(self, task_name: str, execution: Any) -> None:
        """
        Emit task failure event.

        Args:
            task_name: Name of the task
            execution: Task execution context
        """
        logger.warning(f"Task failed: {task_name}")

        if self.on_error and hasattr(execution, "error"):
            try:
                self.on_error(execution.error)
                logger.debug(f"Emitted error event for task: {task_name}")
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def emit_error(self, error: Exception) -> None:
        """
        Emit error event.

        Args:
            error: Exception that occurred
        """
        if self.on_error:
            try:
                self.on_error(error)
                logger.debug("Emitted error event")
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def get_workflow_callbacks(self) -> dict:
        """
        Get workflow-related callbacks for workflow engine registration.

        Returns:
            Dictionary of callback names to callback functions
        """
        return {
            "on_task_start": self.emit_task_start,
            "on_task_complete": self.emit_task_complete,
            "on_task_fail": self.emit_task_fail,
            "on_workflow_complete": self.emit_workflow_complete,
        }

    def reset(self) -> None:
        """Reset all registered callbacks."""
        self.on_workflow_start = None
        self.on_workflow_complete = None
        self.on_task_start = None
        self.on_task_complete = None
        self.on_error = None
        logger.debug("All lifecycle callbacks reset")
