"""
Unit tests for LifecycleManager.

Tests callback management and event emission following the Observer pattern.
"""

import pytest
from unittest.mock import Mock, call
from orchestrator.core.lifecycle import LifecycleManager
from orchestrator.workflow.workflow_engine import WorkflowMetrics


class TestLifecycleManager:
    """Tests for LifecycleManager class."""

    @pytest.fixture
    def lifecycle(self):
        """Create lifecycle manager instance."""
        return LifecycleManager()

    def test_lifecycle_creation(self, lifecycle):
        """Test lifecycle manager can be created."""
        assert lifecycle.on_workflow_start is None
        assert lifecycle.on_workflow_complete is None
        assert lifecycle.on_task_start is None
        assert lifecycle.on_task_complete is None
        assert lifecycle.on_error is None

    def test_register_workflow_start_callback(self, lifecycle):
        """Test workflow start callback registration."""
        callback = Mock()
        lifecycle.register_workflow_start_callback(callback)
        assert lifecycle.on_workflow_start == callback

    def test_register_workflow_complete_callback(self, lifecycle):
        """Test workflow complete callback registration."""
        callback = Mock()
        lifecycle.register_workflow_complete_callback(callback)
        assert lifecycle.on_workflow_complete == callback

    def test_register_task_start_callback(self, lifecycle):
        """Test task start callback registration."""
        callback = Mock()
        lifecycle.register_task_start_callback(callback)
        assert lifecycle.on_task_start == callback

    def test_register_task_complete_callback(self, lifecycle):
        """Test task complete callback registration."""
        callback = Mock()
        lifecycle.register_task_complete_callback(callback)
        assert lifecycle.on_task_complete == callback

    def test_register_error_callback(self, lifecycle):
        """Test error callback registration."""
        callback = Mock()
        lifecycle.register_error_callback(callback)
        assert lifecycle.on_error == callback

    def test_emit_workflow_start(self, lifecycle):
        """Test workflow start event emission."""
        callback = Mock()
        lifecycle.on_workflow_start = callback

        lifecycle.emit_workflow_start()

        callback.assert_called_once()

    def test_emit_workflow_start_no_callback(self, lifecycle):
        """Test workflow start emission without callback."""
        # Should not raise
        lifecycle.emit_workflow_start()

    def test_emit_workflow_start_callback_error(self, lifecycle, caplog):
        """Test workflow start callback error handling."""
        callback = Mock(side_effect=Exception("Callback failed"))
        lifecycle.on_workflow_start = callback

        lifecycle.emit_workflow_start()

        # Should log error but not raise
        assert "Error in workflow start callback" in caplog.text

    def test_emit_workflow_complete(self, lifecycle):
        """Test workflow complete event emission."""
        callback = Mock()
        lifecycle.on_workflow_complete = callback
        metrics = WorkflowMetrics(
            total_tasks=5,
            completed_tasks=5,
            failed_tasks=0,
            total_duration=10.5
        )

        lifecycle.emit_workflow_complete(metrics)

        callback.assert_called_once_with(metrics)

    def test_emit_workflow_complete_no_callback(self, lifecycle):
        """Test workflow complete emission without callback."""
        metrics = WorkflowMetrics(
            total_tasks=5,
            completed_tasks=5,
            failed_tasks=0,
            total_duration=10.5
        )

        # Should not raise, just log
        lifecycle.emit_workflow_complete(metrics)

    def test_emit_workflow_complete_callback_error(self, lifecycle, caplog):
        """Test workflow complete callback error handling."""
        callback = Mock(side_effect=Exception("Callback failed"))
        lifecycle.on_workflow_complete = callback
        metrics = WorkflowMetrics(
            total_tasks=5,
            completed_tasks=5,
            failed_tasks=0,
            total_duration=10.5
        )

        lifecycle.emit_workflow_complete(metrics)

        assert "Error in workflow complete callback" in caplog.text

    def test_emit_task_start(self, lifecycle):
        """Test task start event emission."""
        callback = Mock()
        lifecycle.on_task_start = callback
        execution = Mock()

        lifecycle.emit_task_start("task1", execution)

        callback.assert_called_once_with("task1", execution)

    def test_emit_task_start_no_callback(self, lifecycle):
        """Test task start emission without callback."""
        execution = Mock()

        # Should not raise
        lifecycle.emit_task_start("task1", execution)

    def test_emit_task_start_callback_error(self, lifecycle, caplog):
        """Test task start callback error handling."""
        callback = Mock(side_effect=Exception("Callback failed"))
        lifecycle.on_task_start = callback
        execution = Mock()

        lifecycle.emit_task_start("task1", execution)

        assert "Error in task start callback" in caplog.text

    def test_emit_task_complete(self, lifecycle):
        """Test task complete event emission."""
        callback = Mock()
        lifecycle.on_task_complete = callback
        execution = Mock()

        lifecycle.emit_task_complete("task1", execution)

        callback.assert_called_once_with("task1", execution)

    def test_emit_task_complete_no_callback(self, lifecycle):
        """Test task complete emission without callback."""
        execution = Mock()

        # Should not raise
        lifecycle.emit_task_complete("task1", execution)

    def test_emit_task_complete_callback_error(self, lifecycle, caplog):
        """Test task complete callback error handling."""
        callback = Mock(side_effect=Exception("Callback failed"))
        lifecycle.on_task_complete = callback
        execution = Mock()

        lifecycle.emit_task_complete("task1", execution)

        assert "Error in task complete callback" in caplog.text

    def test_emit_task_fail(self, lifecycle):
        """Test task fail event emission."""
        callback = Mock()
        lifecycle.on_error = callback
        execution = Mock()
        execution.error = Exception("Task failed")

        lifecycle.emit_task_fail("task1", execution)

        callback.assert_called_once_with(execution.error)

    def test_emit_task_fail_no_error_callback(self, lifecycle):
        """Test task fail emission without error callback."""
        execution = Mock()
        execution.error = Exception("Task failed")

        # Should not raise
        lifecycle.emit_task_fail("task1", execution)

    def test_emit_task_fail_no_error_attribute(self, lifecycle):
        """Test task fail emission when execution has no error attribute."""
        execution = Mock(spec=[])  # No attributes

        # Should not raise
        lifecycle.emit_task_fail("task1", execution)

    def test_emit_task_fail_callback_error(self, lifecycle, caplog):
        """Test task fail callback error handling."""
        callback = Mock(side_effect=Exception("Callback failed"))
        lifecycle.on_error = callback
        execution = Mock()
        execution.error = Exception("Task failed")

        lifecycle.emit_task_fail("task1", execution)

        assert "Error in error callback" in caplog.text

    def test_emit_error(self, lifecycle):
        """Test error event emission."""
        callback = Mock()
        lifecycle.on_error = callback
        error = Exception("Test error")

        lifecycle.emit_error(error)

        callback.assert_called_once_with(error)

    def test_emit_error_no_callback(self, lifecycle):
        """Test error emission without callback."""
        error = Exception("Test error")

        # Should not raise
        lifecycle.emit_error(error)

    def test_emit_error_callback_error(self, lifecycle, caplog):
        """Test error callback error handling."""
        callback = Mock(side_effect=Exception("Callback failed"))
        lifecycle.on_error = callback
        error = Exception("Test error")

        lifecycle.emit_error(error)

        assert "Error in error callback" in caplog.text

    def test_get_workflow_callbacks(self, lifecycle):
        """Test getting workflow callbacks dictionary."""
        callbacks = lifecycle.get_workflow_callbacks()

        assert 'on_task_start' in callbacks
        assert 'on_task_complete' in callbacks
        assert 'on_task_fail' in callbacks
        assert 'on_workflow_complete' in callbacks

        # Verify they're callable
        assert callable(callbacks['on_task_start'])
        assert callable(callbacks['on_task_complete'])
        assert callable(callbacks['on_task_fail'])
        assert callable(callbacks['on_workflow_complete'])

    def test_workflow_callbacks_integration(self, lifecycle):
        """Test workflow callbacks can be used with workflow engine."""
        task_start_callback = Mock()
        task_complete_callback = Mock()
        error_callback = Mock()
        workflow_complete_callback = Mock()

        lifecycle.register_task_start_callback(task_start_callback)
        lifecycle.register_task_complete_callback(task_complete_callback)
        lifecycle.register_error_callback(error_callback)
        lifecycle.register_workflow_complete_callback(workflow_complete_callback)

        callbacks = lifecycle.get_workflow_callbacks()

        # Simulate workflow engine calling callbacks
        execution = Mock()
        metrics = WorkflowMetrics(
            total_tasks=1,
            completed_tasks=1,
            failed_tasks=0,
            total_duration=5.0
        )

        callbacks['on_task_start']("task1", execution)
        callbacks['on_task_complete']("task1", execution)
        callbacks['on_workflow_complete'](metrics)

        task_start_callback.assert_called_once_with("task1", execution)
        task_complete_callback.assert_called_once_with("task1", execution)
        workflow_complete_callback.assert_called_once_with(metrics)

    def test_reset(self, lifecycle):
        """Test reset clears all callbacks."""
        lifecycle.on_workflow_start = Mock()
        lifecycle.on_workflow_complete = Mock()
        lifecycle.on_task_start = Mock()
        lifecycle.on_task_complete = Mock()
        lifecycle.on_error = Mock()

        lifecycle.reset()

        assert lifecycle.on_workflow_start is None
        assert lifecycle.on_workflow_complete is None
        assert lifecycle.on_task_start is None
        assert lifecycle.on_task_complete is None
        assert lifecycle.on_error is None

    def test_multiple_callback_registrations(self, lifecycle):
        """Test that callbacks can be registered multiple times (replacement)."""
        callback1 = Mock()
        callback2 = Mock()

        lifecycle.register_workflow_start_callback(callback1)
        assert lifecycle.on_workflow_start == callback1

        lifecycle.register_workflow_start_callback(callback2)
        assert lifecycle.on_workflow_start == callback2

        lifecycle.emit_workflow_start()

        # Only callback2 should be called
        callback1.assert_not_called()
        callback2.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
