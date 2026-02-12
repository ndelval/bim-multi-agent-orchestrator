"""
Tests for orchestrator.core.error_handler module.

Covers ErrorCategory, ErrorContext, RetryPolicy, ErrorResolution,
error categorization, recovery hints, and logging behavior.
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.core.error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorResolution,
    RetryPolicy,
)


# ---------------------------------------------------------------------------
# ErrorCategory tests
# ---------------------------------------------------------------------------


class TestErrorCategory:
    """Test ErrorCategory enum values."""

    def test_all_categories_defined(self):
        expected = {
            "configuration",
            "network",
            "validation",
            "execution",
            "resource",
            "unknown",
        }
        actual = {cat.value for cat in ErrorCategory}
        assert actual == expected

    def test_category_values(self):
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# ErrorContext tests
# ---------------------------------------------------------------------------


class TestErrorContext:
    """Test ErrorContext frozen dataclass."""

    def test_basic_creation(self):
        exc = ValueError("test error")
        ctx = ErrorContext(exception=exc, operation="test_op", component="test_comp")
        assert ctx.exception is exc
        assert ctx.operation == "test_op"
        assert ctx.component == "test_comp"
        assert ctx.context_data == {}
        assert ctx.retry_count == 0
        assert isinstance(ctx.timestamp, datetime)

    def test_error_message_property(self):
        ctx = ErrorContext(
            exception=RuntimeError("something broke"),
            operation="op",
            component="comp",
        )
        assert ctx.error_message == "something broke"

    def test_exception_type_property(self):
        ctx = ErrorContext(
            exception=TypeError("bad type"),
            operation="op",
            component="comp",
        )
        assert ctx.exception_type == "TypeError"

    def test_frozen(self):
        ctx = ErrorContext(exception=ValueError("x"), operation="op", component="comp")
        with pytest.raises(AttributeError):
            ctx.operation = "mutated"


# ---------------------------------------------------------------------------
# RetryPolicy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """Test RetryPolicy frozen dataclass."""

    def test_creation(self):
        policy = RetryPolicy(
            should_retry=True,
            max_retries=3,
            backoff_seconds=1.0,
            recovery_hint="retry later",
        )
        assert policy.should_retry is True
        assert policy.max_retries == 3
        assert policy.backoff_seconds == 1.0
        assert policy.recovery_hint == "retry later"

    def test_frozen(self):
        policy = RetryPolicy(True, 1, 0.5, "hint")
        with pytest.raises(AttributeError):
            policy.should_retry = False


# ---------------------------------------------------------------------------
# ErrorResolution tests
# ---------------------------------------------------------------------------


class TestErrorResolution:
    """Test ErrorResolution frozen dataclass."""

    def test_creation(self):
        res = ErrorResolution(
            should_retry=False,
            max_retries=0,
            retry_count=0,
            recovery_hint="fix config",
            category=ErrorCategory.CONFIGURATION,
        )
        assert res.should_retry is False
        assert res.category == ErrorCategory.CONFIGURATION

    def test_frozen(self):
        res = ErrorResolution(True, 2, 0, "hint", ErrorCategory.NETWORK)
        with pytest.raises(AttributeError):
            res.category = ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# ErrorHandler.categorize_error tests
# ---------------------------------------------------------------------------


class TestCategorizeError:
    """Test error categorization logic."""

    @pytest.fixture
    def handler(self):
        return ErrorHandler()

    # --- By exception type hierarchy ---

    def test_configuration_error(self, handler):
        from orchestrator.core.exceptions import ConfigurationError

        assert (
            handler.categorize_error(ConfigurationError("bad cfg"))
            == ErrorCategory.CONFIGURATION
        )

    def test_validation_error(self, handler):
        from orchestrator.core.exceptions import ValidationError

        assert (
            handler.categorize_error(ValidationError("invalid"))
            == ErrorCategory.CONFIGURATION
        )

    def test_connection_error(self, handler):
        assert (
            handler.categorize_error(ConnectionError("refused"))
            == ErrorCategory.NETWORK
        )

    def test_timeout_error(self, handler):
        assert (
            handler.categorize_error(TimeoutError("timed out")) == ErrorCategory.NETWORK
        )

    def test_value_error(self, handler):
        assert (
            handler.categorize_error(ValueError("bad value"))
            == ErrorCategory.VALIDATION
        )

    def test_type_error(self, handler):
        assert (
            handler.categorize_error(TypeError("wrong type"))
            == ErrorCategory.VALIDATION
        )

    def test_agent_creation_error(self, handler):
        from orchestrator.core.exceptions import AgentCreationError

        assert (
            handler.categorize_error(AgentCreationError("no agent"))
            == ErrorCategory.EXECUTION
        )

    def test_task_execution_error(self, handler):
        from orchestrator.core.exceptions import TaskExecutionError

        assert (
            handler.categorize_error(TaskExecutionError("failed"))
            == ErrorCategory.EXECUTION
        )

    def test_workflow_error(self, handler):
        from orchestrator.core.exceptions import WorkflowError

        assert (
            handler.categorize_error(WorkflowError("workflow died"))
            == ErrorCategory.EXECUTION
        )

    def test_memory_error(self, handler):
        from orchestrator.core.exceptions import MemoryError

        assert handler.categorize_error(MemoryError("oom")) == ErrorCategory.RESOURCE

    def test_provider_error(self, handler):
        from orchestrator.core.exceptions import ProviderError

        assert (
            handler.categorize_error(ProviderError("provider down"))
            == ErrorCategory.RESOURCE
        )

    # --- By message pattern matching ---

    def test_network_pattern_timeout(self, handler):
        assert (
            handler.categorize_error(RuntimeError("request timeout"))
            == ErrorCategory.NETWORK
        )

    def test_network_pattern_rate_limit(self, handler):
        assert (
            handler.categorize_error(RuntimeError("rate limit exceeded"))
            == ErrorCategory.NETWORK
        )

    def test_validation_pattern(self, handler):
        assert (
            handler.categorize_error(RuntimeError("invalid JSON format"))
            == ErrorCategory.VALIDATION
        )

    def test_config_pattern(self, handler):
        assert (
            handler.categorize_error(RuntimeError("missing configuration key"))
            == ErrorCategory.CONFIGURATION
        )

    def test_resource_pattern(self, handler):
        assert (
            handler.categorize_error(RuntimeError("memory quota exceeded"))
            == ErrorCategory.RESOURCE
        )

    # --- Custom category attribute ---

    def test_custom_category_attribute(self, handler):
        exc = RuntimeError("custom")
        exc.category = ErrorCategory.RESOURCE
        assert handler.categorize_error(exc) == ErrorCategory.RESOURCE

    # --- Default to UNKNOWN ---

    def test_unknown_fallback(self, handler):
        assert handler.categorize_error(RuntimeError("xyzzy")) == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# ErrorHandler.handle_error tests
# ---------------------------------------------------------------------------


class TestHandleError:
    """Test the main handle_error entry point."""

    @pytest.fixture
    def handler(self):
        return ErrorHandler()

    def test_returns_error_resolution(self, handler):
        res = handler.handle_error(
            exception=ValueError("bad"),
            operation="parse_input",
            component="router",
        )
        assert isinstance(res, ErrorResolution)

    def test_configuration_not_retried(self, handler):
        from orchestrator.core.exceptions import ConfigurationError

        res = handler.handle_error(
            exception=ConfigurationError("bad"),
            operation="init",
            component="orchestrator",
        )
        assert res.should_retry is False
        assert res.max_retries == 0
        assert res.category == ErrorCategory.CONFIGURATION

    def test_network_retried(self, handler):
        res = handler.handle_error(
            exception=ConnectionError("refused"),
            operation="llm_call",
            component="agent",
        )
        assert res.should_retry is True
        assert res.max_retries == 3
        assert res.category == ErrorCategory.NETWORK

    def test_retry_count_respected(self, handler):
        res = handler.handle_error(
            exception=ConnectionError("refused"),
            operation="llm_call",
            component="agent",
            retry_count=3,  # Already at max
        )
        assert res.should_retry is False  # No more retries

    def test_resource_not_retried(self, handler):
        from orchestrator.core.exceptions import MemoryError

        res = handler.handle_error(
            exception=MemoryError("oom"),
            operation="store",
            component="memory",
        )
        assert res.should_retry is False
        assert res.category == ErrorCategory.RESOURCE

    def test_context_data_passed(self, handler):
        """Ensure context_data is accepted without error."""
        res = handler.handle_error(
            exception=RuntimeError("oops"),
            operation="test",
            component="comp",
            context_data={"agent": "Researcher", "attempt": 1},
        )
        assert isinstance(res, ErrorResolution)


# ---------------------------------------------------------------------------
# ErrorHandler.get_recovery_hint tests
# ---------------------------------------------------------------------------


class TestGetRecoveryHint:
    """Test recovery hint generation."""

    @pytest.fixture
    def handler(self):
        return ErrorHandler()

    def test_custom_hint_from_exception(self, handler):
        exc = RuntimeError("oops")
        exc.recovery_hint = "Try turning it off and on again"
        hint = handler.get_recovery_hint(exc, ErrorCategory.UNKNOWN)
        assert hint == "Try turning it off and on again"

    def test_category_hint_used(self, handler):
        hint = handler.get_recovery_hint(RuntimeError("oops"), ErrorCategory.NETWORK)
        assert (
            "network" in hint.lower()
            or "retry" in hint.lower()
            or "api" in hint.lower()
        )

    def test_fallback_hint(self, handler):
        # Force a category with no policy by using a fresh handler with empty policies
        handler._retry_policies = {}
        hint = handler.get_recovery_hint(RuntimeError("boom"), ErrorCategory.UNKNOWN)
        assert "error occurred" in hint.lower()


# ---------------------------------------------------------------------------
# ErrorHandler._log_error tests
# ---------------------------------------------------------------------------


class TestLogError:
    """Test logging behavior."""

    def test_config_error_logs_at_error_level(self):
        mock_logger = MagicMock(spec=logging.Logger)
        handler = ErrorHandler(logger_instance=mock_logger)

        handler.handle_error(
            exception=RuntimeError("missing configuration"),
            operation="init",
            component="orchestrator",
        )

        mock_logger.error.assert_called()

    def test_network_retry_logs_at_warning_level(self):
        mock_logger = MagicMock(spec=logging.Logger)
        handler = ErrorHandler(logger_instance=mock_logger)

        handler.handle_error(
            exception=ConnectionError("refused"),
            operation="llm_call",
            component="agent",
            retry_count=0,  # Will retry
        )

        mock_logger.warning.assert_called()

    def test_exhausted_retries_logs_at_error_level(self):
        mock_logger = MagicMock(spec=logging.Logger)
        handler = ErrorHandler(logger_instance=mock_logger)

        handler.handle_error(
            exception=ConnectionError("refused"),
            operation="llm_call",
            component="agent",
            retry_count=3,  # Exhausted
        )

        mock_logger.error.assert_called()


# ---------------------------------------------------------------------------
# Retry policies initialization sanity check
# ---------------------------------------------------------------------------


class TestRetryPoliciesInit:
    """Verify default retry policies are configured properly."""

    @pytest.fixture
    def handler(self):
        return ErrorHandler()

    def test_all_categories_have_policies(self, handler):
        for cat in ErrorCategory:
            assert cat in handler._retry_policies

    def test_config_not_retryable(self, handler):
        assert (
            handler._retry_policies[ErrorCategory.CONFIGURATION].should_retry is False
        )

    def test_network_retryable(self, handler):
        policy = handler._retry_policies[ErrorCategory.NETWORK]
        assert policy.should_retry is True
        assert policy.max_retries == 3

    def test_resource_not_retryable(self, handler):
        assert handler._retry_policies[ErrorCategory.RESOURCE].should_retry is False

    def test_execution_retryable(self, handler):
        policy = handler._retry_policies[ErrorCategory.EXECUTION]
        assert policy.should_retry is True
        assert policy.max_retries == 2
