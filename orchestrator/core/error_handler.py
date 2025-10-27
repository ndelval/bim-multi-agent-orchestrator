"""
Centralized error handling with categorization and retry logic.

This module provides a unified error handling system that categorizes errors,
suggests recovery actions, and determines retry eligibility.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    CONFIGURATION = "configuration"  # Config validation errors
    NETWORK = "network"  # API calls, LLM timeouts, connectivity
    VALIDATION = "validation"  # Data validation failures
    EXECUTION = "execution"  # Agent/workflow execution errors
    RESOURCE = "resource"  # Memory, quota, system limits
    UNKNOWN = "unknown"  # Uncategorized errors


@dataclass(frozen=True)
class ErrorContext:
    """
    Context information for error handling.

    Captures all relevant information about an error occurrence
    for categorization, logging, and recovery decisions.

    Attributes:
        exception: The original exception that occurred
        operation: Description of the operation that failed
        component: Component where error occurred (router, agent, workflow)
        context_data: Additional context information
        timestamp: When the error occurred
        retry_count: Number of retry attempts made
    """
    exception: Exception
    operation: str
    component: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

    @property
    def error_message(self) -> str:
        """Get error message from exception."""
        return str(self.exception)

    @property
    def exception_type(self) -> str:
        """Get exception type name."""
        return type(self.exception).__name__


@dataclass(frozen=True)
class RetryPolicy:
    """
    Retry policy for specific error categories.

    Attributes:
        should_retry: Whether errors in this category are retryable
        max_retries: Maximum number of retry attempts
        backoff_seconds: Base backoff time (use exponential backoff)
        recovery_hint: Default recovery suggestion
    """
    should_retry: bool
    max_retries: int
    backoff_seconds: float
    recovery_hint: str


@dataclass(frozen=True)
class ErrorResolution:
    """
    Resolution strategy for handling an error.

    Returned by ErrorHandler.handle_error() to indicate how to
    proceed after an error occurs.

    Attributes:
        should_retry: Whether the operation should be retried
        max_retries: Maximum number of retry attempts allowed
        retry_count: Current retry count
        recovery_hint: User-friendly recovery suggestion
        category: Error category for this resolution
    """
    should_retry: bool
    max_retries: int
    retry_count: int
    recovery_hint: str
    category: ErrorCategory


class ErrorHandler:
    """
    Centralized error handler with categorization and retry logic.

    Provides unified error handling across the orchestrator system with:
    - Automatic error categorization
    - Retry policy determination
    - Recovery hint generation
    - Structured logging

    Example:
        ```python
        error_handler = ErrorHandler()

        try:
            # operation
        except Exception as e:
            resolution = error_handler.handle_error(
                exception=e,
                operation="router_execution",
                component="router"
            )
            if resolution.should_retry and retry_count < resolution.max_retries:
                # retry operation
            else:
                print(resolution.recovery_hint)
        ```
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize error handler.

        Args:
            logger_instance: Optional logger to use (defaults to module logger)
        """
        self.logger = logger_instance or logger
        self._retry_policies = self._init_retry_policies()

    def _init_retry_policies(self) -> Dict[ErrorCategory, RetryPolicy]:
        """Initialize retry policies for each error category."""
        return {
            ErrorCategory.CONFIGURATION: RetryPolicy(
                should_retry=False,
                max_retries=0,
                backoff_seconds=0.0,
                recovery_hint="Review configuration file and fix invalid settings"
            ),
            ErrorCategory.NETWORK: RetryPolicy(
                should_retry=True,
                max_retries=3,
                backoff_seconds=1.0,  # Exponential: 1s, 2s, 4s
                recovery_hint="Check network connection and API availability. Retrying..."
            ),
            ErrorCategory.VALIDATION: RetryPolicy(
                should_retry=True,
                max_retries=1,  # One retry in case of transient LLM output issue
                backoff_seconds=0.5,
                recovery_hint="Validate input data format and schema"
            ),
            ErrorCategory.EXECUTION: RetryPolicy(
                should_retry=True,
                max_retries=2,
                backoff_seconds=1.0,
                recovery_hint="Check agent configuration and task definition. Retrying..."
            ),
            ErrorCategory.RESOURCE: RetryPolicy(
                should_retry=False,
                max_retries=0,
                backoff_seconds=0.0,
                recovery_hint="Reduce batch size or upgrade system resources"
            ),
            ErrorCategory.UNKNOWN: RetryPolicy(
                should_retry=True,
                max_retries=1,
                backoff_seconds=1.0,
                recovery_hint="An unexpected error occurred. Check logs for details"
            ),
        }

    def handle_error(
        self,
        exception: Exception,
        operation: str,
        component: str,
        context_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> ErrorResolution:
        """
        Handle an error with categorization and retry determination.

        Main entry point for error handling. Categorizes the error,
        determines retry policy, logs appropriately, and returns
        a resolution strategy.

        Args:
            exception: The exception that occurred
            operation: Description of the operation that failed
            component: Component where error occurred
            context_data: Additional context information
            retry_count: Current retry attempt number

        Returns:
            ErrorResolution with retry decision and recovery hint
        """
        # Create error context
        error_context = ErrorContext(
            exception=exception,
            operation=operation,
            component=component,
            context_data=context_data or {},
            retry_count=retry_count
        )

        # Categorize error
        category = self.categorize_error(exception)

        # Get retry policy for this category
        policy = self._retry_policies[category]

        # Get recovery hint
        recovery_hint = self.get_recovery_hint(exception, category)

        # Determine if should retry
        should_retry = (
            policy.should_retry and
            retry_count < policy.max_retries
        )

        # Log error with appropriate level
        self._log_error(error_context, category, should_retry)

        return ErrorResolution(
            should_retry=should_retry,
            max_retries=policy.max_retries,
            retry_count=retry_count,
            recovery_hint=recovery_hint,
            category=category
        )

    def categorize_error(self, exception: Exception) -> ErrorCategory:
        """
        Categorize exception based on type and message patterns.

        Categorization strategy:
        1. Check for custom category attribute on exception
        2. Check exception type hierarchy
        3. Check error message patterns
        4. Default to UNKNOWN

        Args:
            exception: Exception to categorize

        Returns:
            ErrorCategory classification
        """
        # Check custom category attribute if present
        if hasattr(exception, 'category') and exception.category:
            return exception.category

        # Import exceptions for type checking
        from .exceptions import (
            ConfigurationError,
            ValidationError,
            AgentCreationError,
            TaskExecutionError,
            WorkflowError,
            MemoryError,
            ProviderError,
        )

        # Check exception type hierarchy
        if isinstance(exception, (ConfigurationError, ValidationError)):
            return ErrorCategory.CONFIGURATION

        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK

        if isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION

        if isinstance(exception, (AgentCreationError, TaskExecutionError, WorkflowError)):
            return ErrorCategory.EXECUTION

        if isinstance(exception, (MemoryError, ProviderError)):
            return ErrorCategory.RESOURCE

        # Check error message patterns
        message = str(exception).lower()

        network_patterns = ["timeout", "connection", "network", "api", "rate limit", "unavailable"]
        if any(pattern in message for pattern in network_patterns):
            return ErrorCategory.NETWORK

        validation_patterns = ["invalid", "validation", "format", "schema", "parse", "decode"]
        if any(pattern in message for pattern in validation_patterns):
            return ErrorCategory.VALIDATION

        config_patterns = ["configuration", "config", "setting", "missing"]
        if any(pattern in message for pattern in config_patterns):
            return ErrorCategory.CONFIGURATION

        resource_patterns = ["memory", "quota", "limit", "resource", "capacity"]
        if any(pattern in message for pattern in resource_patterns):
            return ErrorCategory.RESOURCE

        # Default to UNKNOWN
        return ErrorCategory.UNKNOWN

    def get_recovery_hint(
        self,
        exception: Exception,
        category: ErrorCategory
    ) -> str:
        """
        Get user-friendly recovery suggestion for an error.

        Priority:
        1. Exception-specific recovery_hint attribute
        2. Category-specific default hint from retry policy
        3. Generic fallback hint

        Args:
            exception: The exception that occurred
            category: Error category

        Returns:
            User-friendly recovery suggestion
        """
        # Check if exception has custom recovery hint
        if hasattr(exception, 'recovery_hint') and exception.recovery_hint:
            return exception.recovery_hint

        # Get category-specific hint from retry policy
        policy = self._retry_policies.get(category)
        if policy:
            return policy.recovery_hint

        # Fallback generic hint
        return f"An error occurred: {str(exception)}. Check logs for details"

    def _log_error(
        self,
        error_context: ErrorContext,
        category: ErrorCategory,
        should_retry: bool
    ) -> None:
        """
        Log error with appropriate level and context.

        Logging strategy:
        - CONFIGURATION/RESOURCE errors: ERROR level (user must fix)
        - NETWORK errors with retry: WARNING level (temporary)
        - EXECUTION errors with retry: WARNING level (may resolve)
        - UNKNOWN errors: ERROR level (unexpected)
        - All retries include retry count

        Args:
            error_context: Error context with all details
            category: Error category
            should_retry: Whether operation will be retried
        """
        log_message = (
            f"{category.value.upper()} error in {error_context.component} "
            f"during {error_context.operation}: {error_context.error_message}"
        )

        if should_retry:
            log_message += f" (retry {error_context.retry_count + 1})"

        # Determine log level based on category and retry status
        if category in (ErrorCategory.CONFIGURATION, ErrorCategory.RESOURCE, ErrorCategory.UNKNOWN):
            # Serious errors that need user attention
            self.logger.error(log_message, exc_info=error_context.exception)
        elif should_retry:
            # Transient errors being retried
            self.logger.warning(log_message)
        else:
            # Non-retryable execution/network/validation errors
            self.logger.error(log_message, exc_info=error_context.exception)


__all__ = [
    "ErrorCategory",
    "ErrorContext",
    "ErrorResolution",
    "RetryPolicy",
    "ErrorHandler",
]
