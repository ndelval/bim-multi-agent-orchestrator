"""
Orchestrator Package - Scalable Agent Orchestration System

A class-based architecture for building and managing multi-agent orchestrator systems
with support for parallel execution, conditional routing, and multiple memory providers.
"""

from .core.orchestrator import Orchestrator
from .core.config import OrchestratorConfig
from .core.exceptions import (
    OrchestratorError,
    ConfigurationError,
    AgentCreationError,
    TaskExecutionError,
    MemoryError,
    WorkflowError
)

__version__ = "1.0.0"
__author__ = "Orchestrator Team"

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorError",
    "ConfigurationError", 
    "AgentCreationError",
    "TaskExecutionError",
    "MemoryError",
    "WorkflowError"
]