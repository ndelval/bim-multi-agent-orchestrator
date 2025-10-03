"""
Configuration management for the orchestrator system.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import ConfigurationError, ValidationError


class MemoryProvider(str, Enum):
    """Enum for supported memory providers."""
    HYBRID = "hybrid"
    RAG = "rag"
    MEM0 = "mem0"


class ProcessType(str, Enum):
    """Enum for process execution types."""
    WORKFLOW = "workflow"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"


@dataclass
class EmbedderConfig:
    """Configuration for embedding models."""
    provider: str = "openai"
    config: Dict[str, Any] = field(default_factory=lambda: {"model": "text-embedding-3-large"})


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    provider: MemoryProvider = MemoryProvider.HYBRID
    use_embedding: bool = True
    embedder: Optional[EmbedderConfig] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Provider-specific paths
    hybrid_vector_path: Optional[str] = None
    hybrid_lexical_db_path: Optional[str] = None
    rag_db_path: Optional[str] = None

    def __post_init__(self):
        """Initialize default paths and embedder."""
        if self.embedder is None:
            self.embedder = EmbedderConfig()

        if self.provider == MemoryProvider.HYBRID:
            if not self.hybrid_vector_path:
                self.hybrid_vector_path = ".praison/hybrid_chroma"
            if not self.hybrid_lexical_db_path:
                self.hybrid_lexical_db_path = ".praison/hybrid_lexical.db"
        elif self.provider == MemoryProvider.RAG:
            if not self.rag_db_path:
                self.rag_db_path = ".praison/memory/chroma_db"


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    name: str
    role: str
    goal: str
    backstory: str
    instructions: str
    tools: Union[List[str], List[Callable]] = field(default_factory=list)  # PRIORITY 3 FIX: Support both strings and callables
    enabled: bool = True
    llm: Optional[str] = None  # LLM model identifier (e.g., "gpt-4o-mini")


@dataclass
class TaskConfig:
    """Configuration for individual tasks."""
    name: str
    description: str
    expected_output: str
    agent: str
    context: List[str] = field(default_factory=list)
    enabled: bool = True
    # Workflow engine fields
    agent_name: Optional[str] = None  # Agent name (alias for agent)
    async_execution: bool = False  # Execute task asynchronously
    is_start: bool = False  # Mark as workflow start task
    next_tasks: List[str] = field(default_factory=list)  # List of dependent tasks
    task_type: str = "standard"  # Task type (standard, decision, etc.)
    condition: Optional[Dict[str, List[str]]] = None  # Conditional routing for decision tasks

    def __post_init__(self):
        """Initialize agent_name from agent if not provided."""
        if self.agent_name is None:
            self.agent_name = self.agent


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration."""
    name: str
    process: str = "sequential"
    agents: List[AgentConfig] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    memory: Optional[MemoryConfig] = None
    verbose: int = 1  # Verbosity level (0=quiet, 1=normal, 2=debug)
    max_iter: int = 25  # Maximum iterations for agent execution
    user_id: str = "default_user"  # User ID for session tracking
    run_id: Optional[str] = None  # Run ID for execution tracking
    async_execution: bool = False  # Enable asynchronous execution

    def __post_init__(self):
        """Initialize default memory config if not provided."""
        if self.memory is None:
            self.memory = MemoryConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'OrchestratorConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                raise ConfigurationError("Empty configuration file")

            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrchestratorConfig':
        """Create configuration from dictionary."""
        try:
            # Parse agents
            agents = []
            for agent_data in data.get('agents', []):
                agents.append(AgentConfig(**agent_data))

            # Parse tasks
            tasks = []
            for task_data in data.get('tasks', []):
                tasks.append(TaskConfig(**task_data))

            # Parse memory config
            memory = None
            if 'memory' in data:
                memory_data = data['memory']
                provider = MemoryProvider(memory_data.get('provider', 'hybrid'))
                embedder = None
                if 'embedder' in memory_data:
                    embedder = EmbedderConfig(**memory_data['embedder'])

                memory = MemoryConfig(
                    provider=provider,
                    use_embedding=memory_data.get('use_embedding', True),
                    embedder=embedder,
                    config=memory_data.get('config', {})
                )

            return cls(
                name=data.get('name', 'orchestrator'),
                process=data.get('process', 'sequential'),
                agents=agents,
                tasks=tasks,
                memory=memory,
                verbose=data.get('verbose', False)
            )
        except Exception as e:
            raise ConfigurationError(f"Error parsing configuration: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'process': self.process,
            'agents': [vars(agent) for agent in self.agents],
            'tasks': [vars(task) for task in self.tasks],
            'memory': vars(self.memory) if self.memory else None,
            'verbose': self.verbose
        }

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        # Validate agents
        if not self.agents:
            errors.append("At least one agent must be defined")

        agent_names = {agent.name for agent in self.agents}

        # Validate tasks
        if not self.tasks:
            errors.append("At least one task must be defined")

        for task in self.tasks:
            if task.agent not in agent_names:
                errors.append(f"Task '{task.name}' references unknown agent '{task.agent}'")

            for context_task in task.context:
                if context_task not in {t.name for t in self.tasks}:
                    errors.append(f"Task '{task.name}' references unknown context task '{context_task}'")

        if errors:
            raise ValidationError("Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def load_config(config_path: Union[str, Path]) -> OrchestratorConfig:
    """Load orchestrator configuration from file."""
    return OrchestratorConfig.from_yaml(config_path)


def save_config(config: OrchestratorConfig, config_path: Union[str, Path]) -> None:
    """Save orchestrator configuration to file."""
    config_path = Path(config_path)

    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
