"""
Configuration management for the orchestrator system.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import ConfigurationError, ValidationError


class MemoryProvider(str, Enum):
    """Supported memory providers."""
    RAG = "rag"
    MONGODB = "mongodb"
    MEM0 = "mem0"
    HYBRID = "hybrid"


class ProcessType(str, Enum):
    """Supported process types."""
    WORKFLOW = "workflow"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


@dataclass
class EmbedderConfig:
    """Configuration for embedding models."""
    provider: str = "openai"
    config: Dict[str, Any] = field(default_factory=lambda: {"model": "text-embedding-3-large"})


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    provider: MemoryProvider = MemoryProvider.RAG
    use_embedding: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    short_db: Optional[str] = None
    long_db: Optional[str] = None
    rag_db_path: Optional[str] = None
    embedder: Optional[EmbedderConfig] = None

    def __post_init__(self):
        """Set default paths if not provided."""
        if self.embedder is None:
            self.embedder = EmbedderConfig()
        
        if self.provider == MemoryProvider.RAG:
            if not self.short_db:
                self.short_db = ".praison/memory/short.db"
            if not self.long_db:
                self.long_db = ".praison/memory/long.db"
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
    tools: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class TaskConfig:
    """Configuration for individual tasks."""
    name: str
    description: str
    expected_output: str
    agent_name: str
    async_execution: bool = False
    is_start: bool = False
    context: List[str] = field(default_factory=list)
    next_tasks: List[str] = field(default_factory=list)
    task_type: str = "normal"
    condition: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionConfig:
    """Configuration for execution parameters."""
    process: ProcessType = ProcessType.WORKFLOW
    verbose: int = 1
    max_iter: int = 8
    memory: bool = True
    user_id: str = "default-user"
    async_execution: bool = True


@dataclass
class OrchestratorConfig:
    """Main configuration class for the orchestrator system."""
    name: str = "DefaultOrchestrator"
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    agents: List[AgentConfig] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    embedder: Optional[EmbedderConfig] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.embedder is None:
            self.embedder = EmbedderConfig()
        self.validate()

    def validate(self) -> None:
        """Validate the configuration."""
        try:
            # Validate agent names are unique
            agent_names = [agent.name for agent in self.agents]
            if len(agent_names) != len(set(agent_names)):
                raise ValidationError("Agent names must be unique")
            
            # Validate task names are unique
            task_names = [task.name for task in self.tasks]
            if len(task_names) != len(set(task_names)):
                raise ValidationError("Task names must be unique")
            
            # Validate task references
            for task in self.tasks:
                # Check agent exists
                if task.agent_name not in agent_names:
                    raise ValidationError(f"Task '{task.name}' references non-existent agent '{task.agent_name}'")
                
                # Check context tasks exist
                for context_task in task.context:
                    if context_task not in task_names:
                        raise ValidationError(f"Task '{task.name}' references non-existent context task '{context_task}'")
                
                # Check next tasks exist
                for next_task in task.next_tasks:
                    if next_task not in task_names:
                        raise ValidationError(f"Task '{task.name}' references non-existent next task '{next_task}'")
            
            # Validate memory configuration
            self._validate_memory_config()
            
        except Exception as e:
            raise ValidationError(f"Configuration validation failed: {str(e)}")

    def _validate_memory_config(self) -> None:
        """Validate memory configuration."""
        if self.memory_config.provider == MemoryProvider.MONGODB:
            if not self.memory_config.config.get("connection_string"):
                raise ValidationError("MongoDB provider requires connection_string in config")
        
        elif self.memory_config.provider == MemoryProvider.MEM0:
            if not self.memory_config.config.get("graph_store"):
                raise ValidationError("Mem0 provider requires graph_store configuration")

        elif self.memory_config.provider == MemoryProvider.HYBRID:
            cfg = self.memory_config.config or {}
            if not cfg.get("vector_store"):
                raise ValidationError("Hybrid provider requires vector_store configuration")
            if not cfg.get("lexical"):
                raise ValidationError("Hybrid provider requires lexical configuration")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OrchestratorConfig":
        """Create configuration from dictionary."""
        try:
            # Convert memory config
            memory_config_dict = config_dict.get("memory_config", {})
            memory_config = MemoryConfig(
                provider=MemoryProvider(memory_config_dict.get("provider", "rag")),
                use_embedding=memory_config_dict.get("use_embedding", True),
                config=memory_config_dict.get("config", {}),
                short_db=memory_config_dict.get("short_db"),
                long_db=memory_config_dict.get("long_db"),
                rag_db_path=memory_config_dict.get("rag_db_path"),
                embedder=EmbedderConfig(**memory_config_dict.get("embedder", {}))
            )

            # Convert execution config
            exec_config_dict = config_dict.get("execution_config", {})
            execution_config = ExecutionConfig(
                process=ProcessType(exec_config_dict.get("process", "workflow")),
                verbose=exec_config_dict.get("verbose", 1),
                max_iter=exec_config_dict.get("max_iter", 8),
                memory=exec_config_dict.get("memory", True),
                user_id=exec_config_dict.get("user_id", "default-user"),
                async_execution=exec_config_dict.get("async_execution", True)
            )

            # Convert agents
            agents = [AgentConfig(**agent_dict) for agent_dict in config_dict.get("agents", [])]

            # Convert tasks
            tasks = [TaskConfig(**task_dict) for task_dict in config_dict.get("tasks", [])]

            # Convert embedder
            embedder_dict = config_dict.get("embedder", {})
            embedder = EmbedderConfig(**embedder_dict) if embedder_dict else None

            return cls(
                name=config_dict.get("name", "DefaultOrchestrator"),
                memory_config=memory_config,
                execution_config=execution_config,
                agents=agents,
                tasks=tasks,
                embedder=embedder,
                custom_config=config_dict.get("custom_config", {})
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration from dict: {str(e)}")

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "OrchestratorConfig":
        """Load configuration from file (YAML or JSON)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_path.suffix}")
            
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {str(e)}")

    @classmethod
    def from_env(cls, prefix: str = "ORCHESTRATOR_") -> "OrchestratorConfig":
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Basic configuration
        if os.getenv(f"{prefix}NAME"):
            config_dict["name"] = os.getenv(f"{prefix}NAME")
        
        # Memory configuration
        memory_config = {}
        if os.getenv(f"{prefix}MEMORY_PROVIDER"):
            memory_config["provider"] = os.getenv(f"{prefix}MEMORY_PROVIDER")
        if os.getenv(f"{prefix}MEMORY_USE_EMBEDDING"):
            memory_config["use_embedding"] = os.getenv(f"{prefix}MEMORY_USE_EMBEDDING").lower() == "true"
        
        if memory_config:
            config_dict["memory_config"] = memory_config
        
        # Execution configuration
        exec_config = {}
        if os.getenv(f"{prefix}PROCESS"):
            exec_config["process"] = os.getenv(f"{prefix}PROCESS")
        if os.getenv(f"{prefix}VERBOSE"):
            exec_config["verbose"] = int(os.getenv(f"{prefix}VERBOSE"))
        if os.getenv(f"{prefix}MAX_ITER"):
            exec_config["max_iter"] = int(os.getenv(f"{prefix}MAX_ITER"))
        if os.getenv(f"{prefix}USER_ID"):
            exec_config["user_id"] = os.getenv(f"{prefix}USER_ID")
        
        if exec_config:
            config_dict["execution_config"] = exec_config
        
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "memory_config": {
                "provider": self.memory_config.provider.value,
                "use_embedding": self.memory_config.use_embedding,
                "config": self.memory_config.config,
                "short_db": self.memory_config.short_db,
                "long_db": self.memory_config.long_db,
                "rag_db_path": self.memory_config.rag_db_path,
                "embedder": {
                    "provider": self.memory_config.embedder.provider,
                    "config": self.memory_config.embedder.config
                }
            },
            "execution_config": {
                "process": self.execution_config.process.value,
                "verbose": self.execution_config.verbose,
                "max_iter": self.execution_config.max_iter,
                "memory": self.execution_config.memory,
                "user_id": self.execution_config.user_id,
                "async_execution": self.execution_config.async_execution
            },
            "agents": [
                {
                    "name": agent.name,
                    "role": agent.role,
                    "goal": agent.goal,
                    "backstory": agent.backstory,
                    "instructions": agent.instructions,
                    "tools": agent.tools,
                    "enabled": agent.enabled
                }
                for agent in self.agents
            ],
            "tasks": [
                {
                    "name": task.name,
                    "description": task.description,
                    "expected_output": task.expected_output,
                    "agent_name": task.agent_name,
                    "async_execution": task.async_execution,
                    "is_start": task.is_start,
                    "context": task.context,
                    "next_tasks": task.next_tasks,
                    "task_type": task.task_type,
                    "condition": task.condition
                }
                for task in self.tasks
            ],
            "embedder": {
                "provider": self.embedder.provider,
                "config": self.embedder.config
            } if self.embedder else None,
            "custom_config": self.custom_config
        }

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {str(e)}")

    def merge(self, other: "OrchestratorConfig") -> "OrchestratorConfig":
        """Merge this configuration with another, other takes precedence."""
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(base: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(merged_dict, other_dict)
        return self.from_dict(merged_dict)

    def get_agent_by_name(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_task_by_name(self, name: str) -> Optional[TaskConfig]:
        """Get task configuration by name."""
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def get_enabled_agents(self) -> List[AgentConfig]:
        """Get list of enabled agents."""
        return [agent for agent in self.agents if agent.enabled]
