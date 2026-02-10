"""
Main orchestrator class that coordinates all components.

REFACTORED VERSION: This is the new slim orchestrator that delegates to
specialized components following the Single Responsibility Principle.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Sequence
from pathlib import Path

# LangGraph integration - required
from ..integrations.langchain_integration import (
    LangChainAgent as Agent,
    LangChainTask as Task,
)
from ..factories.graph_factory import GraphFactory
from .config import OrchestratorConfig, AgentConfig, TaskConfig
from .exceptions import (
    OrchestratorError,
    AgentCreationError,
    TaskExecutionError,
    WorkflowError
)
from ..factories.agent_factory import AgentFactory
from ..factories.task_factory import TaskFactory
from ..memory.memory_manager import MemoryManager
from ..workflow.workflow_engine import WorkflowEngine, WorkflowMetrics

# Import new specialized components
from .initializer import OrchestratorInitializer
from .lifecycle import LifecycleManager
from .executor import OrchestratorExecutor

logger = logging.getLogger(__name__)
logger.info("Orchestrator initialized with LangGraph backend (Refactored)")


class Orchestrator:
    """
    Main orchestrator class for managing multi-agent workflows.

    This refactored class follows the Single Responsibility Principle by delegating
    to specialized components:
    - OrchestratorInitializer: Component initialization
    - LifecycleManager: Callback and event management
    - OrchestratorExecutor: Workflow execution

    The orchestrator acts as a facade, providing a clean API while delegating
    implementation details to specialized components.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration object. If None, creates default configuration.
        """
        self.config = config or OrchestratorConfig(name="orchestrator")

        # Initialize specialized components
        self.initializer = OrchestratorInitializer(self.config)
        self.lifecycle = LifecycleManager()
        self.executor: Optional[OrchestratorExecutor] = None

        # Factories (delegated to initializer)
        self.agent_factory = self.initializer.agent_factory
        self.task_factory = self.initializer.task_factory

        # Core components
        self.memory_manager: Optional[MemoryManager] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.graph_factory: Optional[GraphFactory] = None
        self.compiled_graph: Optional[Any] = None

        # State
        self.agents: Dict[str, Agent] = {}
        self.tasks: List[Any] = []
        self.is_initialized = False

        # Backward compatibility: expose lifecycle callbacks
        self._setup_callback_properties()

        # Initialize if config has agents and tasks
        if self.config.agents and self.config.tasks:
            self.initialize()

    def _setup_callback_properties(self) -> None:
        """Setup property delegation for backward compatibility."""
        # Note: Python properties don't work well for function assignment,
        # so we'll handle this through direct lifecycle manager access

    @property
    def on_workflow_start(self) -> Optional[Callable[[], None]]:
        """Get workflow start callback."""
        return self.lifecycle.on_workflow_start

    @on_workflow_start.setter
    def on_workflow_start(self, value: Optional[Callable[[], None]]) -> None:
        """Set workflow start callback."""
        self.lifecycle.on_workflow_start = value

    @property
    def on_workflow_complete(self) -> Optional[Callable[[WorkflowMetrics], None]]:
        """Get workflow complete callback."""
        return self.lifecycle.on_workflow_complete

    @on_workflow_complete.setter
    def on_workflow_complete(self, value: Optional[Callable[[WorkflowMetrics], None]]) -> None:
        """Set workflow complete callback."""
        self.lifecycle.on_workflow_complete = value

    @property
    def on_task_start(self) -> Optional[Callable[[str, Any], None]]:
        """Get task start callback."""
        return self.lifecycle.on_task_start

    @on_task_start.setter
    def on_task_start(self, value: Optional[Callable[[str, Any], None]]) -> None:
        """Set task start callback."""
        self.lifecycle.on_task_start = value

    @property
    def on_task_complete(self) -> Optional[Callable[[str, Any], None]]:
        """Get task complete callback."""
        return self.lifecycle.on_task_complete

    @on_task_complete.setter
    def on_task_complete(self, value: Optional[Callable[[str, Any], None]]) -> None:
        """Set task complete callback."""
        self.lifecycle.on_task_complete = value

    @property
    def on_error(self) -> Optional[Callable[[Exception], None]]:
        """Get error callback."""
        return self.lifecycle.on_error

    @on_error.setter
    def on_error(self, value: Optional[Callable[[Exception], None]]) -> None:
        """Set error callback."""
        self.lifecycle.on_error = value

    def initialize(self) -> None:
        """Initialize all components."""
        try:
            logger.info(f"Initializing orchestrator: {self.config.name}")

            # Initialize memory manager
            self.memory_manager = self.initializer.initialize_memory()

            # Initialize workflow engine with lifecycle callbacks
            callbacks = self.lifecycle.get_workflow_callbacks()
            self.workflow_engine = self.initializer.initialize_workflow_engine(
                on_task_start=callbacks['on_task_start'],
                on_task_complete=callbacks['on_task_complete'],
                on_task_fail=callbacks['on_task_fail'],
                on_workflow_complete=callbacks['on_workflow_complete']
            )

            # Create agents
            self.agents = self.initializer.create_agents()

            # Create LangGraph system
            self.graph_factory, self.compiled_graph = self.initializer.create_langgraph_system(
                self.memory_manager
            )

            # Initialize executor
            self.executor = OrchestratorExecutor(
                self.config,
                self.compiled_graph,
                self.memory_manager,
                self.workflow_engine
            )

            self.is_initialized = True
            logger.info("Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise OrchestratorError(f"Initialization failed: {str(e)}")

    async def run(self) -> Any:
        """
        Run the orchestrator workflow.

        Returns:
            The result of the workflow execution.

        Raises:
            OrchestratorError: If orchestrator not initialized
            WorkflowError: If workflow execution fails
        """
        if not self.is_initialized:
            raise OrchestratorError("Orchestrator not initialized")

        if not self.compiled_graph:
            raise OrchestratorError("LangGraph system not created")

        try:
            logger.info(f"Starting orchestrator workflow: {self.config.name}")

            # Call start callback
            self.lifecycle.emit_workflow_start()

            # Build recall content from memory
            recall_content = None
            try:
                recall_content = self.executor.build_recall_content()
            except Exception as e:
                logger.warning(f"Recall content build failed, continuing without recall: {e}")

            # Execute LangGraph workflow
            result = await self.executor.run_langgraph_workflow(recall_content)

            logger.info("Orchestrator workflow completed successfully")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            self.lifecycle.emit_error(e)
            raise WorkflowError(f"Workflow execution failed: {str(e)}")

    def run_sync(self) -> Any:
        """
        Run the orchestrator workflow synchronously.

        Returns:
            The result of the workflow execution.
        """
        return asyncio.run(self.run())

    def add_agent(self, agent_config: AgentConfig) -> Agent:
        """
        Add a new agent to the orchestrator.

        Args:
            agent_config: Configuration for the new agent.

        Returns:
            The created agent.
        """
        try:
            agent = self.agent_factory.create_agent(agent_config)
            self.agents[agent_config.name] = agent
            self.config.agents.append(agent_config)

            # Reinitialize if already initialized
            if self.is_initialized:
                # Recreate LangGraph system
                self.graph_factory, self.compiled_graph = self.initializer.create_langgraph_system(
                    self.memory_manager
                )
                # Update executor
                self.executor.compiled_graph = self.compiled_graph

            logger.info(f"Added agent: {agent_config.name}")
            return agent
        except Exception as e:
            raise AgentCreationError(f"Failed to add agent: {str(e)}")

    def add_task(self, task_config: TaskConfig) -> Task:
        """
        Add a new task to the orchestrator.

        Args:
            task_config: Configuration for the new task.

        Returns:
            The created task.
        """
        try:
            # Check if agent exists
            agent = self.agents.get(task_config.agent_name)
            if not agent:
                raise TaskExecutionError(f"Agent '{task_config.agent_name}' not found")

            # Create task
            task = self.task_factory.create_task(task_config, agent)
            self.tasks.append(task)
            self.config.tasks.append(task_config)

            # Reinitialize if already initialized
            if self.is_initialized:
                # Recreate LangGraph system
                self.graph_factory, self.compiled_graph = self.initializer.create_langgraph_system(
                    self.memory_manager
                )
                # Update executor
                self.executor.compiled_graph = self.compiled_graph

            logger.info(f"Added task: {task_config.name}")
            return task
        except Exception as e:
            raise TaskExecutionError(f"Failed to add task: {str(e)}")

    def plan_from_prompt(
        self,
        prompt: str,
        agent_sequence: Sequence[str],
        *,
        recall_snippets: Optional[Sequence[str]] = None,
        assignments: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Generate task plan dynamically from prompt and selected agents.

        Args:
            prompt: User prompt
            agent_sequence: Sequence of agent names
            recall_snippets: Optional memory recall snippets
            assignments: Optional task assignments

        Raises:
            ValueError: If agent_sequence is empty
            AgentCreationError: If agents not available
        """
        if not self.is_initialized:
            self.initialize()

        enabled_agents = {agent.name: agent for agent in self.config.agents if agent.enabled}

        # Generate dynamic tasks using executor
        dynamic_tasks = self.executor.plan_from_prompt(
            prompt,
            agent_sequence,
            enabled_agents,
            recall_snippets=recall_snippets,
            assignments=assignments
        )

        # Replace current task configuration and rebuild execution stack
        self.config.tasks = dynamic_tasks
        self.tasks = []

        # Refresh workflow engine state and LangGraph system
        callbacks = self.lifecycle.get_workflow_callbacks()
        self.workflow_engine = self.initializer.initialize_workflow_engine(
            on_task_start=callbacks['on_task_start'],
            on_task_complete=callbacks['on_task_complete'],
            on_task_fail=callbacks['on_task_fail'],
            on_workflow_complete=callbacks['on_workflow_complete']
        )

        self.graph_factory, self.compiled_graph = self.initializer.create_langgraph_system(
            self.memory_manager
        )

        # Update executor
        self.executor.compiled_graph = self.compiled_graph

    def register_agent_template(self, template) -> None:
        """Register a custom agent template."""
        self.agent_factory.register_template(template)

    def register_task_template(self, template) -> None:
        """Register a custom task template."""
        self.task_factory.register_template(template)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def get_task(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        if self.workflow_engine:
            return self.workflow_engine.get_workflow_status()
        return {"status": "not_initialized"}

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator system."""
        info = {
            "name": self.config.name,
            "initialized": self.is_initialized,
            "agents": {
                "total": len(self.config.agents),
                "enabled": len([a for a in self.config.agents if a.enabled]),
                "types": list(self.agent_factory.list_templates())
            },
            "tasks": {
                "total": len(self.config.tasks),
                "types": list(self.task_factory.list_templates())
            },
            "execution": {
                "process": self.config.process,
                "async": self.config.async_execution,
                "memory": (self.config.memory is not None)
            }
        }

        if self.memory_manager:
            info["memory"] = self.memory_manager.get_provider_info()

        return info

    def export_config(self, file_path: Union[str, Path]) -> None:
        """Export current configuration to file."""
        self.config.save_to_file(file_path)
        logger.info(f"Configuration exported to: {file_path}")

    def create_graph_tool(self, *, user_id: Optional[str] = None, run_id: Optional[str] = None):
        """Create a GraphRAG lookup tool if the memory manager supports it."""
        if not self.memory_manager:
            raise MemoryError("Memory manager not initialized; cannot create graph tool")
        user = user_id or self.config.user_id
        run = run_id or self.config.user_id
        return self.memory_manager.create_graph_tool(default_user_id=user, default_run_id=run)

    def import_config(self, file_path: Union[str, Path]) -> None:
        """Import configuration from file and reinitialize."""
        new_config = OrchestratorConfig.from_file(file_path)
        self.config = new_config

        # Update initializer config
        self.initializer.config = new_config

        # Reinitialize with new config
        self.is_initialized = False
        self.initialize()

        logger.info(f"Configuration imported from: {file_path}")

    def merge_config(self, other_config: OrchestratorConfig) -> None:
        """Merge another configuration with current one."""
        self.config = self.config.merge(other_config)

        # Update initializer config
        self.initializer.config = self.config

        # Reinitialize with merged config
        self.is_initialized = False
        self.initialize()

        logger.info("Configuration merged and reinitialized")

    def reset(self) -> None:
        """Reset the orchestrator to initial state."""
        self.agents.clear()
        self.tasks.clear()
        self.compiled_graph = None
        self.is_initialized = False

        if self.memory_manager:
            self.memory_manager.cleanup()
            self.memory_manager = None

        if self.workflow_engine:
            self.workflow_engine.reset_workflow()
            self.workflow_engine = None

        # Reset lifecycle
        self.lifecycle.reset()

        logger.info("Orchestrator reset")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.memory_manager:
            self.memory_manager.cleanup()

        if self.workflow_engine:
            self.workflow_engine.cancel_workflow()

        # Clean up MCP client connections
        if self.agent_factory and hasattr(self.agent_factory, 'cleanup_mcp'):
            try:
                asyncio.run(self.agent_factory.cleanup_mcp())
            except Exception as e:
                logger.warning(f"Error cleaning up MCP connections: {e}")

        self.reset()
        logger.info("Orchestrator cleanup completed")

    # Class methods for quick creation
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Orchestrator":
        """Create orchestrator from configuration file."""
        config = OrchestratorConfig.from_file(file_path)
        return cls(config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Orchestrator":
        """Create orchestrator from configuration dictionary."""
        config = OrchestratorConfig.from_dict(config_dict)
        return cls(config)

    @classmethod
    def from_env(cls, prefix: str = "ORCHESTRATOR_") -> "Orchestrator":
        """Create orchestrator from environment variables."""
        config = OrchestratorConfig.from_env(prefix)
        return cls(config)

    @classmethod
    def create_default(cls, name: str = "DefaultOrchestrator") -> "Orchestrator":
        """Create orchestrator with default configuration and common agents/tasks."""
        from ..factories.agent_factory import (
            OrchestratorAgentTemplate, ResearcherAgentTemplate,
            PlannerAgentTemplate, ImplementerAgentTemplate,
            TesterAgentTemplate, WriterAgentTemplate
        )

        # Create configuration with default agents and tasks
        config = OrchestratorConfig(name=name)

        # Add default agents
        agent_templates = [
            OrchestratorAgentTemplate(),
            ResearcherAgentTemplate(),
            PlannerAgentTemplate(),
            ImplementerAgentTemplate(),
            TesterAgentTemplate(),
            WriterAgentTemplate()
        ]

        for template in agent_templates:
            agent_config = template.get_default_config()
            config.agents.append(agent_config)

        # Add default tasks with dependencies
        task_configs = [
            TaskConfig(
                name="research_task",
                description="Research best practices for the given topic.",
                expected_output="Comprehensive research findings with citations",
                agent_name="Researcher",
                async_execution=True,
                is_start=True
            ),
            TaskConfig(
                name="plan_task",
                description="Create an actionable plan based on research.",
                expected_output="Detailed plan with steps and acceptance criteria",
                agent_name="Planner",
                async_execution=True,
                is_start=True
            ),
            TaskConfig(
                name="implement_task",
                description="Implement the solution according to the plan.",
                expected_output="Implementation details and design notes",
                agent_name="Implementer",
                context=["research_task", "plan_task"]
            ),
            TaskConfig(
                name="test_task",
                description="Create test strategy for the implementation.",
                expected_output="Test plan with cases and expected outcomes",
                agent_name="Tester",
                context=["implement_task"]
            ),
            TaskConfig(
                name="document_task",
                description="Create comprehensive documentation.",
                expected_output="Final documentation with summary and details",
                agent_name="Writer",
                context=["research_task", "plan_task", "implement_task", "test_task"]
            )
        ]

        config.tasks.extend(task_configs)

        return cls(config)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __repr__(self) -> str:
        """String representation."""
        return (f"Orchestrator(name='{self.config.name}', "
                f"agents={len(self.agents)}, tasks={len(self.tasks)}, "
                f"initialized={self.is_initialized})")
