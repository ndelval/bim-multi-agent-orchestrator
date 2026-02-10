"""
Unit tests for refactored Orchestrator class.

Tests the main orchestrator facade that coordinates all specialized components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from pathlib import Path
from typing import Any

from orchestrator.core.orchestrator_refactored import Orchestrator
from orchestrator.core.config import OrchestratorConfig, AgentConfig, TaskConfig, MemoryConfig
from orchestrator.core.exceptions import (
    OrchestratorError,
    AgentCreationError,
    TaskExecutionError,
    WorkflowError,
)
from orchestrator.core.initializer import OrchestratorInitializer
from orchestrator.core.lifecycle import LifecycleManager
from orchestrator.core.executor import OrchestratorExecutor


@pytest.fixture
def basic_config():
    """Create basic orchestrator configuration."""
    return OrchestratorConfig(
        name="test_orchestrator",
        agents=[
            AgentConfig(
                name="TestAgent",
                role="Tester",
                goal="Test things",
                backstory="Expert tester",
                instructions="Test thoroughly",
            )
        ],
        tasks=[
            TaskConfig(
                name="test_task",
                description="Test task",
                expected_output="Test result",
                agent="TestAgent",
                is_start=True,
            )
        ],
    )


@pytest.fixture
def empty_config():
    """Create empty orchestrator configuration."""
    return OrchestratorConfig(name="empty_orchestrator")


@pytest.fixture
def orchestrator_with_config(basic_config):
    """Create orchestrator with configuration but prevent initialization."""
    with patch.object(Orchestrator, 'initialize'):
        return Orchestrator(basic_config)


class TestOrchestratorInit:
    """Test Orchestrator initialization."""

    def test_init_with_config(self, basic_config):
        """Test initialization with config that has agents and tasks."""
        with patch.object(Orchestrator, 'initialize') as mock_init:
            orch = Orchestrator(basic_config)

            assert orch.config == basic_config
            assert isinstance(orch.initializer, OrchestratorInitializer)
            assert isinstance(orch.lifecycle, LifecycleManager)
            assert orch.executor is None
            assert orch.memory_manager is None
            assert orch.workflow_engine is None
            assert orch.graph_factory is None
            assert orch.compiled_graph is None
            assert orch.agents == {}
            assert orch.tasks == []
            assert orch.is_initialized is False

            # Should call initialize() because config has agents and tasks
            mock_init.assert_called_once()

    def test_init_without_config(self):
        """Test initialization without config."""
        orch = Orchestrator()

        assert isinstance(orch.config, OrchestratorConfig)
        assert orch.config.name == "orchestrator"  # Default name
        assert orch.is_initialized is False

    def test_init_with_empty_config(self, empty_config):
        """Test initialization with empty config."""
        orch = Orchestrator(empty_config)

        # Should not auto-initialize because no agents/tasks
        assert orch.is_initialized is False


class TestCallbackProperties:
    """Test callback property setters and getters."""

    def test_on_workflow_start_property(self, orchestrator_with_config):
        """Test workflow start callback property."""
        callback = Mock()
        orchestrator_with_config.on_workflow_start = callback

        assert orchestrator_with_config.on_workflow_start == callback
        assert orchestrator_with_config.lifecycle.on_workflow_start == callback

    def test_on_workflow_complete_property(self, orchestrator_with_config):
        """Test workflow complete callback property."""
        callback = Mock()
        orchestrator_with_config.on_workflow_complete = callback

        assert orchestrator_with_config.on_workflow_complete == callback
        assert orchestrator_with_config.lifecycle.on_workflow_complete == callback

    def test_on_task_start_property(self, orchestrator_with_config):
        """Test task start callback property."""
        callback = Mock()
        orchestrator_with_config.on_task_start = callback

        assert orchestrator_with_config.on_task_start == callback
        assert orchestrator_with_config.lifecycle.on_task_start == callback

    def test_on_task_complete_property(self, orchestrator_with_config):
        """Test task complete callback property."""
        callback = Mock()
        orchestrator_with_config.on_task_complete = callback

        assert orchestrator_with_config.on_task_complete == callback
        assert orchestrator_with_config.lifecycle.on_task_complete == callback

    def test_on_error_property(self, orchestrator_with_config):
        """Test error callback property."""
        callback = Mock()
        orchestrator_with_config.on_error = callback

        assert orchestrator_with_config.on_error == callback
        assert orchestrator_with_config.lifecycle.on_error == callback


class TestInitialize:
    """Test orchestrator initialization."""

    @patch.object(OrchestratorInitializer, 'initialize_memory')
    @patch.object(OrchestratorInitializer, 'initialize_workflow_engine')
    @patch.object(OrchestratorInitializer, 'create_agents')
    @patch.object(OrchestratorInitializer, 'create_langgraph_system')
    def test_initialize_success(
        self,
        mock_create_graph,
        mock_create_agents,
        mock_init_workflow,
        mock_init_memory,
        basic_config,
    ):
        """Test successful initialization."""
        # Setup mocks
        mock_memory = Mock()
        mock_workflow = Mock()
        mock_agents = {"TestAgent": Mock()}
        mock_graph_factory = Mock()
        mock_compiled_graph = Mock()

        mock_init_memory.return_value = mock_memory
        mock_init_workflow.return_value = mock_workflow
        mock_create_agents.return_value = mock_agents
        mock_create_graph.return_value = (mock_graph_factory, mock_compiled_graph)

        # Initialize
        orch = Orchestrator(basic_config)

        assert orch.is_initialized is True
        assert orch.memory_manager == mock_memory
        assert orch.workflow_engine == mock_workflow
        assert orch.agents == mock_agents
        assert orch.graph_factory == mock_graph_factory
        assert orch.compiled_graph == mock_compiled_graph
        assert isinstance(orch.executor, OrchestratorExecutor)

    @patch.object(OrchestratorInitializer, 'initialize_memory')
    def test_initialize_failure(self, mock_init_memory, basic_config):
        """Test initialization failure handling."""
        mock_init_memory.side_effect = Exception("Initialization error")

        with pytest.raises(OrchestratorError, match="Initialization failed"):
            Orchestrator(basic_config)


class TestRunWorkflow:
    """Test workflow execution."""

    @pytest.mark.asyncio
    async def test_run_not_initialized(self, orchestrator_with_config):
        """Test run raises error when not initialized."""
        with pytest.raises(OrchestratorError, match="not initialized"):
            await orchestrator_with_config.run()

    @pytest.mark.asyncio
    async def test_run_no_compiled_graph(self, orchestrator_with_config):
        """Test run raises error when graph not compiled."""
        orchestrator_with_config.is_initialized = True
        orchestrator_with_config.compiled_graph = None

        with pytest.raises(OrchestratorError, match="LangGraph system not created"):
            await orchestrator_with_config.run()

    @pytest.mark.asyncio
    async def test_run_success(self, orchestrator_with_config):
        """Test successful workflow execution."""
        # Setup mocks
        orchestrator_with_config.is_initialized = True
        orchestrator_with_config.compiled_graph = Mock()

        mock_executor = Mock()
        mock_executor.build_recall_content = Mock(return_value="recall content")
        mock_executor.run_langgraph_workflow = AsyncMock(return_value="workflow result")
        orchestrator_with_config.executor = mock_executor

        # Mock lifecycle methods
        with patch.object(orchestrator_with_config.lifecycle, 'emit_workflow_start'):
            # Run workflow
            result = await orchestrator_with_config.run()

            assert result == "workflow result"
            orchestrator_with_config.lifecycle.emit_workflow_start.assert_called_once()
            mock_executor.build_recall_content.assert_called_once()
            mock_executor.run_langgraph_workflow.assert_called_once_with("recall content")

    @pytest.mark.asyncio
    async def test_run_with_recall_failure(self, orchestrator_with_config):
        """Test workflow continues when recall fails."""
        orchestrator_with_config.is_initialized = True
        orchestrator_with_config.compiled_graph = Mock()

        mock_executor = Mock()
        mock_executor.build_recall_content = Mock(side_effect=Exception("Recall failed"))
        mock_executor.run_langgraph_workflow = AsyncMock(return_value="result")
        orchestrator_with_config.executor = mock_executor

        # Should still succeed without recall
        result = await orchestrator_with_config.run()

        assert result == "result"
        mock_executor.run_langgraph_workflow.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_run_workflow_failure(self, orchestrator_with_config):
        """Test workflow execution failure."""
        orchestrator_with_config.is_initialized = True
        orchestrator_with_config.compiled_graph = Mock()

        mock_executor = Mock()
        mock_executor.build_recall_content = Mock(return_value=None)
        mock_executor.run_langgraph_workflow = AsyncMock(side_effect=Exception("Execution failed"))
        orchestrator_with_config.executor = mock_executor

        with patch.object(orchestrator_with_config.lifecycle, 'emit_error') as mock_emit_error:
            with patch.object(orchestrator_with_config.lifecycle, 'emit_workflow_start'):
                with pytest.raises(WorkflowError, match="Workflow execution failed"):
                    await orchestrator_with_config.run()

                mock_emit_error.assert_called_once()

    def test_run_sync(self, orchestrator_with_config):
        """Test synchronous workflow execution."""
        orchestrator_with_config.is_initialized = True
        orchestrator_with_config.compiled_graph = Mock()

        mock_executor = Mock()
        mock_executor.build_recall_content = Mock(return_value=None)
        mock_executor.run_langgraph_workflow = AsyncMock(return_value="sync result")
        orchestrator_with_config.executor = mock_executor

        result = orchestrator_with_config.run_sync()

        assert result == "sync result"


class TestDynamicAgentTask:
    """Test dynamic agent and task addition."""

    def test_add_agent(self, orchestrator_with_config):
        """Test adding agent dynamically."""
        new_agent_config = AgentConfig(
            name="NewAgent",
            role="New Role",
            goal="New goal",
            backstory="New backstory",
            instructions="New instructions",
        )

        mock_agent = Mock()
        orchestrator_with_config.agent_factory.create_agent = Mock(return_value=mock_agent)

        agent = orchestrator_with_config.add_agent(new_agent_config)

        assert agent == mock_agent
        assert "NewAgent" in orchestrator_with_config.agents
        assert new_agent_config in orchestrator_with_config.config.agents

    def test_add_agent_reinitialize_graph(self, orchestrator_with_config):
        """Test adding agent reinitializes graph when already initialized."""
        orchestrator_with_config.is_initialized = True
        orchestrator_with_config.memory_manager = Mock()
        orchestrator_with_config.executor = Mock()

        mock_graph_factory = Mock()
        mock_compiled_graph = Mock()
        orchestrator_with_config.initializer.create_langgraph_system = Mock(
            return_value=(mock_graph_factory, mock_compiled_graph)
        )

        new_agent_config = AgentConfig(
            name="NewAgent",
            role="Role",
            goal="Goal",
            backstory="Story",
            instructions="Instructions",
        )

        mock_agent = Mock()
        orchestrator_with_config.agent_factory.create_agent = Mock(return_value=mock_agent)

        orchestrator_with_config.add_agent(new_agent_config)

        # Should recreate graph system
        orchestrator_with_config.initializer.create_langgraph_system.assert_called_once()
        assert orchestrator_with_config.executor.compiled_graph == mock_compiled_graph

    def test_add_task(self, orchestrator_with_config):
        """Test adding task dynamically."""
        # Add an agent first
        agent_config = AgentConfig(
            name="TaskAgent",
            role="Role",
            goal="Goal",
            backstory="Story",
            instructions="Instructions",
        )
        mock_agent = Mock()
        orchestrator_with_config.agents["TaskAgent"] = mock_agent

        task_config = TaskConfig(
            name="new_task",
            description="New task",
            expected_output="Output",
            agent="TaskAgent",
        )

        mock_task = Mock()
        orchestrator_with_config.task_factory.create_task = Mock(return_value=mock_task)

        task = orchestrator_with_config.add_task(task_config)

        assert task == mock_task
        assert mock_task in orchestrator_with_config.tasks
        assert task_config in orchestrator_with_config.config.tasks

    def test_add_task_agent_not_found(self, orchestrator_with_config):
        """Test adding task with non-existent agent."""
        task_config = TaskConfig(
            name="new_task",
            description="New task",
            expected_output="Output",
            agent="NonExistentAgent",
        )

        with pytest.raises(TaskExecutionError, match="Agent 'NonExistentAgent' not found"):
            orchestrator_with_config.add_task(task_config)


class TestPlanFromPrompt:
    """Test dynamic task planning."""

    @patch.object(OrchestratorInitializer, 'initialize_memory')
    @patch.object(OrchestratorInitializer, 'initialize_workflow_engine')
    @patch.object(OrchestratorInitializer, 'create_agents')
    @patch.object(OrchestratorInitializer, 'create_langgraph_system')
    def test_plan_from_prompt(
        self,
        mock_create_graph,
        mock_create_agents,
        mock_init_workflow,
        mock_init_memory,
        basic_config,
    ):
        """Test dynamic plan from prompt."""
        # Setup mocks
        mock_create_graph.return_value = (Mock(), Mock())
        mock_create_agents.return_value = {}
        mock_init_workflow.return_value = Mock()
        mock_init_memory.return_value = Mock()

        orch = Orchestrator(basic_config)

        # Mock executor plan_from_prompt
        mock_dynamic_tasks = [
            TaskConfig(
                name="dynamic_task",
                description="Dynamic",
                expected_output="Output",
                agent="TestAgent",
            )
        ]
        orch.executor.plan_from_prompt = Mock(return_value=mock_dynamic_tasks)

        orch.plan_from_prompt(
            prompt="Create a plan",
            agent_sequence=["TestAgent"],
        )

        # Should replace config tasks
        assert orch.config.tasks == mock_dynamic_tasks
        assert orch.tasks == []

    def test_plan_from_prompt_not_initialized(self, orchestrator_with_config):
        """Test plan_from_prompt initializes if needed."""
        orchestrator_with_config.is_initialized = False
        orchestrator_with_config.executor = Mock()
        orchestrator_with_config.executor.plan_from_prompt = Mock(return_value=[])

        with patch.object(orchestrator_with_config, 'initialize'):
            with patch.object(orchestrator_with_config.initializer, 'initialize_workflow_engine', return_value=Mock()):
                with patch.object(orchestrator_with_config.initializer, 'create_langgraph_system', return_value=(Mock(), Mock())):
                    orchestrator_with_config.plan_from_prompt(
                        prompt="Test",
                        agent_sequence=["TestAgent"],
                    )

                    orchestrator_with_config.initialize.assert_called_once()


class TestHelperMethods:
    """Test helper and utility methods."""

    def test_register_agent_template(self, orchestrator_with_config):
        """Test registering agent template."""
        mock_template = Mock()
        orchestrator_with_config.agent_factory = Mock()
        orchestrator_with_config.register_agent_template(mock_template)

        orchestrator_with_config.agent_factory.register_template.assert_called_once_with(mock_template)

    def test_register_task_template(self, orchestrator_with_config):
        """Test registering task template."""
        mock_template = Mock()
        orchestrator_with_config.task_factory = Mock()
        orchestrator_with_config.register_task_template(mock_template)

        orchestrator_with_config.task_factory.register_template.assert_called_once_with(mock_template)

    def test_get_agent(self, orchestrator_with_config):
        """Test getting agent by name."""
        mock_agent = Mock()
        orchestrator_with_config.agents["TestAgent"] = mock_agent

        assert orchestrator_with_config.get_agent("TestAgent") == mock_agent
        assert orchestrator_with_config.get_agent("NonExistent") is None

    def test_get_task(self, orchestrator_with_config):
        """Test getting task by name."""
        mock_task = Mock()
        mock_task.name = "test_task"
        orchestrator_with_config.tasks.append(mock_task)

        assert orchestrator_with_config.get_task("test_task") == mock_task
        assert orchestrator_with_config.get_task("nonexistent") is None

    def test_get_workflow_status(self, orchestrator_with_config):
        """Test getting workflow status."""
        mock_workflow = Mock()
        mock_workflow.get_workflow_status = Mock(return_value={"status": "running"})
        orchestrator_with_config.workflow_engine = mock_workflow

        status = orchestrator_with_config.get_workflow_status()

        assert status == {"status": "running"}

    def test_get_workflow_status_not_initialized(self, orchestrator_with_config):
        """Test getting workflow status when not initialized."""
        orchestrator_with_config.workflow_engine = None

        status = orchestrator_with_config.get_workflow_status()

        assert status == {"status": "not_initialized"}

    def test_get_system_info(self, orchestrator_with_config):
        """Test getting system information."""
        orchestrator_with_config.is_initialized = True

        info = orchestrator_with_config.get_system_info()

        assert info["name"] == orchestrator_with_config.config.name
        assert info["initialized"] is True
        assert "agents" in info
        assert "tasks" in info
        assert "execution" in info


class TestResourceManagement:
    """Test resource management methods."""

    def test_reset(self, orchestrator_with_config):
        """Test resetting orchestrator."""
        # Setup state
        orchestrator_with_config.agents = {"test": Mock()}
        orchestrator_with_config.tasks = [Mock()]
        orchestrator_with_config.compiled_graph = Mock()
        orchestrator_with_config.is_initialized = True

        mock_memory = Mock()
        mock_workflow = Mock()
        orchestrator_with_config.memory_manager = mock_memory
        orchestrator_with_config.workflow_engine = mock_workflow

        with patch.object(orchestrator_with_config.lifecycle, 'reset') as mock_lifecycle_reset:
            orchestrator_with_config.reset()

            assert len(orchestrator_with_config.agents) == 0
            assert len(orchestrator_with_config.tasks) == 0
            assert orchestrator_with_config.compiled_graph is None
            assert orchestrator_with_config.is_initialized is False
            mock_memory.cleanup.assert_called_once()
            mock_workflow.reset_workflow.assert_called_once()
            mock_lifecycle_reset.assert_called_once()

    def test_cleanup(self, orchestrator_with_config):
        """Test cleaning up resources."""
        mock_memory = Mock()
        mock_workflow = Mock()
        mock_agent_factory = Mock()
        mock_agent_factory.cleanup_mcp = AsyncMock()

        orchestrator_with_config.memory_manager = mock_memory
        orchestrator_with_config.workflow_engine = mock_workflow
        orchestrator_with_config.agent_factory = mock_agent_factory

        with patch.object(orchestrator_with_config, 'reset'):
            orchestrator_with_config.cleanup()

            mock_memory.cleanup.assert_called_once()
            mock_workflow.cancel_workflow.assert_called_once()
            orchestrator_with_config.reset.assert_called_once()


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager(self, basic_config):
        """Test using orchestrator as context manager."""
        with patch.object(Orchestrator, 'initialize'):
            with patch.object(Orchestrator, 'cleanup') as mock_cleanup:
                with Orchestrator(basic_config) as orch:
                    assert orch is not None

                mock_cleanup.assert_called_once()


class TestRepr:
    """Test string representation."""

    def test_repr(self, orchestrator_with_config):
        """Test __repr__ method."""
        orchestrator_with_config.agents = {"agent1": Mock(), "agent2": Mock()}
        orchestrator_with_config.tasks = [Mock(), Mock(), Mock()]
        orchestrator_with_config.is_initialized = True

        repr_str = repr(orchestrator_with_config)

        assert "test_orchestrator" in repr_str
        assert "agents=2" in repr_str
        assert "tasks=3" in repr_str
        assert "initialized=True" in repr_str
