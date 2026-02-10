"""
Integration tests for refactored orchestrator modules.

Tests the interaction between Orchestrator, Initializer, Lifecycle, and Executor.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any

from orchestrator.core.orchestrator_refactored import Orchestrator
from orchestrator.core.config import OrchestratorConfig, AgentConfig, TaskConfig
from orchestrator.core.initializer import OrchestratorInitializer
from orchestrator.core.lifecycle import LifecycleManager
from orchestrator.core.executor import OrchestratorExecutor
from orchestrator.workflow.workflow_engine import WorkflowMetrics


@pytest.fixture
def full_config():
    """Create full orchestrator configuration for integration testing."""
    return OrchestratorConfig(
        name="integration_test",
        agents=[
            AgentConfig(
                name="ResearchAgent",
                role="Researcher",
                goal="Research topics",
                backstory="Expert researcher",
                instructions="Research thoroughly",
            ),
            AgentConfig(
                name="AnalystAgent",
                role="Analyst",
                goal="Analyze data",
                backstory="Data expert",
                instructions="Analyze carefully",
            ),
        ],
        tasks=[
            TaskConfig(
                name="research_task",
                description="Research the topic",
                expected_output="Research findings",
                agent="ResearchAgent",
                is_start=True,
            ),
            TaskConfig(
                name="analysis_task",
                description="Analyze the research",
                expected_output="Analysis report",
                agent="AnalystAgent",
                context=["research_task"],
            ),
        ],
    )


class TestOrchestratorInitializerIntegration:
    """Test integration between Orchestrator and Initializer."""

    @patch('orchestrator.memory.memory_manager.MemoryManager')
    @patch('orchestrator.workflow.workflow_engine.WorkflowEngine')
    @patch('orchestrator.factories.graph_factory.GraphFactory')
    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    def test_full_initialization_flow(
        self,
        mock_create_agent,
        mock_graph_factory,
        mock_workflow_engine,
        mock_memory_manager,
        full_config,
    ):
        """Test full initialization flow from orchestrator through initializer."""
        # Setup mocks
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        mock_memory = Mock()
        mock_memory_manager.return_value = mock_memory

        mock_workflow = Mock()
        mock_workflow_engine.return_value = mock_workflow

        mock_factory = Mock()
        mock_compiled = Mock()
        mock_graph_factory.return_value.build_graph.return_value = (mock_factory, mock_compiled)

        # Create and initialize orchestrator
        orch = Orchestrator(full_config)

        # Verify initialization chain
        assert orch.is_initialized is True
        assert orch.memory_manager is not None
        assert orch.workflow_engine is not None
        assert len(orch.agents) == 2
        assert "ResearchAgent" in orch.agents
        assert "AnalystAgent" in orch.agents
        assert orch.executor is not None
        assert isinstance(orch.executor, OrchestratorExecutor)

    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    def test_initializer_creates_correct_components(self, mock_create_agent, full_config):
        """Test that initializer creates all required components correctly."""
        mock_create_agent.return_value = Mock()

        with patch.object(OrchestratorInitializer, 'initialize_memory', return_value=None):
            with patch.object(OrchestratorInitializer, 'initialize_workflow_engine', return_value=Mock()):
                with patch.object(OrchestratorInitializer, 'create_langgraph_system', return_value=(Mock(), Mock())):
                    orch = Orchestrator(full_config)

                    # Verify initializer is set up correctly
                    assert orch.initializer is not None
                    assert orch.initializer.config == full_config
                    assert orch.agent_factory is orch.initializer.agent_factory
                    assert orch.task_factory is orch.initializer.task_factory


class TestOrchestratorLifecycleIntegration:
    """Test integration between Orchestrator and LifecycleManager."""

    @pytest.mark.asyncio
    async def test_workflow_callbacks_integration(self, full_config):
        """Test that lifecycle callbacks are properly integrated in workflow."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(full_config)
            orch.is_initialized = True
            orch.compiled_graph = Mock()

            # Setup callback tracking
            workflow_start_called = []
            workflow_complete_called = []

            def on_start():
                workflow_start_called.append(True)

            def on_complete(metrics):
                workflow_complete_called.append(metrics)

            # Set callbacks through orchestrator properties
            orch.on_workflow_start = on_start
            orch.on_workflow_complete = on_complete

            # Verify callbacks are set in lifecycle manager
            assert orch.lifecycle.on_workflow_start == on_start
            assert orch.lifecycle.on_workflow_complete == on_complete

            # Mock executor
            mock_executor = Mock()
            mock_executor.build_recall_content = Mock(return_value=None)
            mock_executor.run_langgraph_workflow = AsyncMock(return_value="result")
            orch.executor = mock_executor

            # Run workflow
            await orch.run()

            # Verify callback was triggered
            assert len(workflow_start_called) == 1

    def test_callback_property_delegation(self, full_config):
        """Test that callback properties properly delegate to lifecycle manager."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(full_config)

            # Test all callback properties
            callbacks = {
                'on_workflow_start': Mock(),
                'on_workflow_complete': Mock(),
                'on_task_start': Mock(),
                'on_task_complete': Mock(),
                'on_error': Mock(),
            }

            for prop_name, callback in callbacks.items():
                setattr(orch, prop_name, callback)
                assert getattr(orch, prop_name) == callback
                assert getattr(orch.lifecycle, prop_name) == callback


class TestOrchestratorExecutorIntegration:
    """Test integration between Orchestrator and Executor."""

    @pytest.mark.asyncio
    async def test_workflow_execution_flow(self, full_config):
        """Test complete workflow execution through orchestrator and executor."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(full_config)
            orch.is_initialized = True
            orch.compiled_graph = Mock()
            orch.memory_manager = Mock()

            # Create real executor with mocked graph
            orch.executor = OrchestratorExecutor(
                config=full_config,
                compiled_graph=orch.compiled_graph,
                memory_manager=orch.memory_manager,
                workflow_engine=None,
            )

            # Mock graph execution
            mock_result = Mock()
            mock_result.final_output = "Execution completed"
            orch.compiled_graph.invoke = Mock(return_value=mock_result)

            # Run workflow
            result = await orch.run()

            # Verify execution
            assert result == "Execution completed"
            orch.compiled_graph.invoke.assert_called_once()

    def test_recall_content_building(self, full_config):
        """Test memory recall integration with executor."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(full_config)
            orch.is_initialized = True

            # Setup memory manager with recall config
            mock_memory = Mock()
            mock_memory.retrieve_filtered = Mock(return_value=[
                {"content": "Previous research finding", "metadata": {"source": "doc1"}},
                {"content": "Related analysis", "id": "mem2"},
            ])
            orch.memory_manager = mock_memory

            # Add recall config
            full_config.custom_config = {
                "recall": {
                    "query": "research findings",
                    "limit": 5,
                }
            }

            # Create executor
            orch.executor = OrchestratorExecutor(
                config=full_config,
                compiled_graph=Mock(),
                memory_manager=mock_memory,
                workflow_engine=None,
            )

            # Build recall
            recall = orch.executor.build_recall_content()

            # Verify recall content
            assert recall is not None
            assert "MEMORY RECALL CONTEXT:" in recall
            assert "Previous research finding" in recall
            assert "Related analysis" in recall


class TestDynamicComponentIntegration:
    """Test dynamic agent/task addition with workflow recreation."""

    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    def test_add_agent_recreates_workflow(self, mock_create_agent, full_config):
        """Test that adding agent recreates the workflow system."""
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        with patch.object(OrchestratorInitializer, 'initialize_memory', return_value=None):
            with patch.object(OrchestratorInitializer, 'initialize_workflow_engine', return_value=Mock()):
                with patch.object(OrchestratorInitializer, 'create_langgraph_system') as mock_create_graph:
                    mock_create_graph.return_value = (Mock(), Mock())

                    orch = Orchestrator(full_config)
                    initial_graph = orch.compiled_graph

                    # Add new agent
                    new_agent_config = AgentConfig(
                        name="NewAgent",
                        role="New Role",
                        goal="New goal",
                        backstory="New backstory",
                        instructions="New instructions",
                    )

                    orch.add_agent(new_agent_config)

                    # Verify graph was recreated
                    assert mock_create_graph.call_count == 2  # Once during init, once during add_agent
                    assert "NewAgent" in orch.agents
                    assert new_agent_config in orch.config.agents

    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    @patch('orchestrator.factories.task_factory.TaskFactory.create_task')
    def test_add_task_recreates_workflow(
        self,
        mock_create_task,
        mock_create_agent,
        full_config,
    ):
        """Test that adding task recreates the workflow system."""
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        with patch.object(OrchestratorInitializer, 'initialize_memory', return_value=None):
            with patch.object(OrchestratorInitializer, 'initialize_workflow_engine', return_value=Mock()):
                with patch.object(OrchestratorInitializer, 'create_langgraph_system') as mock_create_graph:
                    mock_create_graph.return_value = (Mock(), Mock())

                    orch = Orchestrator(full_config)

                    # Add new task
                    new_task_config = TaskConfig(
                        name="new_task",
                        description="New task",
                        expected_output="New output",
                        agent="ResearchAgent",
                    )

                    orch.add_task(new_task_config)

                    # Verify graph was recreated
                    assert mock_create_graph.call_count == 2
                    assert mock_task in orch.tasks
                    assert new_task_config in orch.config.tasks


class TestPlanFromPromptIntegration:
    """Test dynamic planning integration."""

    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    def test_plan_from_prompt_full_flow(self, mock_create_agent, full_config):
        """Test complete plan_from_prompt flow."""
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        with patch.object(OrchestratorInitializer, 'initialize_memory', return_value=Mock()):
            with patch.object(OrchestratorInitializer, 'initialize_workflow_engine', return_value=Mock()):
                with patch.object(OrchestratorInitializer, 'create_langgraph_system', return_value=(Mock(), Mock())):
                    orch = Orchestrator(full_config)

                    # Plan from prompt
                    orch.plan_from_prompt(
                        prompt="Create a research and analysis workflow",
                        agent_sequence=["ResearchAgent", "AnalystAgent"],
                        recall_snippets=["Previous context 1", "Previous context 2"],
                    )

                    # Verify tasks were generated
                    assert len(orch.config.tasks) > 0
                    # Tasks should reference the agents in sequence
                    task_agents = [task.agent for task in orch.config.tasks]
                    assert "ResearchAgent" in task_agents or "AnalystAgent" in task_agents


class TestResourceManagementIntegration:
    """Test resource management across modules."""

    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    def test_cleanup_cleans_all_components(self, mock_create_agent, full_config):
        """Test that cleanup properly cleans all components."""
        mock_create_agent.return_value = Mock()

        mock_memory = Mock()
        mock_workflow = Mock()
        mock_agent_factory = Mock()
        mock_agent_factory.cleanup_mcp = AsyncMock()

        with patch.object(OrchestratorInitializer, 'initialize_memory', return_value=mock_memory):
            with patch.object(OrchestratorInitializer, 'initialize_workflow_engine', return_value=mock_workflow):
                with patch.object(OrchestratorInitializer, 'create_langgraph_system', return_value=(Mock(), Mock())):
                    orch = Orchestrator(full_config)
                    orch.agent_factory = mock_agent_factory

                    # Cleanup
                    orch.cleanup()

                    # Verify all cleanup methods called (may be called multiple times if reset is called)
                    assert mock_memory.cleanup.called
                    assert mock_workflow.cancel_workflow.called
                    assert orch.is_initialized is False
                    assert len(orch.agents) == 0

    @patch('orchestrator.factories.agent_factory.AgentFactory.create_agent')
    def test_reset_preserves_config(self, mock_create_agent, full_config):
        """Test that reset clears state but preserves config."""
        mock_create_agent.return_value = Mock()

        with patch.object(OrchestratorInitializer, 'initialize_memory', return_value=Mock()):
            with patch.object(OrchestratorInitializer, 'initialize_workflow_engine', return_value=Mock()):
                with patch.object(OrchestratorInitializer, 'create_langgraph_system', return_value=(Mock(), Mock())):
                    orch = Orchestrator(full_config)
                    original_config = orch.config

                    # Reset
                    orch.reset()

                    # Verify config preserved
                    assert orch.config == original_config
                    assert orch.is_initialized is False
                    assert len(orch.agents) == 0
                    assert len(orch.tasks) == 0


class TestWorkflowMetricsIntegration:
    """Test workflow metrics collection integration."""

    def test_get_workflow_status_integration(self, full_config):
        """Test getting workflow status from workflow engine."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(full_config)

            mock_workflow = Mock()
            mock_workflow.get_workflow_status = Mock(return_value={
                "status": "running",
                "completed_tasks": 1,
                "total_tasks": 2,
            })
            orch.workflow_engine = mock_workflow

            status = orch.get_workflow_status()

            assert status["status"] == "running"
            assert status["completed_tasks"] == 1
            assert status["total_tasks"] == 2

    def test_get_system_info_integration(self, full_config):
        """Test getting comprehensive system info."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(full_config)
            orch.is_initialized = True

            mock_memory = Mock()
            mock_memory.get_provider_info = Mock(return_value={"provider": "hybrid"})
            orch.memory_manager = mock_memory

            info = orch.get_system_info()

            assert info["name"] == "integration_test"
            assert info["initialized"] is True
            assert info["agents"]["total"] == 2
            assert info["tasks"]["total"] == 2
            assert info["memory"]["provider"] == "hybrid"
