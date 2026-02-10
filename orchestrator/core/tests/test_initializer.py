"""
Unit tests for OrchestratorInitializer.

Tests component initialization logic following the Single Responsibility Principle.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from orchestrator.core.initializer import OrchestratorInitializer
from orchestrator.core.config import OrchestratorConfig, AgentConfig, MemoryConfig
from orchestrator.core.exceptions import AgentCreationError, OrchestratorError


class TestOrchestratorInitializer:
    """Tests for OrchestratorInitializer class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return OrchestratorConfig(
            name="TestOrchestrator",
            agents=[
                AgentConfig(
                    name="TestAgent",
                    role="Tester",
                    goal="Test things",
                    enabled=True
                )
            ]
        )

    @pytest.fixture
    def config_with_memory(self):
        """Configuration with memory enabled."""
        return OrchestratorConfig(
            name="TestOrchestrator",
            memory=MemoryConfig(
                provider="rag"
            ),
            agents=[
                AgentConfig(
                    name="TestAgent",
                    role="Tester",
                    goal="Test things",
                    enabled=True
                )
            ]
        )

    @pytest.fixture
    def initializer(self, basic_config):
        """Create initializer instance."""
        return OrchestratorInitializer(basic_config)

    def test_initializer_creation(self, basic_config):
        """Test initializer can be created."""
        initializer = OrchestratorInitializer(basic_config)
        assert initializer.config == basic_config
        assert initializer.agent_factory is not None
        assert initializer.task_factory is not None

    def test_initialize_memory_none(self, initializer):
        """Test memory initialization when config is None."""
        result = initializer.initialize_memory()
        assert result is None

    @patch('orchestrator.core.initializer.MemoryManager')
    def test_initialize_memory_success(self, mock_memory_manager, config_with_memory):
        """Test successful memory initialization."""
        initializer = OrchestratorInitializer(config_with_memory)
        mock_instance = Mock()
        mock_memory_manager.return_value = mock_instance

        result = initializer.initialize_memory()

        assert result == mock_instance
        mock_memory_manager.assert_called_once_with(config_with_memory.memory)

    @patch('orchestrator.core.initializer.MemoryManager')
    def test_initialize_memory_failure(self, mock_memory_manager, config_with_memory):
        """Test memory initialization failure handling."""
        initializer = OrchestratorInitializer(config_with_memory)
        mock_memory_manager.side_effect = Exception("Memory init failed")

        with pytest.raises(OrchestratorError) as exc_info:
            initializer.initialize_memory()

        assert "Memory initialization failed" in str(exc_info.value)

    @patch('orchestrator.core.initializer.WorkflowEngine')
    def test_initialize_workflow_engine(self, mock_workflow_engine, initializer):
        """Test workflow engine initialization."""
        mock_instance = Mock()
        mock_workflow_engine.return_value = mock_instance

        callback_start = Mock()
        callback_complete = Mock()
        callback_fail = Mock()
        callback_workflow_complete = Mock()

        result = initializer.initialize_workflow_engine(
            on_task_start=callback_start,
            on_task_complete=callback_complete,
            on_task_fail=callback_fail,
            on_workflow_complete=callback_workflow_complete
        )

        assert result == mock_instance
        assert mock_instance.on_task_start == callback_start
        assert mock_instance.on_task_complete == callback_complete
        assert mock_instance.on_task_fail == callback_fail
        assert mock_instance.on_workflow_complete == callback_workflow_complete

    @patch('orchestrator.core.initializer.WorkflowEngine')
    def test_initialize_workflow_engine_failure(self, mock_workflow_engine, initializer):
        """Test workflow engine initialization failure."""
        mock_workflow_engine.side_effect = Exception("Engine init failed")

        with pytest.raises(OrchestratorError) as exc_info:
            initializer.initialize_workflow_engine()

        assert "Workflow engine initialization failed" in str(exc_info.value)

    def test_create_agents_success(self, initializer):
        """Test successful agent creation."""
        with patch.object(initializer.agent_factory, 'create_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            result = initializer.create_agents()

            assert len(result) == 1
            assert "TestAgent" in result
            assert result["TestAgent"] == mock_agent

    def test_create_agents_skips_disabled(self):
        """Test that disabled agents are skipped."""
        config = OrchestratorConfig(
            agents=[
                AgentConfig(name="Enabled", role="Role1", goal="Goal1", enabled=True),
                AgentConfig(name="Disabled", role="Role2", goal="Goal2", enabled=False),
            ]
        )
        initializer = OrchestratorInitializer(config)

        with patch.object(initializer.agent_factory, 'create_agent') as mock_create:
            mock_create.return_value = Mock()
            result = initializer.create_agents()

            assert len(result) == 1
            assert "Enabled" in result
            assert "Disabled" not in result

    def test_create_agents_failure(self, initializer):
        """Test agent creation failure handling."""
        with patch.object(initializer.agent_factory, 'create_agent') as mock_create:
            mock_create.side_effect = Exception("Agent creation failed")

            with pytest.raises(AgentCreationError) as exc_info:
                initializer.create_agents()

            assert "Failed to create agents" in str(exc_info.value)

    def test_create_dynamic_tools_no_memory(self, initializer):
        """Test dynamic tool creation without memory."""
        result = initializer.create_dynamic_tools(None)
        assert result == {}

    def test_create_dynamic_tools_with_memory(self, initializer):
        """Test dynamic tool creation with memory."""
        mock_memory = Mock()
        mock_tool = Mock()
        mock_memory.create_graph_tool.return_value = mock_tool

        result = initializer.create_dynamic_tools(mock_memory)

        assert 'graph_rag_lookup' in result
        assert result['graph_rag_lookup'] == mock_tool

    def test_create_dynamic_tools_failure(self, initializer):
        """Test dynamic tool creation handles failures gracefully."""
        mock_memory = Mock()
        mock_memory.create_graph_tool.side_effect = Exception("Tool creation failed")

        # Should not raise, just log warning
        result = initializer.create_dynamic_tools(mock_memory)
        assert result == {}

    def test_enrich_agent_configs_with_tools(self):
        """Test agent config enrichment with dynamic tools."""
        config = OrchestratorConfig(
            agents=[
                AgentConfig(
                    name="Agent1",
                    role="Role1",
                    goal="Goal1",
                    instructions="Use graph_rag_lookup tool",
                    enabled=True,
                    tools=[]
                ),
                AgentConfig(
                    name="Agent2",
                    role="Role2",
                    goal="Goal2",
                    instructions="Regular agent",
                    enabled=True,
                    tools=[]
                )
            ]
        )
        initializer = OrchestratorInitializer(config)
        dynamic_tools = {'graph_rag_lookup': Mock()}

        result = initializer.enrich_agent_configs_with_tools(dynamic_tools)

        assert len(result) == 2
        assert 'graph_rag_lookup' in result[0].tools  # Agent1 should have it
        assert 'graph_rag_lookup' not in result[1].tools  # Agent2 should not

    def test_enrich_agent_configs_skips_disabled(self):
        """Test that disabled agents are skipped during enrichment."""
        config = OrchestratorConfig(
            agents=[
                AgentConfig(name="Enabled", role="R1", goal="G1", enabled=True),
                AgentConfig(name="Disabled", role="R2", goal="G2", enabled=False)
            ]
        )
        initializer = OrchestratorInitializer(config)

        result = initializer.enrich_agent_configs_with_tools({})

        assert len(result) == 1
        assert result[0].name == "Enabled"

    @patch('orchestrator.core.initializer.GraphFactory')
    def test_create_langgraph_system_with_tasks(self, mock_graph_factory, basic_config):
        """Test LangGraph system creation with tasks."""
        from orchestrator.core.config import TaskConfig

        basic_config.tasks = [
            TaskConfig(
                name="task1",
                description="Test task",
                expected_output="Output",
                agent_name="TestAgent"
            )
        ]
        initializer = OrchestratorInitializer(basic_config)

        mock_factory_instance = Mock()
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile.return_value = mock_compiled
        mock_factory_instance.create_workflow_graph.return_value = mock_graph
        mock_graph_factory.return_value = mock_factory_instance

        factory, compiled = initializer.create_langgraph_system(None)

        assert factory == mock_factory_instance
        assert compiled == mock_compiled
        mock_factory_instance.create_workflow_graph.assert_called_once()

    @patch('orchestrator.core.initializer.GraphFactory')
    def test_create_langgraph_system_without_tasks(self, mock_graph_factory, basic_config):
        """Test LangGraph system creation without tasks (chat mode)."""
        initializer = OrchestratorInitializer(basic_config)

        mock_factory_instance = Mock()
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile.return_value = mock_compiled
        mock_factory_instance.create_chat_graph.return_value = mock_graph
        mock_graph_factory.return_value = mock_factory_instance

        factory, compiled = initializer.create_langgraph_system(None)

        assert factory == mock_factory_instance
        assert compiled == mock_compiled
        mock_factory_instance.create_chat_graph.assert_called_once()

    @patch('orchestrator.core.initializer.GraphFactory')
    def test_create_langgraph_system_with_tools(self, mock_graph_factory, basic_config):
        """Test LangGraph system creation with dynamic tools."""
        initializer = OrchestratorInitializer(basic_config)
        mock_memory = Mock()
        mock_tool = Mock()
        mock_memory.create_graph_tool.return_value = mock_tool

        mock_factory_instance = Mock()
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile.return_value = mock_compiled
        mock_factory_instance.create_chat_graph.return_value = mock_graph
        mock_graph_factory.return_value = mock_factory_instance

        factory, compiled = initializer.create_langgraph_system(mock_memory)

        # Verify tools were registered
        mock_factory_instance.register_dynamic_tools.assert_called_once()
        call_args = mock_factory_instance.register_dynamic_tools.call_args[0][0]
        assert 'graph_rag_lookup' in call_args

    @patch('orchestrator.core.initializer.GraphFactory')
    def test_create_langgraph_system_failure(self, mock_graph_factory, basic_config):
        """Test LangGraph system creation failure handling."""
        initializer = OrchestratorInitializer(basic_config)
        mock_graph_factory.side_effect = Exception("Graph creation failed")

        with pytest.raises(OrchestratorError) as exc_info:
            initializer.create_langgraph_system(None)

        assert "Failed to create LangGraph system" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
