"""
Unit tests for OrchestratorExecutor class.

Tests the workflow execution logic, memory recall building,
and dynamic task planning functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Any, Optional

from orchestrator.core.executor import OrchestratorExecutor
from orchestrator.core.config import OrchestratorConfig, AgentConfig, TaskConfig
from orchestrator.core.exceptions import WorkflowError, AgentCreationError
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.workflow.workflow_engine import WorkflowEngine
from orchestrator.integrations.langchain_integration import (
    OrchestratorState,
    HumanMessage,
    AIMessage,
)


@pytest.fixture
def basic_config():
    """Create basic orchestrator configuration."""
    return OrchestratorConfig(
        name="test_orchestrator",
        user_id="test_user",
        max_iterations=10,
    )


@pytest.fixture
def agent_configs():
    """Create sample agent configurations."""
    return {
        "Researcher": AgentConfig(
            name="Researcher",
            role="Research Specialist",
            goal="Conduct thorough research",
            backstory="Expert researcher",
            instructions="Research thoroughly",
        ),
        "Analyst": AgentConfig(
            name="Analyst",
            role="Data Analyst",
            goal="Analyze data comprehensively",
            backstory="Experienced analyst",
            instructions="Analyze carefully",
        ),
    }


@pytest.fixture
def mock_compiled_graph():
    """Create mock compiled LangGraph."""
    graph = Mock()
    graph.invoke = Mock()
    return graph


@pytest.fixture
def mock_memory_manager():
    """Create mock memory manager."""
    memory = Mock(spec=MemoryManager)
    memory.retrieve = Mock(return_value=[])
    memory.retrieve_filtered = Mock(return_value=[])
    return memory


@pytest.fixture
def mock_workflow_engine():
    """Create mock workflow engine."""
    return Mock(spec=WorkflowEngine)


@pytest.fixture
def executor(basic_config, mock_compiled_graph, mock_memory_manager, mock_workflow_engine):
    """Create OrchestratorExecutor instance."""
    return OrchestratorExecutor(
        config=basic_config,
        compiled_graph=mock_compiled_graph,
        memory_manager=mock_memory_manager,
        workflow_engine=mock_workflow_engine,
    )


class TestOrchestratorExecutorInit:
    """Test OrchestratorExecutor initialization."""

    def test_init_with_all_parameters(self, basic_config, mock_compiled_graph, mock_memory_manager, mock_workflow_engine):
        """Test initialization with all parameters."""
        executor = OrchestratorExecutor(
            config=basic_config,
            compiled_graph=mock_compiled_graph,
            memory_manager=mock_memory_manager,
            workflow_engine=mock_workflow_engine,
        )

        assert executor.config == basic_config
        assert executor.compiled_graph == mock_compiled_graph
        assert executor.memory_manager == mock_memory_manager
        assert executor.workflow_engine == mock_workflow_engine

    def test_init_without_optional_parameters(self, basic_config, mock_compiled_graph):
        """Test initialization without optional parameters."""
        executor = OrchestratorExecutor(
            config=basic_config,
            compiled_graph=mock_compiled_graph,
        )

        assert executor.config == basic_config
        assert executor.compiled_graph == mock_compiled_graph
        assert executor.memory_manager is None
        assert executor.workflow_engine is None


class TestRunLanggraphWorkflow:
    """Test LangGraph workflow execution."""

    @pytest.mark.asyncio
    async def test_run_workflow_with_recall_content(self, executor, mock_compiled_graph):
        """Test workflow execution with recall content."""
        recall_content = "Memory context\nAnother line"

        # Mock result with final_output
        mock_result = Mock()
        mock_result.final_output = "Workflow completed successfully"
        mock_compiled_graph.invoke.return_value = mock_result

        result = await executor.run_langgraph_workflow(recall_content=recall_content)

        assert result == "Workflow completed successfully"
        mock_compiled_graph.invoke.assert_called_once()

        # Verify initial state construction
        call_args = mock_compiled_graph.invoke.call_args[0][0]
        assert isinstance(call_args, OrchestratorState)
        assert call_args.input_prompt == recall_content
        assert call_args.memory_context == recall_content
        assert call_args.max_iterations == 10
        assert call_args.recall_items == ["Memory context", "Another line"]

    @pytest.mark.asyncio
    async def test_run_workflow_without_recall_content(self, executor, mock_compiled_graph):
        """Test workflow execution without recall content."""
        mock_result = Mock()
        mock_result.final_output = "Success"
        mock_compiled_graph.invoke.return_value = mock_result

        result = await executor.run_langgraph_workflow()

        assert result == "Success"

        # Verify default prompt
        call_args = mock_compiled_graph.invoke.call_args[0][0]
        assert call_args.input_prompt == "Execute orchestrator workflow"
        assert call_args.recall_items == []

    @pytest.mark.asyncio
    async def test_run_workflow_raises_workflow_error_on_failure(self, executor, mock_compiled_graph):
        """Test workflow raises WorkflowError on execution failure."""
        mock_compiled_graph.invoke.side_effect = Exception("Graph execution failed")

        with pytest.raises(WorkflowError, match="LangGraph execution failed"):
            await executor.run_langgraph_workflow()


class TestExtractFinalOutput:
    """Test final output extraction from workflow results."""

    def test_extract_with_final_output_attribute(self, executor):
        """Test extraction when result has final_output attribute."""
        result = Mock()
        result.final_output = "Final output content"

        output = executor._extract_final_output(result)
        assert output == "Final output content"

    def test_extract_with_messages_containing_ai_message(self, executor):
        """Test extraction from messages list with AI message."""
        # Create a proper AIMessage mock that will be recognized
        ai_message = AIMessage(content="AI response")

        human_message = HumanMessage(content="User input")

        result = Mock()
        result.final_output = None
        result.messages = [human_message, ai_message, Mock()]

        output = executor._extract_final_output(result)
        assert output == "AI response"

    def test_extract_with_no_final_output_or_ai_message(self, executor):
        """Test extraction fallback to string representation."""
        result = Mock()
        result.final_output = None
        result.messages = []

        output = executor._extract_final_output(result)
        assert isinstance(output, str)


class TestBuildRecallContent:
    """Test memory recall content building."""

    def test_build_recall_without_memory_manager(self, basic_config, mock_compiled_graph):
        """Test recall building returns None without memory manager."""
        executor = OrchestratorExecutor(
            config=basic_config,
            compiled_graph=mock_compiled_graph,
            memory_manager=None,
        )

        result = executor.build_recall_content()
        assert result is None

    def test_build_recall_without_recall_config(self, executor):
        """Test recall building returns None without recall config."""
        result = executor.build_recall_content()
        assert result is None

    def test_build_recall_without_query(self, executor, basic_config):
        """Test recall building returns None without query."""
        executor.config.custom_config = {"recall": {"limit": 5}}

        result = executor.build_recall_content()
        assert result is None

    def test_build_recall_with_valid_config(self, executor, basic_config, mock_memory_manager):
        """Test successful recall building with valid configuration."""
        executor.config.custom_config = {
            "recall": {
                "query": "test query",
                "limit": 3,
                "agent_id": "researcher",
                "run_id": "run_001",
                "user_id": "custom_user",
                "rerank": True,
            }
        }

        mock_memory_manager.retrieve_filtered.return_value = [
            {"content": "Memory 1", "metadata": {"filename": "doc1.pdf"}},
            {"content": "Memory 2", "id": "mem2"},
            {"content": "Memory 3"},
        ]

        result = executor.build_recall_content()

        assert "MEMORY RECALL CONTEXT:" in result
        assert "Memory 1 [src: doc1.pdf]" in result
        assert "Memory 2 [src: mem2]" in result
        assert "Memory 3" in result

        mock_memory_manager.retrieve_filtered.assert_called_once_with(
            "test query",
            limit=3,
            user_id="custom_user",
            agent_id="researcher",
            run_id="run_001",
            rerank=True,
        )

    def test_build_recall_fallback_to_simple_retrieval(self, executor, basic_config, mock_memory_manager):
        """Test fallback to simple retrieval when filtered retrieval fails."""
        executor.config.custom_config = {"recall": {"query": "test", "limit": 2}}

        mock_memory_manager.retrieve_filtered.side_effect = Exception("Filter error")
        mock_memory_manager.retrieve.return_value = [
            {"content": "Fallback memory"},
        ]

        result = executor.build_recall_content()

        assert "Fallback memory" in result
        mock_memory_manager.retrieve.assert_called_once_with("test", limit=2)

    def test_build_recall_with_empty_results(self, executor, basic_config, mock_memory_manager):
        """Test recall building with empty results."""
        executor.config.custom_config = {"recall": {"query": "test"}}
        mock_memory_manager.retrieve_filtered.return_value = []

        result = executor.build_recall_content()
        assert result is None

    def test_build_recall_uses_top_k_as_fallback(self, executor, basic_config, mock_memory_manager):
        """Test that top_k is used as fallback for limit."""
        executor.config.custom_config = {"recall": {"query": "test", "top_k": 7}}
        mock_memory_manager.retrieve_filtered.return_value = [{"content": "test"}]

        executor.build_recall_content()

        call_args = mock_memory_manager.retrieve_filtered.call_args
        assert call_args.kwargs["limit"] == 7


class TestPlanFromPrompt:
    """Test dynamic task planning from prompt."""

    def test_plan_with_single_agent(self, executor, agent_configs):
        """Test planning with single agent."""
        tasks = executor.plan_from_prompt(
            prompt="Research AI trends",
            agent_sequence=["Researcher"],
            enabled_agents=agent_configs,
        )

        assert len(tasks) == 1
        assert tasks[0].name == "researcher_task_1"
        assert tasks[0].agent == "Researcher"
        assert tasks[0].is_start is True
        assert "Research AI trends" in tasks[0].description
        assert "Research Specialist" in tasks[0].description

    def test_plan_with_multiple_agents(self, executor, agent_configs):
        """Test planning with multiple agents in sequence."""
        tasks = executor.plan_from_prompt(
            prompt="Analyze market data",
            agent_sequence=["Researcher", "Analyst"],
            enabled_agents=agent_configs,
        )

        assert len(tasks) == 2
        assert tasks[0].name == "researcher_task_1"
        assert tasks[0].is_start is True
        assert tasks[0].context == []

        assert tasks[1].name == "analyst_task_2"
        assert tasks[1].is_start is False
        assert tasks[1].context == ["researcher_task_1"]

    def test_plan_with_assignments(self, executor, agent_configs):
        """Test planning with specific assignments."""
        assignments = [
            {
                "objective": "Find recent papers",
                "expected_output": "List of 10 papers",
                "tags": ["academic", "recent"],
            },
            {
                "description": "Analyze trends",
                "deliverable": "Trend report",
            },
        ]

        tasks = executor.plan_from_prompt(
            prompt="Research trends",
            agent_sequence=["Researcher", "Analyst"],
            enabled_agents=agent_configs,
            assignments=assignments,
        )

        assert "Find recent papers" in tasks[0].description
        assert "academic, recent" in tasks[0].description
        assert "List of 10 papers" in tasks[0].expected_output

        assert "Analyze trends" in tasks[1].description
        assert "Trend report" in tasks[1].expected_output

    def test_plan_with_recall_snippets(self, executor, agent_configs):
        """Test planning with recall snippets."""
        tasks = executor.plan_from_prompt(
            prompt="Continue research",
            agent_sequence=["Researcher"],
            enabled_agents=agent_configs,
            recall_snippets=["Previous finding 1", "Previous finding 2"],
        )

        assert "Contexto recuperado:" in tasks[0].description
        assert "Previous finding 1" in tasks[0].description
        assert "Previous finding 2" in tasks[0].description

    def test_plan_raises_error_for_empty_sequence(self, executor, agent_configs):
        """Test that planning raises error for empty agent sequence."""
        with pytest.raises(ValueError, match="agent_sequence must contain at least one agent"):
            executor.plan_from_prompt(
                prompt="Test",
                agent_sequence=[],
                enabled_agents=agent_configs,
            )

    def test_plan_raises_error_for_missing_agents(self, executor, agent_configs):
        """Test that planning raises error for unavailable agents."""
        with pytest.raises(AgentCreationError, match="Agents not available"):
            executor.plan_from_prompt(
                prompt="Test",
                agent_sequence=["Researcher", "NonExistentAgent"],
                enabled_agents=agent_configs,
            )


class TestStaticHelperMethods:
    """Test static helper methods."""

    def test_generate_task_name(self):
        """Test task name generation."""
        assert OrchestratorExecutor._generate_task_name("Researcher", 0) == "researcher_task_1"
        assert OrchestratorExecutor._generate_task_name("Data Analyst", 2) == "data_analyst_task_3"

    def test_compose_task_description_basic(self, agent_configs):
        """Test basic task description composition."""
        description = OrchestratorExecutor._compose_task_description(
            agent_cfg=agent_configs["Researcher"],
            prompt="Test prompt",
        )

        assert "Research Specialist" in description
        assert "Conduct thorough research" in description
        assert "Test prompt" in description

    def test_compose_task_description_with_all_parameters(self, agent_configs):
        """Test task description with all optional parameters."""
        description = OrchestratorExecutor._compose_task_description(
            agent_cfg=agent_configs["Analyst"],
            prompt="Analyze data",
            recall_snippets=["Context 1", "Context 2"],
            task_hint="analysis",
            assignment_objective="Deep dive analysis",
            assignment_tags=["urgent", "priority"],
        )

        assert "Contexto recuperado:" in description
        assert "Context 1" in description
        assert "Tipo de tarea sugerido: analysis" in description
        assert "Objetivo específico: Deep dive analysis" in description
        assert "urgent, priority" in description

    def test_compose_expected_output_basic(self, agent_configs):
        """Test basic expected output composition."""
        output = OrchestratorExecutor._compose_expected_output(
            agent_cfg=agent_configs["Researcher"],
            prompt="Research topic",
        )

        assert "Conduct thorough research" in output
        assert "Research topic" in output

    def test_compose_expected_output_with_deliverable(self, agent_configs):
        """Test expected output with deliverable."""
        output = OrchestratorExecutor._compose_expected_output(
            agent_cfg=agent_configs["Analyst"],
            prompt="Analyze",
            deliverable="Comprehensive report",
        )

        assert "Comprehensive report" in output
        assert "Analyze data comprehensively" in output
        assert "Analyze" in output

    def test_task_type_hint(self):
        """Test task type hint lookup."""
        assert OrchestratorExecutor._task_type_hint("Researcher") == "research"
        assert OrchestratorExecutor._task_type_hint("Analyst") == "analysis"
        assert OrchestratorExecutor._task_type_hint("Planner") == "planning"
        assert OrchestratorExecutor._task_type_hint("StandardsAgent") == "review"
        assert OrchestratorExecutor._task_type_hint("QuickResponder") == "documentation"
        assert OrchestratorExecutor._task_type_hint("UnknownAgent") is None
