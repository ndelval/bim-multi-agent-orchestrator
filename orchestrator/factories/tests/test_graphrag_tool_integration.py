"""
Integration tests for GraphRAG tool attachment to agent_configs in LangGraph workflows.

This test suite validates the complete integration flow:
1. GraphRAG tool creation from memory manager
2. Tool attachment to agent configurations
3. LangGraph StateGraph compilation with tools
4. Agent execution with tool access
5. Error handling and edge cases

Test execution:
    uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any, Optional

from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.config import (
    OrchestratorConfig,
    AgentConfig,
    MemoryConfig,
    MemoryProvider,
)
from orchestrator.core.exceptions import (
    OrchestratorError,
    ConfigurationError,
    AgentCreationError,
    GraphCreationError,
)
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.tools.graph_rag_tool import GraphRAGTool, create_graph_rag_tool

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def _check_langgraph_available() -> bool:
    """Check if LangGraph is available for testing."""
    try:
        from langgraph.graph import StateGraph

        return True
    except ImportError:
        return False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager with graph tool capabilities."""
    manager = Mock(spec=MemoryManager)
    manager.config = Mock()
    manager.provider = Mock()

    # Mock retrieve_with_graph method
    manager.retrieve_with_graph = Mock(
        return_value=[
            {
                "content": "Test document content about safety standards",
                "metadata": {
                    "document_id": "doc_001",
                    "section": "Section 1.1",
                    "source_url": "https://example.com/doc001",
                },
                "score": 0.95,
            },
            {
                "content": "Additional context about compliance requirements",
                "metadata": {"document_id": "doc_002", "section": "Section 2.3"},
                "score": 0.87,
            },
        ]
    )

    # Mock create_graph_tool method
    def mock_create_graph_tool(default_user_id=None, default_run_id=None):
        return create_graph_rag_tool(
            manager, default_user_id=default_user_id, default_run_id=default_run_id
        )

    manager.create_graph_tool = Mock(side_effect=mock_create_graph_tool)
    manager.get_provider_info = Mock(
        return_value={"provider": "hybrid", "status": "ready"}
    )
    manager.cleanup = Mock()

    return manager


@pytest.fixture
def base_memory_config():
    """Create base memory configuration for hybrid provider."""
    return MemoryConfig(
        provider=MemoryProvider.HYBRID,
        use_embedding=True,
        config={
            "vector_path": ".praison/test_vector",
            "lexical_db": ".praison/test_lexical.db",
            "neo4j_url": "bolt://localhost:7687",
        },
    )


@pytest.fixture
def agent_config_with_tools():
    """Create agent configuration with tools list."""
    return AgentConfig(
        name="TestAgent",
        role="Research Specialist",
        goal="Gather and analyze information",
        backstory="Expert researcher with access to document databases",
        instructions="Use available tools to retrieve relevant information",
        tools=[],  # Will be populated dynamically
        llm="gpt-4o-mini",
        enabled=True,
    )


@pytest.fixture
def orchestrator_config_base(base_memory_config, agent_config_with_tools):
    """Create base orchestrator configuration."""
    return OrchestratorConfig(
        name="GraphRAGTestOrchestrator",
        memory=base_memory_config,
        process="workflow",
        user_id="test_user_001",
        verbose=1,
        agents=[agent_config_with_tools],
        tasks=[],
    )


# ============================================================================
# Test Class: Basic Tool Creation
# ============================================================================


class TestGraphRAGToolCreation:
    """Test suite for GraphRAG tool creation and validation."""

    def test_basic_tool_creation(self, mock_memory_manager):
        """Test basic GraphRAG tool creation from memory manager."""
        # Create tool
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Validate tool properties (StructuredTool uses .name and .description)
        assert callable(tool)
        assert hasattr(tool, "name")
        assert tool.name == "graph_rag_lookup"
        assert hasattr(tool, "description")
        assert "documents" in tool.description or "hybrid" in tool.description

        logger.info("Basic tool creation successful")

    def test_tool_execution_basic(self, mock_memory_manager):
        """Test basic tool execution with query."""
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Execute tool via invoke (StructuredTool API)
        result = tool.invoke(
            {"query": "safety standards for industrial equipment", "top_k": 5}
        )

        # Validate results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "GraphRAG" in result or "document" in result

        # Verify memory manager was called
        mock_memory_manager.retrieve_with_graph.assert_called_once()
        call_args = mock_memory_manager.retrieve_with_graph.call_args
        assert call_args[0][0] == "safety standards for industrial equipment"

        logger.info("Tool execution successful")

    def test_tool_execution_with_filters(self, mock_memory_manager):
        """Test tool execution with filtering parameters."""
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Execute with filters via invoke
        result = tool.invoke(
            {
                "query": "compliance requirements",
                "tags": "safety,industrial",
                "documents": "doc_001,doc_002",
                "sections": "Section 1.1",
                "top_k": 3,
            }
        )

        # Validate call
        call_args = mock_memory_manager.retrieve_with_graph.call_args
        kwargs = call_args[1]

        assert kwargs["tags"] == ["safety", "industrial"]
        assert kwargs["document_ids"] == ["doc_001", "doc_002"]
        assert kwargs["sections"] == ["Section 1.1"]
        assert kwargs["limit"] == 3

        logger.info("Tool filtering successful")

    def test_tool_without_memory_manager(self, orchestrator_config_base):
        """Test error handling when memory manager is not initialized."""
        config = orchestrator_config_base
        config.memory = None

        orchestrator = Orchestrator(config)

        # Should raise error when trying to create tool without memory manager
        with pytest.raises(Exception) as exc_info:
            orchestrator.create_graph_tool()

        assert "Memory" in str(exc_info.value) or "not initialized" in str(
            exc_info.value
        )

        logger.info("No memory manager error handling successful")


# ============================================================================
# Test Class: Tool Attachment to Agent Configs
# ============================================================================


class TestToolAttachmentToAgentConfigs:
    """Test suite for attaching GraphRAG tools to agent configurations."""

    def test_attach_tool_to_single_agent(
        self, mock_memory_manager, agent_config_with_tools
    ):
        """Test attaching tool to a single agent configuration."""
        # Create tool
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Attach to agent config
        agent_config_with_tools.tools = [tool]

        # Validate attachment
        assert len(agent_config_with_tools.tools) == 1
        assert callable(agent_config_with_tools.tools[0])

        logger.info("Single agent tool attachment successful")

    def test_attach_tool_to_multiple_agents(self, mock_memory_manager):
        """Test attaching tool to multiple agent configurations."""
        # Create tool
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Create multiple agent configs
        agents = [
            AgentConfig(
                name=f"Agent{i}",
                role=f"Role{i}",
                goal=f"Goal{i}",
                backstory=f"Backstory{i}",
                instructions=f"Instructions{i}",
                tools=[tool],
                enabled=True,
            )
            for i in range(3)
        ]

        # Validate all agents have tool
        for agent in agents:
            assert len(agent.tools) == 1
            assert callable(agent.tools[0])

        logger.info("Multiple agent tool attachment successful")

    def test_attach_multiple_tools_to_agent(
        self, mock_memory_manager, agent_config_with_tools
    ):
        """Test attaching multiple tools including GraphRAG to agent."""
        # Create tools
        graph_tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Create mock additional tool
        other_tool = Mock()
        other_tool.__name__ = "other_tool"

        # Attach multiple tools
        agent_config_with_tools.tools = [graph_tool, other_tool]

        # Validate
        assert len(agent_config_with_tools.tools) == 2
        assert callable(agent_config_with_tools.tools[0])
        assert callable(agent_config_with_tools.tools[1])

        logger.info("Multiple tool attachment successful")

    def test_tool_config_validation(self, mock_memory_manager, agent_config_with_tools):
        """Test that invalid tool names are silently skipped during agent creation.

        AgentFactory._resolve_tools logs unrecognised string tool names and
        drops them rather than raising, so the agent is created with an empty
        tool list.
        """
        agent_config_with_tools.tools = ["not_a_tool"]

        from orchestrator.factories.agent_factory import AgentFactory

        factory = AgentFactory()

        # Unknown string tool names are silently skipped
        agent = factory.create_agent(agent_config_with_tools)
        assert agent is not None
        # The invalid string should have been filtered out
        assert len(agent.tools) == 0

        logger.info("Tool validation successful")


# ============================================================================
# Test Class: LangGraph Integration
# ============================================================================


class TestLangGraphWorkflowWithTools:
    """Test suite for LangGraph StateGraph compilation with tools."""

    @pytest.mark.skipif(
        not _check_langgraph_available(), reason="LangGraph not available"
    )
    def test_graph_compilation_with_tools(
        self, mock_memory_manager, orchestrator_config_base
    ):
        """Test StateGraph compilation with tool-enabled agents."""
        # Create and attach tool
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )
        orchestrator_config_base.agents[0].tools = [tool]

        # Mock orchestrator with memory manager
        with patch.object(Orchestrator, "_create_langgraph_system") as mock_create:
            orchestrator = Orchestrator(orchestrator_config_base)
            orchestrator.memory_manager = mock_memory_manager

            # Initialize should create graph system
            orchestrator.initialize()

            # Verify graph creation was attempted
            mock_create.assert_called_once()

        logger.info("Graph compilation with tools successful")

    @pytest.mark.skipif(
        not _check_langgraph_available(), reason="LangGraph not available"
    )
    def test_agent_node_with_tool_access(
        self, mock_memory_manager, orchestrator_config_base
    ):
        """Test agent configuration preserves tool access."""
        # Create and attach tool
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )
        orchestrator_config_base.agents[0].tools = [tool]

        # Verify tools are preserved in config
        assert len(orchestrator_config_base.agents[0].tools) == 1
        assert callable(orchestrator_config_base.agents[0].tools[0])
        assert orchestrator_config_base.agents[0].tools[0].name == "graph_rag_lookup"

        logger.info("Agent node tool access successful")

    def test_graph_state_with_tool_results(self, mock_memory_manager):
        """Test StateGraph state management with tool execution results."""
        from orchestrator.integrations.langchain_integration import OrchestratorState

        # Create initial state
        state = OrchestratorState(
            input_prompt="Find safety standards", max_iterations=5
        )

        # Simulate tool execution
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )
        tool_result = tool.invoke({"query": "safety standards", "top_k": 3})

        # Update state with tool result
        state.node_outputs["graph_rag_lookup"] = tool_result
        state.execution_path.append(
            "graph_rag_lookup"
        )  # Track execution instead of iteration

        # Validate state
        assert "graph_rag_lookup" in state.node_outputs
        assert len(state.node_outputs["graph_rag_lookup"]) > 0
        assert (
            state.execution_depth == 1
        )  # Use derived property instead of current_iteration

        logger.info("Graph state with tool results successful")


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCasesAndErrors:
    """Test suite for edge cases and error scenarios."""

    def test_tool_execution_empty_results(self, mock_memory_manager):
        """Test tool behavior with empty results from memory."""
        # Configure mock to return empty results
        mock_memory_manager.retrieve_with_graph = Mock(return_value=[])

        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        result = tool.invoke({"query": "nonexistent query", "top_k": 5})

        # Should return friendly message
        assert isinstance(result, str)
        assert "no" in result.lower() or "not" in result.lower()

        logger.info("Empty results handling successful")

    def test_tool_execution_memory_error(self, mock_memory_manager):
        """Test tool behavior when memory retrieval fails."""
        # Configure mock to raise error
        mock_memory_manager.retrieve_with_graph = Mock(
            side_effect=Exception("Memory retrieval failed")
        )

        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Should raise or handle gracefully
        with pytest.raises(Exception) as exc_info:
            tool.invoke({"query": "test query", "top_k": 5})

        assert "Memory retrieval failed" in str(exc_info.value)

        logger.info("Memory error handling successful")

    @pytest.mark.skip(
        reason="LangGraph is always available - PraisonAI fallback removed"
    )
    def test_graph_compilation_without_langgraph(
        self, mock_memory_manager, orchestrator_config_base
    ):
        """Test fallback behavior when LangGraph is unavailable."""
        pass

    def test_tool_with_invalid_parameters(self, mock_memory_manager):
        """Test GraphRAGTool behavior with invalid top_k type.

        Note: Tests the underlying GraphRAGTool directly since StructuredTool
        validates input via Pydantic schema before the function is called.
        """
        # Test GraphRAGTool directly (bypassing StructuredTool Pydantic validation)
        tool_instance = GraphRAGTool(
            mock_memory_manager,
            default_user_id="test_user",
            default_run_id="test_run",
        )

        # Test with invalid top_k - should gracefully default to 5
        result = tool_instance(query="test", top_k="invalid")

        mock_memory_manager.retrieve_with_graph.assert_called()
        call_kwargs = mock_memory_manager.retrieve_with_graph.call_args[1]
        assert call_kwargs["limit"] == 5

        logger.info("Invalid parameter handling successful")

    def test_tool_with_very_large_top_k(self, mock_memory_manager):
        """Test that unreasonably large top_k is rejected by validation."""
        from pydantic import ValidationError

        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # top_k > 50 should be rejected by GraphRAGInput validation (BP-AGENT-10)
        with pytest.raises(ValidationError, match="less than or equal to 50"):
            tool.invoke({"query": "test", "top_k": 10000})

        logger.info("Large top_k validation successful")

    def test_concurrent_tool_execution(self, mock_memory_manager):
        """Test concurrent execution of tool from multiple agents."""
        import threading

        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        results = []
        errors = []

        def execute_tool(query_id):
            try:
                result = tool.invoke({"query": f"query_{query_id}", "top_k": 3})
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Execute concurrently
        threads = [threading.Thread(target=execute_tool, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Validate
        assert len(results) == 5
        assert len(errors) == 0
        assert mock_memory_manager.retrieve_with_graph.call_count == 5

        logger.info("Concurrent execution successful")

    def test_tool_with_missing_metadata(self, mock_memory_manager):
        """Test tool behavior when results have incomplete metadata."""
        # Configure mock with incomplete metadata
        mock_memory_manager.retrieve_with_graph = Mock(
            return_value=[{"content": "Content without metadata", "score": 0.8}]
        )

        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        result = tool.invoke({"query": "test", "top_k": 5})

        # Should handle gracefully with defaults
        assert isinstance(result, str)
        assert "unknown" in result.lower()

        logger.info("Missing metadata handling successful")


# ============================================================================
# Test Class: Integration Flow End-to-End
# ============================================================================


class TestIntegrationFlowEndToEnd:
    """Test suite for complete integration flow."""

    def test_complete_workflow_with_graphrag_tool(
        self, mock_memory_manager, orchestrator_config_base
    ):
        """Test complete workflow: config -> tool creation -> graph compilation -> execution."""
        # Step 1: Create orchestrator with memory
        orchestrator = Orchestrator(orchestrator_config_base)
        orchestrator.memory_manager = mock_memory_manager

        # Step 2: Create and attach GraphRAG tool
        tool = orchestrator.create_graph_tool(user_id="test_user", run_id="test_run")
        orchestrator.config.agents[0].tools = [tool]

        # Step 3: Validate tool attachment
        assert len(orchestrator.config.agents[0].tools) == 1

        # Step 4: Test tool execution via invoke
        result = tool.invoke({"query": "test query", "top_k": 3})
        assert isinstance(result, str)
        assert len(result) > 0

        # Step 5: Verify memory manager interaction
        mock_memory_manager.retrieve_with_graph.assert_called()
        mock_memory_manager.create_graph_tool.assert_called_once()

        logger.info("Complete workflow integration successful")

    def test_multi_agent_workflow_with_shared_tool(self, mock_memory_manager):
        """Test workflow with multiple agents sharing GraphRAG tool."""
        # Create tool
        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Create multiple agents with shared tool
        agents = [
            AgentConfig(
                name=f"Agent{i}",
                role=f"Specialist{i}",
                goal=f"Execute task {i}",
                backstory=f"Expert {i}",
                instructions=f"Execute task {i} using available tools",
                tools=[tool],
                enabled=True,
            )
            for i in range(3)
        ]

        # Create config and orchestrator
        config = OrchestratorConfig(
            name="MultiAgentTest",
            agents=agents,
            process="workflow",
            user_id="test_user",
        )

        orchestrator = Orchestrator(config)
        orchestrator.memory_manager = mock_memory_manager

        # Validate all agents have tool access
        for agent in orchestrator.config.agents:
            assert len(agent.tools) == 1
            assert callable(agent.tools[0])

        logger.info("Multi-agent shared tool workflow successful")


# ============================================================================
# Performance and Stress Tests
# ============================================================================


class TestPerformanceAndStress:
    """Test suite for performance and stress scenarios."""

    def test_tool_execution_performance(self, mock_memory_manager):
        """Test tool execution performance with multiple queries."""
        import time

        tool = mock_memory_manager.create_graph_tool(
            default_user_id="test_user", default_run_id="test_run"
        )

        # Execute multiple queries
        start_time = time.time()
        for i in range(10):
            result = tool.invoke({"query": f"query_{i}", "top_k": 5})
            assert isinstance(result, str)

        elapsed_time = time.time() - start_time

        # Should complete reasonably fast (< 1 second for mocked calls)
        assert elapsed_time < 1.0
        assert mock_memory_manager.retrieve_with_graph.call_count == 10

        logger.info(f"Performance test successful: {elapsed_time:.3f}s for 10 queries")

    def test_memory_manager_cleanup(
        self, mock_memory_manager, orchestrator_config_base
    ):
        """Test proper cleanup of memory manager and tools."""
        orchestrator = Orchestrator(orchestrator_config_base)
        orchestrator.memory_manager = mock_memory_manager

        # Create tool
        tool = orchestrator.create_graph_tool(user_id="test_user", run_id="test_run")

        # Execute tool
        result = tool.invoke({"query": "test", "top_k": 3})
        assert isinstance(result, str)

        # Cleanup
        orchestrator.cleanup()

        # Verify cleanup was called (cleanup() calls it once, then reset()
        # calls it again, so we verify it was called at least once)
        mock_memory_manager.cleanup.assert_called()

        logger.info("Cleanup test successful")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
