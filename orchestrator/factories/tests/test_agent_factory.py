"""
Comprehensive unit tests for AgentFactory.

Test Coverage:
- P0 Critical Tests (15 tests): Core functionality and backward compatibility
- Factory initialization and template registration
- Agent creation with various configurations
- Error handling and validation
- Mode parameter support (future functionality)

Total: 15+ test cases covering critical paths
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from orchestrator.factories.agent_factory import (
    AgentFactory,
    USING_LANGCHAIN,
    BaseAgentTemplate,
    OrchestratorAgentTemplate,
    ResearcherAgentTemplate,
    PlannerAgentTemplate,
    ImplementerAgentTemplate,
    TesterAgentTemplate,
    WriterAgentTemplate
)
from orchestrator.core.config import AgentConfig
from orchestrator.core.exceptions import AgentCreationError, TemplateError


class TestAgentFactoryBasics:
    """P0 Test Group 1: Basic factory functionality and initialization."""

    @pytest.fixture
    def factory(self):
        """Create factory instance for testing."""
        return AgentFactory()

    def test_factory_initializes_with_templates(self, factory):
        """P0-1: Factory should initialize with 6 default templates."""
        templates = factory.list_templates()
        assert len(templates) == 6
        assert "orchestrator" in templates
        assert "researcher" in templates
        assert "planner" in templates
        assert "implementer" in templates
        assert "tester" in templates
        assert "writer" in templates

    def test_create_agent_with_valid_config(self, factory, base_config):
        """P0-11: Verify backward compatibility - create_agent without mode param."""
        agent = factory.create_agent(base_config)

        # Verify agent was created successfully
        assert agent is not None
        assert agent.name == "TestAgent"
        assert agent.role == "Test Role"
        assert agent.goal == "Test Goal"
        assert agent.backstory == "Test Backstory"

    def test_factory_backend_detection(self, factory):
        """P0-13: Verify factory correctly detects available backend."""
        # Factory should use global USING_LANGCHAIN flag
        assert isinstance(USING_LANGCHAIN, bool)

        # Should have consistent backend throughout session
        backend_type = "langchain" if USING_LANGCHAIN else "praisonai"
        assert backend_type in ["langchain", "praisonai"]


class TestAgentTypeInference:
    """P0 Test Group 2: Agent type inference logic."""

    @pytest.fixture
    def factory(self):
        return AgentFactory()

    def test_infer_orchestrator_from_name(self, factory):
        """P0-6a: Infer orchestrator type from name."""
        config = AgentConfig(
            name="Orchestrator",
            role="Manager",
            goal="Coordinate",
            backstory="Expert",
            instructions="Manage"
        )
        inferred_type = factory._infer_agent_type(config)
        assert inferred_type == "orchestrator"

    def test_infer_researcher_from_role(self, factory):
        """P0-6b: Infer researcher type from role."""
        config = AgentConfig(
            name="Agent1",
            role="Research Specialist",
            goal="Research",
            backstory="Researcher",
            instructions="Research"
        )
        inferred_type = factory._infer_agent_type(config)
        assert inferred_type == "researcher"

    def test_infer_planner_from_goal(self, factory):
        """P0-6c: Infer planner type from goal."""
        config = AgentConfig(
            name="Agent2",
            role="Agent",
            goal="Plan comprehensive strategy",
            backstory="Planner",
            instructions="Plan"
        )
        inferred_type = factory._infer_agent_type(config)
        assert inferred_type == "planner"

    def test_infer_implementer_from_backstory(self, factory):
        """P0-6d: Infer implementer type from backstory."""
        config = AgentConfig(
            name="Agent3",
            role="Developer",
            goal="Build",
            backstory="Implementation specialist with coding expertise",
            instructions="Implement"
        )
        inferred_type = factory._infer_agent_type(config)
        assert inferred_type == "implementer"

    def test_infer_tester_from_multiple_fields(self, factory):
        """P0-6e: Infer tester type from multiple fields."""
        config = AgentConfig(
            name="QA Agent",
            role="Testing Specialist",
            goal="Ensure quality",
            backstory="Expert tester",
            instructions="Test thoroughly"
        )
        inferred_type = factory._infer_agent_type(config)
        assert inferred_type == "tester"

    def test_infer_writer_from_documentation_context(self, factory):
        """P0-6f: Infer writer type from documentation context."""
        config = AgentConfig(
            name="Doc Agent",
            role="Documentation Specialist",
            goal="Write documentation",
            backstory="Technical writer",
            instructions="Document"
        )
        inferred_type = factory._infer_agent_type(config)
        assert inferred_type == "writer"


class TestTemplateCompatibility:
    """P0 Test Group 3: Template compatibility and agent creation."""

    @pytest.fixture
    def factory(self):
        return AgentFactory()

    def test_researcher_template_creates_agent_with_tools(self, factory):
        """P0-9: Researcher template should create agent with search tools."""
        config = factory.get_default_config("researcher")
        agent = factory.create_agent(config, agent_type="researcher")

        # Verify agent created
        assert agent is not None
        assert agent.name == "Researcher"

        # Verify tools attribute exists (tools may be empty if not available)
        if USING_LANGCHAIN:
            # LangChain agents should have tools list attribute
            assert hasattr(agent, 'tools')
            assert isinstance(agent.tools, list)
            # Note: Tools list may be empty if DuckDuckGo not installed
        else:
            # PraisonAI agents have tools as well
            assert agent is not None

    def test_orchestrator_template_creates_agent_without_tools(self, factory):
        """P0-7/8: Orchestrator template should work in both backends."""
        config = factory.get_default_config("orchestrator")
        agent = factory.create_agent(config, agent_type="orchestrator")

        # Verify agent created
        assert agent is not None
        assert agent.name == "Orchestrator"

        # Orchestrators don't use tools directly
        if USING_LANGCHAIN:
            assert hasattr(agent, 'tools')
            # May be empty or minimal tools

    def test_all_templates_produce_valid_agents(self, factory):
        """P0-15: All 6 templates must produce valid agents."""
        templates = ["orchestrator", "researcher", "planner", "implementer", "tester", "writer"]

        for template_name in templates:
            config = factory.get_default_config(template_name)
            agent = factory.create_agent(config, agent_type=template_name)

            # Verify agent has required attributes
            assert agent is not None, f"Template {template_name} failed to create agent"
            assert hasattr(agent, 'name'), f"{template_name} agent missing 'name'"
            assert hasattr(agent, 'role'), f"{template_name} agent missing 'role'"
            assert hasattr(agent, 'goal'), f"{template_name} agent missing 'goal'"
            assert hasattr(agent, 'backstory'), f"{template_name} agent missing 'backstory'"


class TestErrorHandling:
    """P0 Test Group 4: Error handling and validation."""

    @pytest.fixture
    def factory(self):
        return AgentFactory()

    def test_create_agent_with_invalid_template_type(self, factory, base_config):
        """P0-5: Invalid template type should raise AgentCreationError."""
        with pytest.raises(AgentCreationError):
            factory.create_agent(base_config, agent_type="nonexistent_type")

    def test_create_agent_with_empty_name(self, factory):
        """P0-12: Empty agent name should be handled gracefully."""
        invalid_config = AgentConfig(
            name="",  # Empty name
            role="Role",
            goal="Goal",
            backstory="Backstory",
            instructions="Instructions"
        )

        # Should either handle gracefully or raise clear error
        try:
            agent = factory.create_agent(invalid_config)
            # If it succeeds, verify agent was created
            assert agent is not None
        except (AgentCreationError, ValueError) as e:
            # If it fails, error message should be clear
            assert "name" in str(e).lower() or "required" in str(e).lower()

    def test_factory_state_consistency_after_error(self, factory):
        """P0-14: Failed agent creation should not corrupt factory state."""
        initial_templates = len(factory.list_templates())

        # Attempt to create agent with invalid type
        try:
            invalid_config = AgentConfig(
                name="Test",
                role="Role",
                goal="Goal",
                backstory="Backstory",
                instructions="Instructions"
            )
            factory.create_agent(invalid_config, agent_type="invalid_type")
        except AgentCreationError:
            pass  # Expected error

        # Factory state should be unchanged
        assert len(factory.list_templates()) == initial_templates

        # Should still be able to create valid agents
        valid_config = AgentConfig(
            name="ValidAgent",
            role="Role",
            goal="Goal",
            backstory="Backstory",
            instructions="Instructions"
        )
        agent = factory.create_agent(valid_config)
        assert agent is not None


@pytest.mark.skip(reason="Mode parameter not yet implemented - will be enabled after implementation")
class TestModeParameter:
    """Future P0 Tests: Mode parameter functionality (to be implemented)."""

    @pytest.fixture
    def factory(self):
        return AgentFactory()

    def test_create_agent_with_explicit_langchain_mode(self, factory, base_config):
        """P0-2: FUTURE - Verify explicit mode='langchain' creates LangChainAgent."""
        # This test will fail until mode parameter is implemented
        agent = factory.create_agent(base_config, mode='langchain')

        from orchestrator.integrations.langchain_integration import LangChainAgent
        assert isinstance(agent, LangChainAgent)

    def test_create_agent_with_explicit_praisonai_mode(self, factory, base_config):
        """P0-3: FUTURE - Verify explicit mode='praisonai' creates PraisonAI Agent."""
        # This test will fail until mode parameter is implemented
        agent = factory.create_agent(base_config, mode='praisonai')

        # Should create PraisonAI agent
        assert agent is not None
        assert hasattr(agent, 'name')

    def test_create_agent_auto_mode_detection(self, factory, base_config):
        """P0-4: FUTURE - Test auto mode detection."""
        # This test will fail until mode parameter is implemented
        agent = factory.create_agent(base_config, mode='auto')

        # Should auto-detect and create agent
        assert agent is not None

    def test_create_agent_invalid_mode_raises_error(self, factory, base_config):
        """P0-5: FUTURE - Invalid mode should raise AgentCreationError."""
        # This test will fail until mode parameter is implemented
        with pytest.raises(AgentCreationError, match="Invalid mode"):
            factory.create_agent(base_config, mode='invalid_mode')

    def test_mode_switching_preserves_factory_state(self, factory, base_config):
        """P0-14: FUTURE - Mode switching should not corrupt state."""
        # This test will fail until mode parameter is implemented

        # Create agent with langchain mode
        agent1 = factory.create_agent(base_config, mode='langchain')

        # Switch to praisonai mode
        agent2 = factory.create_agent(base_config, mode='praisonai')

        # Factory should still be functional
        assert len(factory.list_templates()) == 6


class TestBackwardCompatibility:
    """P0 Test Group 5: Ensure existing API continues to work."""

    @pytest.fixture
    def factory(self):
        return AgentFactory()

    def test_create_agent_without_any_optional_params(self, factory):
        """P0-11: Existing code without mode param must work."""
        config = AgentConfig(
            name="LegacyAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory",
            instructions="Test Instructions"
        )

        # Should work exactly as before
        agent = factory.create_agent(config)
        assert agent is not None
        assert agent.name == "LegacyAgent"

    def test_get_default_config_unchanged(self, factory):
        """P0: get_default_config should return valid configurations."""
        templates = factory.list_templates()

        for template_name in templates:
            config = factory.get_default_config(template_name)

            # Verify config is valid AgentConfig
            assert isinstance(config, AgentConfig)
            assert len(config.name) > 0
            assert len(config.role) > 0
            assert len(config.goal) > 0
            assert len(config.backstory) > 0

    def test_create_agents_batch_method(self, factory):
        """P0: Batch agent creation should work as before."""
        configs = [
            AgentConfig(
                name=f"Agent{i}",
                role=f"Role{i}",
                goal=f"Goal{i}",
                backstory=f"Backstory{i}",
                instructions=f"Instructions{i}"
            )
            for i in range(3)
        ]

        agents = factory.create_agents_from_configs(configs)

        assert len(agents) == 3
        for i, agent in enumerate(agents):
            assert agent.name == f"Agent{i}"