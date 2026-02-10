"""
Tests for ToT planner prompt wrapper fixes.

Validates that the prompt wrapper methods work correctly without
confusing NotImplementedError exceptions, and that dynamic method
assignment integrates properly with the Tree-of-Thought library.
"""

import pytest
from orchestrator.planning.tot_planner import OrchestratorPlanningTask
from orchestrator.core.config import AgentConfig


class TestPromptWrapperFixes:
    """Test prompt wrapper method implementations."""

    @pytest.fixture
    def agent_catalog(self):
        """Create sample agent catalog for testing."""
        return [
            AgentConfig(
                name="Researcher",
                role="Research Specialist",
                goal="Gather comprehensive information",
                backstory="Expert at finding and synthesizing information"
            ),
            AgentConfig(
                name="Analyst",
                role="Data Analyst",
                goal="Analyze patterns and insights",
                backstory="Skilled at identifying trends and anomalies"
            ),
        ]

    @pytest.fixture
    def planning_task_standard(self, agent_catalog):
        """Create OrchestratorPlanningTask with standard prompt style."""
        return OrchestratorPlanningTask(
            problem_statement="Test problem",
            agent_catalog=agent_catalog,
            max_steps=3,
            prompt_style="standard"
        )

    @pytest.fixture
    def planning_task_cot(self, agent_catalog):
        """Create OrchestratorPlanningTask with CoT prompt style."""
        return OrchestratorPlanningTask(
            problem_statement="Test problem",
            agent_catalog=agent_catalog,
            max_steps=3,
            prompt_style="cot"
        )

    def test_standard_prompt_wrap_exists(self, planning_task_standard):
        """Test that standard_prompt_wrap method exists and is callable."""
        assert hasattr(planning_task_standard, "standard_prompt_wrap")
        assert callable(planning_task_standard.standard_prompt_wrap)

    def test_cot_prompt_wrap_exists(self, planning_task_cot):
        """Test that cot_prompt_wrap method exists and is callable."""
        assert hasattr(planning_task_cot, "cot_prompt_wrap")
        assert callable(planning_task_cot.cot_prompt_wrap)

    def test_standard_prompt_wrap_returns_string(self, planning_task_standard):
        """Test standard_prompt_wrap returns valid string prompt."""
        prompt = planning_task_standard.standard_prompt_wrap(
            x="Analyze system architecture",
            y=""
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Analyze system architecture" in prompt

    def test_cot_prompt_wrap_returns_string(self, planning_task_cot):
        """Test cot_prompt_wrap returns valid string prompt with CoT guidance."""
        prompt = planning_task_cot.cot_prompt_wrap(
            x="Design microservices architecture",
            y=""
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Design microservices architecture" in prompt
        # Should include CoT guidance
        assert "paso a paso" in prompt.lower()

    def test_standard_prompt_includes_problem(self, planning_task_standard):
        """Test standard prompt includes problem statement."""
        problem = "Build authentication system"
        prompt = planning_task_standard.standard_prompt_wrap(x=problem, y="")

        assert problem in prompt

    def test_standard_prompt_includes_agents(self, planning_task_standard):
        """Test standard prompt includes agent information."""
        prompt = planning_task_standard.standard_prompt_wrap(
            x="Test problem",
            y=""
        )

        # Should list available agents
        assert "Researcher" in prompt
        assert "Analyst" in prompt

    def test_cot_prompt_includes_cot_guidance(self, planning_task_cot):
        """Test CoT prompt includes chain-of-thought guidance."""
        prompt = planning_task_cot.cot_prompt_wrap(
            x="Optimize database queries",
            y=""
        )

        # Should include thinking guidance in Spanish
        assert "paso a paso" in prompt.lower() or "piensa" in prompt.lower()

    def test_prompt_with_partial_plan(self, planning_task_standard):
        """Test prompt generation with partial plan context."""
        partial_plan = "Agent: Researcher | Objective: Gather requirements"
        prompt = planning_task_standard.standard_prompt_wrap(
            x="Build feature",
            y=partial_plan
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_no_notimplementederror_on_standard(self, planning_task_standard):
        """Test standard_prompt_wrap does not raise NotImplementedError."""
        try:
            prompt = planning_task_standard.standard_prompt_wrap(x="Test", y="")
            assert isinstance(prompt, str)
        except NotImplementedError as e:
            pytest.fail(f"standard_prompt_wrap should not raise NotImplementedError: {e}")

    def test_no_notimplementederror_on_cot(self, planning_task_cot):
        """Test cot_prompt_wrap does not raise NotImplementedError."""
        try:
            prompt = planning_task_cot.cot_prompt_wrap(x="Test", y="")
            assert isinstance(prompt, str)
        except NotImplementedError as e:
            pytest.fail(f"cot_prompt_wrap should not raise NotImplementedError: {e}")

    def test_init_prompt_wrappers_called(self, agent_catalog):
        """Test _init_prompt_wrappers is called during initialization."""
        # This should complete without errors
        task = OrchestratorPlanningTask(
            problem_statement="Test",
            agent_catalog=agent_catalog,
            max_steps=2,
            prompt_style="cot"
        )

        # Verify methods are accessible
        assert hasattr(task, "standard_prompt_wrap")
        assert hasattr(task, "cot_prompt_wrap")

    def test_prompt_style_standard(self, agent_catalog):
        """Test prompt style selection for standard mode."""
        task = OrchestratorPlanningTask(
            problem_statement="Test",
            agent_catalog=agent_catalog,
            max_steps=2,
            prompt_style="standard"
        )

        assert task.prompt_style == "standard"

        # Should still work without CoT guidance
        prompt = task.standard_prompt_wrap(x="Test", y="")
        assert "paso a paso" not in prompt.lower() or "piensa" not in prompt.lower()

    def test_prompt_style_cot(self, agent_catalog):
        """Test prompt style selection for CoT mode."""
        task = OrchestratorPlanningTask(
            problem_statement="Test",
            agent_catalog=agent_catalog,
            max_steps=2,
            prompt_style="cot"
        )

        assert task.prompt_style == "cot"

    def test_prompt_style_fallback(self, agent_catalog):
        """Test prompt style falls back to 'cot' for invalid values."""
        task = OrchestratorPlanningTask(
            problem_statement="Test",
            agent_catalog=agent_catalog,
            max_steps=2,
            prompt_style="invalid_style"
        )

        # Should fallback to 'cot'
        assert task.prompt_style == "cot"

    def test_method_binding_preserved(self, planning_task_standard):
        """Test that methods remain properly bound to instance."""
        # Call multiple times to ensure binding is stable
        prompt1 = planning_task_standard.standard_prompt_wrap(x="Test1", y="")
        prompt2 = planning_task_standard.standard_prompt_wrap(x="Test2", y="")

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)
        assert "Test1" in prompt1
        assert "Test2" in prompt2

    def test_docstrings_present(self, planning_task_standard):
        """Test that methods have proper docstrings."""
        assert planning_task_standard.standard_prompt_wrap.__doc__ is not None
        assert planning_task_standard.cot_prompt_wrap.__doc__ is not None

        # Docstrings should explain the dynamic assignment behavior
        standard_doc = planning_task_standard.standard_prompt_wrap.__doc__
        assert "prompt" in standard_doc.lower()

    def test_base_prompt_helper(self, planning_task_standard):
        """Test _base_prompt helper method."""
        base_prompt = planning_task_standard._base_prompt(
            x="Build API",
            y=""
        )

        assert isinstance(base_prompt, str)
        assert "Build API" in base_prompt
        assert "Agentes disponibles" in base_prompt

    def test_wrap_prompt_helper(self, planning_task_standard):
        """Test _wrap_prompt helper method."""
        wrapped = planning_task_standard._wrap_prompt(
            x="Test problem",
            y="partial plan"
        )

        assert isinstance(wrapped, str)
        assert len(wrapped) > 0


class TestPromptWrapperIntegration:
    """Integration tests for prompt wrappers with ToT library expectations."""

    @pytest.fixture
    def agent_catalog(self):
        """Create sample agent catalog for testing."""
        return [
            AgentConfig(
                name="Planner",
                role="Strategic Planner",
                goal="Create comprehensive plans",
                backstory="Expert at breaking down complex problems"
            ),
        ]

    def test_propose_prompt_wrap_fallback(self, agent_catalog):
        """Test propose_prompt_wrap uses _wrap_prompt as fallback."""
        task = OrchestratorPlanningTask(
            problem_statement="Test",
            agent_catalog=agent_catalog,
            max_steps=2,
            prompt_style="standard"
        )

        prompt = task.propose_prompt_wrap(x="Propose test", y="")

        assert isinstance(prompt, str)
        assert "Propose test" in prompt

    def test_multiple_instances_independent(self, agent_catalog):
        """Test that multiple task instances are independent."""
        task1 = OrchestratorPlanningTask(
            problem_statement="Problem 1",
            agent_catalog=agent_catalog,
            max_steps=2,
            prompt_style="standard"
        )

        task2 = OrchestratorPlanningTask(
            problem_statement="Problem 2",
            agent_catalog=agent_catalog,
            max_steps=3,
            prompt_style="cot"
        )

        prompt1 = task1.standard_prompt_wrap(x="Test1", y="")
        prompt2 = task2.cot_prompt_wrap(x="Test2", y="")

        assert "Test1" in prompt1
        assert "Test2" in prompt2
        assert task1.prompt_style == "standard"
        assert task2.prompt_style == "cot"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
