"""
Shared fixtures and configuration for AgentFactory tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import os
from orchestrator.core.config import AgentConfig


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    # Set dummy OpenAI API key for tests to avoid LangChain errors
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-testing-only"
    yield
    # Cleanup after all tests
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-test"):
        del os.environ["OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def mock_langchain_llm(monkeypatch):
    """Mock LangChain's ChatOpenAI to avoid actual API calls in tests."""
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value="Mocked LLM response")
    mock_llm.ainvoke = Mock(return_value="Mocked async LLM response")

    # Mock the ChatOpenAI class
    def mock_chatopenai(*args, **kwargs):
        return mock_llm

    try:
        monkeypatch.setattr("orchestrator.integrations.langchain_integration.ChatOpenAI", mock_chatopenai)
    except:
        pass  # If import fails, skip mocking (PraisonAI mode)

    return mock_llm


@pytest.fixture
def base_config():
    """Standard agent configuration for tests."""
    return AgentConfig(
        name="TestAgent",
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory",
        instructions="Test Instructions",
        tools=[]
    )


@pytest.fixture
def researcher_config():
    """Researcher agent configuration."""
    return AgentConfig(
        name="Researcher",
        role="Research Specialist",
        goal="Gather comprehensive information",
        backstory="Expert researcher",
        instructions="Research thoroughly",
        tools=["duckduckgo"]
    )


@pytest.fixture
def orchestrator_config():
    """Orchestrator agent configuration."""
    return AgentConfig(
        name="Orchestrator",
        role="Orchestrator Agent",
        goal="Coordinate agents",
        backstory="Expert coordinator",
        instructions="Coordinate effectively",
        tools=[]
    )


@pytest.fixture
def mock_langchain_agent():
    """Mock LangChainAgent for testing without API calls."""
    mock = Mock()
    mock.name = "MockAgent"
    mock.role = "Mock Role"
    mock.goal = "Mock Goal"
    mock.backstory = "Mock Backstory"
    mock.instructions = "Mock Instructions"
    mock.tools = []
    mock.execute = Mock(return_value="Mocked response")
    return mock


@pytest.fixture
def mock_praisonai_agent():
    """Mock PraisonAI Agent for testing."""
    mock = Mock()
    mock.name = "MockAgent"
    mock.role = "Mock Role"
    mock.goal = "Mock Goal"
    mock.backstory = "Mock Backstory"
    mock.instructions = "Mock Instructions"
    return mock


@pytest.fixture(autouse=True)
def reset_factory_state():
    """Reset factory state before each test."""
    # This fixture runs before each test to ensure clean state
    yield
    # Cleanup after test if needed