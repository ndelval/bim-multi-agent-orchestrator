"""
Agent factory for creating and managing agents with registry pattern.

This module now supports mode-based agent creation via the Strategy Pattern.
Backends (LangChain, PraisonAI) are selected at runtime via the mode parameter.
"""

from typing import Dict, Type, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging

from ..core.config import AgentConfig
from ..core.exceptions import AgentCreationError, TemplateError
from .agent_backends import AgentBackend, BackendRegistry

# Maintain backward compatibility with global flag for existing code
try:
    from ..integrations.langchain_integration import LangChainAgent as Agent
    USING_LANGCHAIN = True
except ImportError:
    from ..integrations.praisonai import Agent
    USING_LANGCHAIN = False

logger = logging.getLogger(__name__)
logger.info(f"AgentFactory module loaded (backward compat mode: {'LangChain' if USING_LANGCHAIN else 'PraisonAI'})")


class BaseAgentTemplate(ABC):
    """Base class for agent templates with mode-aware backend support.

    Templates now use the Strategy Pattern to delegate agent creation
    to pluggable backends (LangChain, PraisonAI, etc.) based on mode.
    """

    def __init__(self, backend_registry: Optional[BackendRegistry] = None):
        """Initialize template with backend registry.

        Args:
            backend_registry: Registry of available backends (auto-created if None)
        """
        self._backend_registry = backend_registry or BackendRegistry()

    def create_agent(self, config: AgentConfig, mode: Optional[str] = None, **kwargs) -> Any:
        """Create an agent from configuration using specified backend mode.

        Args:
            config: Agent configuration
            mode: Backend mode ('langchain', 'praisonai', or None for auto-detect)
            **kwargs: Additional arguments passed to backend

        Returns:
            Created agent instance

        Raises:
            AgentCreationError: If agent creation fails
        """
        # Auto-detect mode if not specified
        if mode is None:
            mode = self._detect_default_mode()

        # Get backend strategy
        backend = self._backend_registry.get(mode)
        if backend is None:
            available = self._backend_registry.available_backends()
            raise AgentCreationError(
                f"Backend mode '{mode}' not available. "
                f"Available modes: {available}"
            )

        # Prepare tools if needed
        tools = kwargs.get('tools')
        if not tools and config.tools:
            tools = backend.get_tools(config.tools)
            kwargs['tools'] = tools

        # Delegate to backend
        try:
            agent = backend.create_agent(config, **kwargs)
            logger.info(f"Created agent '{config.name}' using {mode} backend")
            return agent
        except Exception as e:
            raise AgentCreationError(
                f"Failed to create agent '{config.name}' with {mode} backend: {str(e)}"
            )

    def _detect_default_mode(self) -> str:
        """Auto-detect preferred backend mode.

        Returns:
            Default mode name

        Raises:
            RuntimeError: If no backends available
        """
        backend = self._backend_registry.get_default_backend()
        if backend is None:
            raise RuntimeError("No agent backends available")

        return backend.backend_name

    @abstractmethod
    def get_default_config(self) -> AgentConfig:
        """Get default configuration for this agent type."""
        pass

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Agent type identifier."""
        pass


class OrchestratorAgentTemplate(BaseAgentTemplate):
    """Template for orchestrator agents.

    Orchestrators coordinate work among specialized agents.
    They don't use tools directly as they delegate to other agents.
    """

    @property
    def agent_type(self) -> str:
        return "orchestrator"

    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="Orchestrator",
            role="AI Orchestrator",
            goal=(
                "Plan, route, and coordinate work among specialized agents "
                "to produce a high‑quality final deliverable."
            ),
            backstory=(
                "You are a seasoned project orchestrator that breaks down goals into sub‑tasks, "
                "assigns them to the right specialists, monitors progress, adapts to feedback, "
                "and ensures timely, high‑quality outcomes."
            ),
            instructions=(
                "Analyze objectives, propose a minimal viable plan, assign parallelizable tasks, "
                "monitor results, and request revisions when quality risks are detected."
            ),
            tools=[]
        )


class ResearcherAgentTemplate(BaseAgentTemplate):
    """Template for research agents.

    Researchers use web search tools to gather information.
    Tools are automatically mapped via the backend strategy system.
    """

    @property
    def agent_type(self) -> str:
        return "researcher"

    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="Researcher",
            role="Web Research Specialist",
            goal="Gather up‑to‑date, sourced information",
            backstory="Expert in web research and summarization.",
            instructions="Use web search to collect reliable, relevant information with sources.",
            tools=["duckduckgo"]
        )


class PlannerAgentTemplate(BaseAgentTemplate):
    """Template for planner agents.

    Planners create actionable plans from goals and research.
    They use the backend strategy system for agent creation.
    """

    @property
    def agent_type(self) -> str:
        return "planner"

    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="Planner",
            role="Solution Planner",
            goal="Transform goals and research into an actionable plan",
            backstory="You create pragmatic plans that balance speed and quality.",
            instructions=(
                "Propose a concise plan with steps, owners, and acceptance criteria. "
                "Prefer parallelizable steps where safe."
            ),
            tools=[]
        )


class ImplementerAgentTemplate(BaseAgentTemplate):
    """Template for implementer agents.

    Implementers build prototypes based on plans.
    They use the backend strategy system for agent creation.
    """

    @property
    def agent_type(self) -> str:
        return "implementer"

    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="Implementer",
            role="Prototype Builder",
            goal="Create a simple proof‑of‑concept based on the plan",
            backstory="You build minimal prototypes quickly and document trade‑offs.",
            instructions=(
                "Implement the simplest viable approach that satisfies the plan's acceptance criteria."
            ),
            tools=[]
        )


class TesterAgentTemplate(BaseAgentTemplate):
    """Template for tester agents.

    Testers validate functionality and quality.
    They use the backend strategy system for agent creation.
    """

    @property
    def agent_type(self) -> str:
        return "tester"

    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="Tester",
            role="QA Specialist",
            goal="Validate functionality and quality",
            backstory="You design lean checks to validate core functionality.",
            instructions="Test critical paths; report defects with clear reproduction steps.",
            tools=[]
        )


class WriterAgentTemplate(BaseAgentTemplate):
    """Template for writer agents.

    Writers create clear technical documentation and summaries.
    They use the backend strategy system for agent creation.
    """

    @property
    def agent_type(self) -> str:
        return "writer"

    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="Writer",
            role="Technical Writer",
            goal="Produce a crisp executive summary",
            backstory="You synthesize complex outputs into clear narratives.",
            instructions=(
                "Create a concise report: objective, approach, key findings, limitations, next steps."
            ),
            tools=[]
        )


class AgentFactory:
    """Factory for creating agents with registry pattern and mode support.

    The factory now supports multiple backend modes (langchain, praisonai)
    via the Strategy Pattern. Mode can be specified per-agent or set as default.

    Example:
        # Auto-detect mode (prefers LangChain)
        factory = AgentFactory()

        # Explicit default mode
        factory = AgentFactory(default_mode="praisonai")

        # Per-agent mode override
        agent = factory.create_agent(config, mode="langchain")
    """

    def __init__(self, default_mode: Optional[str] = None):
        """Initialize the agent factory.

        Args:
            default_mode: Default backend mode ('langchain' or 'praisonai').
                         If None, auto-detects based on available backends.
        """
        self._templates: Dict[str, BaseAgentTemplate] = {}
        self._backend_registry = BackendRegistry()
        self._default_mode = default_mode or self._detect_default_mode()
        self._register_default_templates()
        logger.info(f"AgentFactory initialized with default mode: {self._default_mode}")
    
    def _detect_default_mode(self) -> str:
        """Auto-detect default backend mode.

        Returns:
            Default mode name (prefers 'langchain', falls back to 'praisonai')

        Raises:
            RuntimeError: If no backends available
        """
        backend = self._backend_registry.get_default_backend()
        if backend is None:
            raise RuntimeError("No agent backends available")

        return backend.backend_name

    def _register_default_templates(self) -> None:
        """Register default agent templates."""
        # Pass backend registry to templates for consistency
        default_templates = [
            OrchestratorAgentTemplate(self._backend_registry),
            ResearcherAgentTemplate(self._backend_registry),
            PlannerAgentTemplate(self._backend_registry),
            ImplementerAgentTemplate(self._backend_registry),
            TesterAgentTemplate(self._backend_registry),
            WriterAgentTemplate(self._backend_registry)
        ]

        for template in default_templates:
            self.register_template(template)
    
    def register_template(self, template: BaseAgentTemplate) -> None:
        """Register an agent template."""
        if not isinstance(template, BaseAgentTemplate):
            raise TemplateError(f"Template must inherit from BaseAgentTemplate")
        
        self._templates[template.agent_type] = template
        logger.info(f"Registered agent template: {template.agent_type}")
    
    def unregister_template(self, agent_type: str) -> None:
        """Unregister an agent template."""
        if agent_type in self._templates:
            del self._templates[agent_type]
            logger.info(f"Unregistered agent template: {agent_type}")
    
    def get_template(self, agent_type: str) -> Optional[BaseAgentTemplate]:
        """Get an agent template by type."""
        return self._templates.get(agent_type)
    
    def list_templates(self) -> List[str]:
        """List all registered agent templates."""
        return list(self._templates.keys())

    def set_default_mode(self, mode: str) -> None:
        """Set default backend mode for factory.

        Args:
            mode: Backend mode name ('langchain' or 'praisonai')

        Raises:
            ValueError: If mode is not available
        """
        available = self._backend_registry.available_backends()
        if mode not in available:
            raise ValueError(
                f"Invalid mode '{mode}'. Available modes: {available}"
            )

        self._default_mode = mode
        logger.info(f"Set default agent backend mode to: {mode}")

    def get_default_mode(self) -> str:
        """Get current default backend mode.

        Returns:
            Current default mode name
        """
        return self._default_mode

    def get_available_modes(self) -> List[str]:
        """Get list of available backend modes.

        Returns:
            List of mode names (e.g., ['langchain', 'praisonai'])
        """
        return self._backend_registry.available_backends()
    
    def create_agent(
        self,
        config: AgentConfig,
        agent_type: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create an agent from configuration.

        Args:
            config: Agent configuration
            agent_type: Override agent type (defaults to inferring from role/name)
            mode: Backend mode ('langchain', 'praisonai', or None for factory default)
            **kwargs: Additional arguments to pass to agent creation

        Returns:
            Created agent instance

        Raises:
            AgentCreationError: If agent creation fails

        Example:
            # Use factory default mode
            agent = factory.create_agent(config)

            # Override with specific mode
            agent = factory.create_agent(config, mode="langchain")
        """
        # Determine agent type
        if agent_type is None:
            agent_type = self._infer_agent_type(config)

        # Get template
        template = self.get_template(agent_type)
        if template is None:
            raise AgentCreationError(f"No template found for agent type: {agent_type}")

        # Use factory default mode if not specified
        if mode is None:
            mode = self._default_mode

        # Create agent with specified mode
        try:
            agent = template.create_agent(config, mode=mode, **kwargs)
            logger.info(f"Created agent '{config.name}' of type '{agent_type}' using {mode} backend")
            return agent
        except Exception as e:
            raise AgentCreationError(f"Failed to create agent '{config.name}': {str(e)}")
    
    def create_agents_from_configs(self, configs: List[AgentConfig], **kwargs) -> List[Agent]:
        """Create multiple agents from configurations."""
        agents = []
        for config in configs:
            if config.enabled:
                agent = self.create_agent(config, **kwargs)
                agents.append(agent)
        return agents
    
    def _infer_agent_type(self, config: AgentConfig) -> str:
        """Infer agent type from configuration.

        Checks name, role, goal, and backstory for agent type patterns.
        Returns the first matching type or defaults to 'implementer'.
        """
        # Gather all text fields for matching
        name_lower = config.name.lower()
        role_lower = config.role.lower()
        goal_lower = config.goal.lower()
        backstory_lower = config.backstory.lower()

        # Map common patterns to agent types
        type_patterns = {
            "orchestrator": ["orchestrator", "manager", "coordinator"],
            "researcher": ["research", "search", "web", "information"],
            "planner": ["planner", "planning", "strategy", "design"],
            "implementer": ["implement", "builder", "developer", "coder"],
            "tester": ["test", "qa", "quality", "validation"],
            "writer": ["writer", "documentation", "report", "summary"]
        }

        # Check all fields for pattern matches
        for agent_type, patterns in type_patterns.items():
            for pattern in patterns:
                if (pattern in name_lower or
                    pattern in role_lower or
                    pattern in goal_lower or
                    pattern in backstory_lower):
                    return agent_type

        # Default to generic type if no match
        return "implementer"
    
    def get_default_config(self, agent_type: str) -> AgentConfig:
        """Get default configuration for an agent type."""
        template = self.get_template(agent_type)
        if template is None:
            raise TemplateError(f"No template found for agent type: {agent_type}")
        
        return template.get_default_config()
    
    def create_default_agent(self, agent_type: str, name: Optional[str] = None, **kwargs) -> Agent:
        """Create an agent with default configuration."""
        config = self.get_default_config(agent_type)
        if name:
            config.name = name
        
        return self.create_agent(config, agent_type, **kwargs)
    
    def validate_config(self, config: AgentConfig, agent_type: Optional[str] = None) -> bool:
        """Validate agent configuration."""
        try:
            if agent_type is None:
                agent_type = self._infer_agent_type(config)
            
            template = self.get_template(agent_type)
            if template is None:
                raise TemplateError(f"No template found for agent type: {agent_type}")
            
            # Basic validation
            if not config.name or not config.role or not config.goal:
                return False
            
            return True
        except Exception:
            return False