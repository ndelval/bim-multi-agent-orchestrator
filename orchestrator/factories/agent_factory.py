"""
Agent factory for creating and managing agents with registry pattern.
"""

from typing import Dict, Type, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging

from praisonaiagents import Agent
from praisonaiagents.tools import duckduckgo

from ..core.config import AgentConfig
from ..core.exceptions import AgentCreationError, TemplateError


logger = logging.getLogger(__name__)


class BaseAgentTemplate(ABC):
    """Base class for agent templates."""
    
    @abstractmethod
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create an agent from configuration."""
        pass
    
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
    """Template for orchestrator agents."""
    
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
    
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create an orchestrator agent."""
        try:
            return Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                **kwargs
            )
        except Exception as e:
            raise AgentCreationError(f"Failed to create orchestrator agent '{config.name}': {str(e)}")


class ResearcherAgentTemplate(BaseAgentTemplate):
    """Template for research agents."""
    
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
    
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create a researcher agent."""
        try:
            # Add default tools if not specified
            tools = self._get_tools(config.tools)
            
            return Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                tools=tools,
                **kwargs
            )
        except Exception as e:
            raise AgentCreationError(f"Failed to create researcher agent '{config.name}': {str(e)}")
    
    def _get_tools(self, tool_names: List[str]) -> List[Any]:
        """Get tool objects from tool names."""
        tool_map = {
            "duckduckgo": duckduckgo
        }
        
        tools = []
        for tool_name in tool_names:
            if tool_name in tool_map:
                tools.append(tool_map[tool_name])
            else:
                logger.warning(f"Unknown tool: {tool_name}")
        
        # Add default tools if none specified
        if not tools and "duckduckgo" not in tool_names:
            tools.append(duckduckgo)
        
        return tools


class PlannerAgentTemplate(BaseAgentTemplate):
    """Template for planner agents."""
    
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
    
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create a planner agent."""
        try:
            return Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                **kwargs
            )
        except Exception as e:
            raise AgentCreationError(f"Failed to create planner agent '{config.name}': {str(e)}")


class ImplementerAgentTemplate(BaseAgentTemplate):
    """Template for implementer agents."""
    
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
    
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create an implementer agent."""
        try:
            return Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                **kwargs
            )
        except Exception as e:
            raise AgentCreationError(f"Failed to create implementer agent '{config.name}': {str(e)}")


class TesterAgentTemplate(BaseAgentTemplate):
    """Template for tester agents."""
    
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
    
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create a tester agent."""
        try:
            return Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                **kwargs
            )
        except Exception as e:
            raise AgentCreationError(f"Failed to create tester agent '{config.name}': {str(e)}")


class WriterAgentTemplate(BaseAgentTemplate):
    """Template for writer agents."""
    
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
    
    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create a writer agent."""
        try:
            return Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                **kwargs
            )
        except Exception as e:
            raise AgentCreationError(f"Failed to create writer agent '{config.name}': {str(e)}")


class AgentFactory:
    """Factory for creating agents with registry pattern."""
    
    def __init__(self):
        """Initialize the agent factory."""
        self._templates: Dict[str, BaseAgentTemplate] = {}
        self._register_default_templates()
    
    def _register_default_templates(self) -> None:
        """Register default agent templates."""
        default_templates = [
            OrchestratorAgentTemplate(),
            ResearcherAgentTemplate(),
            PlannerAgentTemplate(),
            ImplementerAgentTemplate(),
            TesterAgentTemplate(),
            WriterAgentTemplate()
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
    
    def create_agent(self, config: AgentConfig, agent_type: Optional[str] = None, **kwargs) -> Agent:
        """
        Create an agent from configuration.
        
        Args:
            config: Agent configuration
            agent_type: Override agent type (defaults to inferring from role/name)
            **kwargs: Additional arguments to pass to agent creation
        
        Returns:
            Created agent instance
        """
        # Determine agent type
        if agent_type is None:
            agent_type = self._infer_agent_type(config)
        
        # Get template
        template = self.get_template(agent_type)
        if template is None:
            raise AgentCreationError(f"No template found for agent type: {agent_type}")
        
        # Create agent
        try:
            agent = template.create_agent(config, **kwargs)
            logger.info(f"Created agent '{config.name}' of type '{agent_type}'")
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
        """Infer agent type from configuration."""
        # Simple heuristic based on role and name
        role_lower = config.role.lower()
        name_lower = config.name.lower()
        
        # Map common patterns to agent types
        type_patterns = {
            "orchestrator": ["orchestrator", "manager", "coordinator"],
            "researcher": ["research", "search", "web", "information"],
            "planner": ["planner", "planning", "strategy", "design"],
            "implementer": ["implement", "builder", "developer", "coder"],
            "tester": ["test", "qa", "quality", "validation"],
            "writer": ["writer", "documentation", "report", "summary"]
        }
        
        for agent_type, patterns in type_patterns.items():
            for pattern in patterns:
                if pattern in role_lower or pattern in name_lower:
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