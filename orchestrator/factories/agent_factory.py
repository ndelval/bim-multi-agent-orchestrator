"""
Agent factory for creating and managing agents.

This module provides agent creation using LangChain/LangGraph integration.
Simplified from previous Strategy Pattern implementation.
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging
import asyncio

from ..core.config import AgentConfig
from ..core.exceptions import AgentCreationError, TemplateError

# Import MCP components (optional - graceful degradation if not available)
try:
    from ..mcp import MCPClientManager, MCPToolAdapter, MCPServerConfig
    MCP_AVAILABLE = True
except ImportError:
    MCPClientManager = None
    MCPToolAdapter = None
    MCPServerConfig = None
    MCP_AVAILABLE = False

# Import LangChain components - required
from ..integrations.langchain_integration import LangChainAgent as Agent

# Import LangChain tools - optional
try:
    from ..integrations.langchain_integration import DuckDuckGoSearchRun
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DuckDuckGoSearchRun = None
    DUCKDUCKGO_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.info("AgentFactory module loaded with direct LangChain integration")


class BaseAgentTemplate(ABC):
    """Base class for agent templates.

    Templates provide default configurations for different agent types
    and handle agent creation using direct LangChain integration.
    """

    def create_agent(self, config: AgentConfig, **kwargs) -> Agent:
        """Create an agent from configuration using LangChain.

        Args:
            config: Agent configuration
            **kwargs: Additional arguments (llm, tools, etc.)

        Returns:
            Created LangChainAgent instance

        Raises:
            AgentCreationError: If agent creation fails
        """
        try:
            # Extract parameters
            llm = kwargs.get('llm', config.llm or 'gpt-4o-mini')
            tools = kwargs.get('tools') or self._resolve_tools(config.tools)

            # Create LangChain agent directly
            agent = Agent(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions or "",
                llm=llm,
                tools=tools
            )
            logger.info(f"Created LangChain agent: {config.name}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent '{config.name}': {e}")
            raise AgentCreationError(
                f"Failed to create agent '{config.name}': {str(e)}"
            )

    def _resolve_tools(self, tool_names: List[str]) -> List[Any]:
        """Resolve tool names to LangChain tool instances.

        Args:
            tool_names: List of tool identifiers or callable instances

        Returns:
            List of LangChain tool instances
        """
        tools = []
        for item in tool_names:
            # Handle callable tools (e.g., dynamic tools from MCP)
            if callable(item) and not isinstance(item, str):
                try:
                    from langchain.tools import Tool as LangChainTool
                    tool_name = getattr(item, '__name__', 'dynamic_tool')
                    tool_desc = getattr(item, '__doc__', 'Dynamic tool')
                    wrapped = LangChainTool(
                        name=tool_name,
                        description=tool_desc or f"Tool: {tool_name}",
                        func=item
                    )
                    tools.append(wrapped)
                    logger.debug(f"Wrapped dynamic tool: {wrapped.name}")
                except Exception as e:
                    logger.warning(f"Failed to wrap dynamic tool: {e}")

            # Handle string tool names
            elif item == "duckduckgo" and DUCKDUCKGO_AVAILABLE:
                try:
                    tools.append(DuckDuckGoSearchRun())
                    logger.debug("Added DuckDuckGo search tool")
                except Exception as e:
                    logger.warning(f"Failed to add DuckDuckGo tool: {e}")
            elif isinstance(item, str):
                logger.debug(f"Unknown tool name: {item}")

        return tools

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
    """Factory for creating agents using LangChain integration.

    Simplified factory that creates agents directly without backend abstraction.
    All agents are created using LangChain/LangGraph.

    Example:
        factory = AgentFactory()
        agent = factory.create_agent(config)
    """

    def __init__(self):
        """Initialize the agent factory with LangChain integration."""
        self._templates: Dict[str, BaseAgentTemplate] = {}
        self._register_default_templates()

        # Initialize MCP components if available
        self._mcp_client_manager: Optional[MCPClientManager] = None
        self._mcp_tool_adapter: Optional[MCPToolAdapter] = None
        if MCP_AVAILABLE:
            try:
                self._mcp_client_manager = MCPClientManager()
                self._mcp_tool_adapter = MCPToolAdapter(self._mcp_client_manager)
                logger.info("MCP support enabled in AgentFactory")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP support: {e}")
                self._mcp_client_manager = None
                self._mcp_tool_adapter = None

        logger.info("AgentFactory initialized with direct LangChain integration")

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
    
    def create_agent(
        self,
        config: AgentConfig,
        agent_type: Optional[str] = None,
        **kwargs
    ) -> Agent:
        """Create an agent from configuration using LangChain.

        Args:
            config: Agent configuration
            agent_type: Override agent type (defaults to inferring from role/name)
            **kwargs: Additional arguments to pass to agent creation (llm, tools, etc.)

        Returns:
            Created LangChainAgent instance

        Raises:
            AgentCreationError: If agent creation fails

        Example:
            factory = AgentFactory()
            agent = factory.create_agent(config)
        """
        # Determine agent type
        if agent_type is None:
            agent_type = self._infer_agent_type(config)

        # Get template
        template = self.get_template(agent_type)
        if template is None:
            raise AgentCreationError(f"No template found for agent type: {agent_type}")

        # Process MCP servers if configured
        mcp_tools = []
        if config.mcp_servers and self._mcp_tool_adapter:
            try:
                mcp_tools = self._create_mcp_tools(config.mcp_servers)
                logger.info(f"Created {len(mcp_tools)} MCP tool(s) for agent '{config.name}'")
            except Exception as e:
                logger.warning(f"Failed to create MCP tools for '{config.name}': {e}")

        # Merge MCP tools with existing tools
        if mcp_tools:
            if isinstance(config.tools, list):
                from copy import deepcopy
                config = deepcopy(config)
                config.tools = list(config.tools) + mcp_tools
            else:
                logger.warning(f"Agent '{config.name}' has non-list tools, skipping MCP tool merge")

        # Create agent using LangChain
        try:
            agent = template.create_agent(config, **kwargs)
            logger.info(f"Created agent '{config.name}' of type '{agent_type}' using LangChain")
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
    
    def _create_mcp_tools(self, mcp_servers: List[Any]) -> List[Callable]:
        """
        Create tools from MCP server configurations.

        Args:
            mcp_servers: List of MCPServerConfig instances

        Returns:
            List of callable tool functions

        Raises:
            RuntimeError: If MCP is not available or tool creation fails
        """
        if not MCP_AVAILABLE or not self._mcp_tool_adapter:
            raise RuntimeError("MCP support not available")

        all_tools = []

        # Process each MCP server configuration
        for server_config in mcp_servers:
            # Convert dict to MCPServerConfig if needed
            if isinstance(server_config, dict):
                server_config = MCPServerConfig.from_dict(server_config)

            if not server_config.enabled:
                logger.debug(f"Skipping disabled MCP server: {server_config.name}")
                continue

            try:
                # Create tools asynchronously (run in event loop)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in event loop, we need to handle this carefully
                    # For now, log a warning
                    logger.warning(
                        f"Cannot create MCP tools for '{server_config.name}' "
                        f"synchronously from running event loop. Skipping."
                    )
                    continue
                else:
                    tools = loop.run_until_complete(
                        self._mcp_tool_adapter.create_tools(server_config)
                    )
                    all_tools.extend(tools)
                    logger.debug(
                        f"Created {len(tools)} tool(s) from MCP server '{server_config.name}'"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create tools from MCP server '{server_config.name}': {e}"
                )
                # Continue with other servers even if one fails
                continue

        return all_tools

    async def _create_mcp_tools_async(self, mcp_servers: List[Any]) -> List[Callable]:
        """
        Create tools from MCP server configurations asynchronously.

        This method must be called from async context to avoid event loop issues.
        It properly handles async MCP tool creation without deadlocks.

        Args:
            mcp_servers: List of MCPServerConfig instances

        Returns:
            List of callable tool functions

        Raises:
            RuntimeError: If MCP is not available or tool creation fails
        """
        if not MCP_AVAILABLE or not self._mcp_tool_adapter:
            raise RuntimeError("MCP support not available")

        all_tools = []

        # Process each MCP server configuration
        for server_config in mcp_servers:
            # Convert dict to MCPServerConfig if needed
            if isinstance(server_config, dict):
                server_config = MCPServerConfig.from_dict(server_config)

            if not server_config.enabled:
                logger.debug(f"Skipping disabled MCP server: {server_config.name}")
                continue

            try:
                # Create tools asynchronously - no event loop issues
                tools = await self._mcp_tool_adapter.create_tools(server_config)
                all_tools.extend(tools)
                logger.debug(
                    f"Created {len(tools)} tool(s) from MCP server '{server_config.name}'"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create tools from MCP server '{server_config.name}': {e}"
                )
                # Continue with other servers even if one fails
                continue

        return all_tools

    async def create_agent_async(
        self,
        config: AgentConfig,
        **kwargs
    ) -> Agent:
        """
        Create an agent asynchronously with MCP support.

        This method properly handles async MCP tool creation without event loop issues.
        Use this method instead of create_agent() when working in async contexts.

        Args:
            config: Agent configuration
            **kwargs: Additional arguments passed to agent creation

        Returns:
            Created LangChainAgent instance

        Raises:
            AgentCreationError: If agent creation fails
        """
        # Process MCP servers if configured
        if config.mcp_servers and self._mcp_tool_adapter:
            try:
                mcp_tools = await self._create_mcp_tools_async(config.mcp_servers)
                logger.info(f"Created {len(mcp_tools)} MCP tool(s) for agent '{config.name}'")

                # Merge MCP tools with existing tools
                if mcp_tools:
                    if isinstance(config.tools, list):
                        config.tools = list(config.tools) + mcp_tools
                    else:
                        logger.warning(
                            f"Agent '{config.name}' has non-list tools, "
                            f"skipping MCP tool merge"
                        )
            except Exception as e:
                logger.warning(f"Failed to create MCP tools for '{config.name}': {e}")

        # Delegate to base create_agent for actual agent creation
        return self.create_agent(config, **kwargs)

    async def cleanup_mcp(self) -> None:
        """Clean up MCP client connections."""
        if self._mcp_client_manager:
            await self._mcp_client_manager.cleanup()
            logger.info("MCP client connections cleaned up")

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