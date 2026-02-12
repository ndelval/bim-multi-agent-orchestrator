"""
Component initialization and setup for the Orchestrator.

This module handles the initialization logic for all orchestrator components,
following the Single Responsibility Principle by separating initialization
concerns from execution and lifecycle management.
"""

import logging
from typing import Dict, List, Any, Optional
from copy import deepcopy

from ..integrations.langchain_integration import LangChainAgent as Agent
from ..factories.agent_factory import AgentFactory
from ..factories.task_factory import TaskFactory
from ..factories.graph_factory import GraphFactory
from ..memory.memory_manager import MemoryManager
from .config import OrchestratorConfig, AgentConfig
from .exceptions import AgentCreationError, OrchestratorError

logger = logging.getLogger(__name__)


class OrchestratorInitializer:
    """
    Handles initialization of orchestrator components.

    This class is responsible for setting up all necessary components
    including memory manager, workflow engine, agents, and the LangGraph system.
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the orchestrator initializer.

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory()

    def initialize_memory(self) -> Optional[MemoryManager]:
        """
        Initialize memory manager if configured.

        Returns:
            Initialized MemoryManager instance or None if not configured
        """
        if self.config.memory is None:
            logger.debug(
                "Memory configuration not provided, skipping memory initialization"
            )
            return None

        try:
            memory_manager = MemoryManager(self.config.memory)
            logger.info("Memory manager initialized successfully")
            return memory_manager
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {str(e)}")
            raise OrchestratorError(f"Memory initialization failed: {str(e)}")

    def create_agents(self) -> Dict[str, Agent]:
        """
        Create agents from configuration.

        Returns:
            Dictionary mapping agent names to Agent instances

        Raises:
            AgentCreationError: If agent creation fails
        """
        try:
            agents = {}
            # Get enabled agents from configuration
            enabled_agents = [agent for agent in self.config.agents if agent.enabled]

            for agent_config in enabled_agents:
                agent = self.agent_factory.create_agent(agent_config)
                agents[agent_config.name] = agent
                logger.debug(f"Created agent: {agent_config.name}")

            logger.info(f"Created {len(agents)} agents successfully")
            return agents
        except Exception as e:
            raise AgentCreationError(f"Failed to create agents: {str(e)}")

    def create_dynamic_tools(
        self, memory_manager: Optional[MemoryManager]
    ) -> Dict[str, Any]:
        """
        Create dynamic tools for agent use.

        Args:
            memory_manager: Optional memory manager for GraphRAG tool

        Returns:
            Dictionary of tool name to tool instance
        """
        dynamic_tools = {}

        if memory_manager:
            try:
                # Import here to avoid circular dependencies
                from ..memory.memory_manager import MemoryManager as MM

                graph_tool = memory_manager.create_graph_tool(
                    default_user_id=self.config.user_id,
                    default_run_id=self.config.run_id,
                )
                dynamic_tools["graph_rag_lookup"] = graph_tool
                logger.info("Created GraphRAG tool for agent attachment")
            except Exception as e:
                logger.warning(f"Failed to create GraphRAG tool: {e}")

        return dynamic_tools

    def enrich_agent_configs_with_tools(
        self, dynamic_tools: Dict[str, Any]
    ) -> List[AgentConfig]:
        """
        Enrich agent configurations with dynamic tool names.

        Args:
            dynamic_tools: Dictionary of available dynamic tools

        Returns:
            List of enriched agent configurations
        """
        enriched_agent_configs = []

        for agent_config in self.config.agents:
            if not agent_config.enabled:
                continue

            # Clone to avoid mutation
            enriched = deepcopy(agent_config)

            # Add GraphRAG tool if agent instructions reference it
            if "graph_rag_lookup" in dynamic_tools:
                if any(
                    keyword in agent_config.instructions.lower()
                    for keyword in ["graph_rag_lookup", "graphrag"]
                ):
                    if "graph_rag_lookup" not in enriched.tools:
                        enriched.tools.append("graph_rag_lookup")
                        logger.debug(f"Added GraphRAG tool to agent {enriched.name}")

            enriched_agent_configs.append(enriched)

        return enriched_agent_configs

    def create_langgraph_system(
        self, memory_manager: Optional[MemoryManager]
    ) -> tuple[GraphFactory, Any]:
        """
        Create the LangGraph StateGraph system with tool pre-registration.

        Args:
            memory_manager: Optional memory manager for tool creation

        Returns:
            Tuple of (GraphFactory, compiled_graph)

        Raises:
            OrchestratorError: If LangGraph system creation fails
        """
        try:
            # Initialize graph factory
            graph_factory = GraphFactory(self.agent_factory)

            # Create and register dynamic tools
            dynamic_tools = self.create_dynamic_tools(memory_manager)
            if dynamic_tools:
                graph_factory.register_dynamic_tools(dynamic_tools)

            # Enrich agent configs with tool names
            enriched_agent_configs = self.enrich_agent_configs_with_tools(dynamic_tools)
            self.config.agents = enriched_agent_configs

            # Create StateGraph
            compiled_graph = self._create_stategraph(
                graph_factory, enriched_agent_configs
            )

            logger.info("LangGraph system created successfully with tool integration")
            return graph_factory, compiled_graph

        except Exception as e:
            raise OrchestratorError(f"Failed to create LangGraph system: {str(e)}")

    def _create_stategraph(
        self, graph_factory: GraphFactory, enriched_agent_configs: List[AgentConfig]
    ) -> Any:
        """
        Create StateGraph based on configuration.

        Args:
            graph_factory: GraphFactory instance
            enriched_agent_configs: Agent configurations with enriched tool lists

        Returns:
            Compiled StateGraph
        """
        if self.config.tasks:
            # Use workflow graph if tasks are defined
            compiled_graph = graph_factory.create_workflow_graph(self.config).compile()
            logger.info("Created workflow StateGraph from tasks")
        else:
            # Use chat graph for agent-only configurations
            router_agent = None

            # Find router agent if available
            for agent_config in enriched_agent_configs:
                if (
                    "orchestrator" in agent_config.name.lower()
                    or "router" in agent_config.role.lower()
                ):
                    router_agent = agent_config
                    break

            # Create chat graph
            other_agents = [a for a in enriched_agent_configs if a != router_agent]
            compiled_graph = graph_factory.create_chat_graph(
                other_agents, router_agent
            ).compile()
            logger.info(f"Created chat StateGraph with {len(other_agents)} agents")

        return compiled_graph
