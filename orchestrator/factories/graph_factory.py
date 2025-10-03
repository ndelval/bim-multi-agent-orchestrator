"""
Graph factory for creating LangGraph StateGraphs from agent configurations.

This factory builds dynamic StateGraphs that can replace PraisonAI workflows
with more controlled, observable, and debuggable agent orchestration.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import asdict

from ..integrations.langchain_integration import (
    StateGraph,
    START,
    END,
    OrchestratorState,
    LangChainAgent,
    HumanMessage,
    AIMessage,
    is_available,
)
from ..core.config import AgentConfig, TaskConfig, OrchestratorConfig
from ..core.exceptions import GraphCreationError
from .agent_factory import AgentFactory
from ..cli.events import (
    emit_node_start,
    emit_node_complete,
    emit_tool_invocation,
    emit_tool_complete,
)

logger = logging.getLogger(__name__)


class GraphFactory:
    """Factory for creating LangGraph StateGraphs from orchestrator configurations."""

    def __init__(self, agent_factory: Optional[AgentFactory] = None):
        """Initialize graph factory with agent factory."""
        if not is_available():
            raise GraphCreationError(
                "LangChain components not available for graph creation"
            )

        self.agent_factory = agent_factory or AgentFactory()
        self._created_graphs: Dict[str, StateGraph] = {}
        self._dynamic_tools: Dict[str, Any] = {}  # Registry for runtime tools
        logger.info("GraphFactory initialized with LangChain StateGraph support")

    def create_workflow_graph(
        self,
        config: OrchestratorConfig,
        custom_routing: Optional[Dict[str, str]] = None,
    ) -> StateGraph:
        """
        Create a StateGraph workflow from orchestrator configuration.

        Args:
            config: Orchestrator configuration with agents and tasks
            custom_routing: Optional custom routing logic between agents

        Returns:
            StateGraph instance ready for compilation and execution
        """
        try:
            # Create base graph with state schema
            workflow = StateGraph(OrchestratorState)

            # Track created agents for node creation
            agent_nodes = {}

            # Create agent nodes
            for agent_config in config.agents:
                if agent_config.enabled:
                    agent = self._create_agent_from_config(agent_config)
                    node_name = agent_config.name.lower().replace(" ", "_")

                    # Create agent execution function
                    agent_func = self._create_agent_function(agent, agent_config)

                    # Add node to graph
                    workflow.add_node(node_name, agent_func)
                    agent_nodes[agent_config.name] = node_name

                    logger.debug(f"Added agent node: {node_name}")

            # Add special nodes for routing and completion
            workflow.add_node("router", self._create_router_function())
            workflow.add_node("final_output", self._create_completion_function())

            # Add edges based on tasks or custom routing
            self._add_workflow_edges(workflow, config, agent_nodes, custom_routing)

            # Set entry point
            workflow.set_entry_point("router")

            # Cache the created graph
            graph_name = config.name or "default_workflow"
            self._created_graphs[graph_name] = workflow

            logger.info(
                f"Created StateGraph workflow '{graph_name}' with {len(agent_nodes)} agent nodes"
            )
            return workflow

        except Exception as e:
            raise GraphCreationError(f"Failed to create workflow graph: {str(e)}")

    def create_chat_graph(
        self,
        agent_configs: List[AgentConfig],
        routing_agent: Optional[AgentConfig] = None,
    ) -> StateGraph:
        """
        Create a chat-style StateGraph for interactive agent conversations.

        Args:
            agent_configs: List of agent configurations for the chat
            routing_agent: Optional router agent for decision making

        Returns:
            StateGraph configured for chat-style interactions
        """
        try:
            workflow = StateGraph(OrchestratorState)

            # Create routing logic
            if routing_agent:
                router = self._create_agent_from_config(routing_agent)
                router_func = self._create_router_agent_function(router)
                workflow.add_node("router", router_func)
            else:
                workflow.add_node("router", self._create_simple_router_function())

            # Add agent nodes
            agent_nodes = {}
            for agent_config in agent_configs:
                if agent_config.enabled:
                    agent = self._create_agent_from_config(agent_config)
                    node_name = agent_config.name.lower().replace(" ", "_")

                    agent_func = self._create_chat_agent_function(agent, agent_config)
                    workflow.add_node(node_name, agent_func)
                    agent_nodes[agent_config.name] = node_name

            # Add completion node
            workflow.add_node("complete", self._create_chat_completion_function())

            # Add chat-style routing edges
            self._add_chat_edges(workflow, agent_nodes)

            # Set entry point
            workflow.set_entry_point("router")

            logger.info(f"Created chat StateGraph with {len(agent_nodes)} agents")
            return workflow

        except Exception as e:
            raise GraphCreationError(f"Failed to create chat graph: {str(e)}")

    def create_sequential_graph(self, agent_configs: List[AgentConfig]) -> StateGraph:
        """
        Create a sequential StateGraph where agents execute in order.

        Args:
            agent_configs: List of agent configurations in execution order

        Returns:
            StateGraph configured for sequential execution
        """
        try:
            workflow = StateGraph(OrchestratorState)

            agent_nodes = []
            for i, agent_config in enumerate(agent_configs):
                if agent_config.enabled:
                    agent = self._create_agent_from_config(agent_config)
                    node_name = (
                        f"agent_{i}_{agent_config.name.lower().replace(' ', '_')}"
                    )

                    agent_func = self._create_sequential_agent_function(
                        agent, agent_config, i
                    )
                    workflow.add_node(node_name, agent_func)
                    agent_nodes.append(node_name)

            # Add sequential edges
            for i in range(len(agent_nodes) - 1):
                workflow.add_edge(agent_nodes[i], agent_nodes[i + 1])

            # Set entry and exit points
            if agent_nodes:
                workflow.set_entry_point(agent_nodes[0])
                workflow.add_edge(agent_nodes[-1], END)

            logger.info(f"Created sequential StateGraph with {len(agent_nodes)} agents")
            return workflow

        except Exception as e:
            raise GraphCreationError(f"Failed to create sequential graph: {str(e)}")

    def register_dynamic_tools(self, tools: Dict[str, Any]) -> None:
        """Register dynamic tools that need runtime context.

        Args:
            tools: Dictionary mapping tool names to callable instances
        """
        self._dynamic_tools.update(tools)
        logger.info(f"Registered {len(tools)} dynamic tools: {list(tools.keys())}")

    def _create_agent_from_config(
        self, config: AgentConfig, mode: str = "langchain"
    ) -> LangChainAgent:
        """Create a LangChain agent from configuration with tool resolution.

        Args:
            config: Agent configuration with tools list
            mode: Backend mode (default: 'langchain' for LangGraph)

        Returns:
            LangChainAgent with tools properly attached
        """
        # Extract tool names from config
        tool_names = config.tools or []

        # Resolve tools through backend
        backend = self.agent_factory._backend_registry.get(mode)
        static_tools = backend.get_tools(tool_names) if (backend and tool_names) else []

        # Add dynamic tools
        dynamic_tools = [
            self._dynamic_tools[name]
            for name in tool_names
            if name in self._dynamic_tools
        ]

        # Combine all tools
        all_tools = static_tools + dynamic_tools

        # Create agent with tools
        return self.agent_factory.create_agent(
            config, mode=mode, tools=all_tools  # Pass resolved tools
        )

    def _create_agent_function(
        self, agent: LangChainAgent, config: AgentConfig
    ) -> Callable:
        """Create a basic agent function that returns only modified fields."""

        def agent_function(state: OrchestratorState) -> Dict[str, Any]:
            """Execute agent and return only modified state fields."""
            start_time = time.time()
            try:
                # Emit node start event
                emit_node_start(
                    node_name=config.name,
                    agent_name=config.name,
                    tools=config.tools or [],
                )

                # Get task description
                task_description = state.input_prompt
                if state.memory_context:
                    task_description = f"{state.memory_context}\n\n{task_description}"

                # Execute agent
                result = agent.execute(task_description, {"state": asdict(state)})

                # Emit node complete event
                duration = time.time() - start_time
                emit_node_complete(
                    node_name=config.name,
                    agent_name=config.name,
                    duration=duration,
                    output_length=len(result),
                )

                logger.debug(
                    f"Agent {config.name} completed execution in {duration:.2f}s"
                )

                # Return ONLY modified fields
                return {
                    "agent_outputs": {**state.agent_outputs, config.name: result},
                    "completed_agents": state.completed_agents + [config.name],
                    "current_iteration": state.current_iteration + 1,
                    "messages": [AIMessage(content=result)]  # add_messages will append
                }

            except Exception as e:
                logger.error(f"Agent {config.name} execution failed: {e}")

                # Emit node complete with error
                duration = time.time() - start_time
                emit_node_complete(
                    node_name=config.name,
                    agent_name=config.name,
                    duration=duration,
                    output_length=0,
                )

                return {
                    "errors": state.errors + [{
                        "agent": config.name,
                        "error": str(e),
                        "timestamp": str(datetime.now())
                    }],
                    "error_state": f"Agent {config.name} failed: {str(e)}"
                }

        return agent_function

    def _create_router_function(self) -> Callable:
        """Create a router function that returns only modified fields."""

        def router_function(state: OrchestratorState) -> Dict[str, Any]:
            """Execute router and return only modified state fields."""
            try:
                logger.info("[Router] Evaluating prompt: %s", state.input_prompt)

                # Routing logic
                prompt_lower = state.input_prompt.lower()

                # Determine route based on keywords
                if any(kw in prompt_lower for kw in ["quick", "simple", "hello", "hola", "hi"]):
                    route = "quick"
                    logger.info("[Router] Matched quick keyword for prompt")
                elif any(kw in prompt_lower for kw in ["research", "search", "find", "investiga", "busca", "encuentra"]):
                    route = "research"
                    logger.info("[Router] Matched research keyword for prompt")
                elif any(kw in prompt_lower for kw in ["analyze", "analysis", "deep", "detailed", "analiza", "analisis", "análisis", "detallado"]):
                    route = "analysis"
                    logger.info("[Router] Matched analysis keyword for prompt")
                elif any(kw in prompt_lower for kw in ["standard", "compliance", "norm", "norma", "cumplimiento"]):
                    route = "standards"
                    logger.info("[Router] Matched standards keyword for prompt")
                else:
                    route = "analysis"  # Default to analysis for complex queries
                    logger.info("[Router] No keyword matched; defaulting to analysis")

                logger.info("Router selected route: %s (rule_based)", route)

                # Return ONLY modified fields
                return {
                    "current_route": route,
                    "router_decision": {
                        "route": route,
                        "confidence": 0.8,
                        "reason": "Rule-based keyword matching"
                    },
                    "messages": [AIMessage(content=f"Routing to: {route}")]
                }

            except Exception as e:
                logger.error(f"Router function failed: {e}")
                return {
                    "current_route": "analysis",  # Default route
                    "errors": state.errors + [{
                        "agent": "router",
                        "error": str(e)
                    }]
                }

        return router_function

    def _create_completion_function(self) -> Callable:
        """Create a completion function that returns only modified fields."""

        def completion_function(state: OrchestratorState) -> Dict[str, Any]:
            """Finalize workflow and return only modified state fields."""
            try:
                # Extract final output from messages
                final_result = ""
                if state.messages:
                    for msg in reversed(state.messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_result = msg.content
                            break

                # If no messages, try agent outputs
                if not final_result and state.agent_outputs:
                    final_result = "\n\n".join(
                        f"**{agent}**: {output}"
                        for agent, output in state.agent_outputs.items()
                    )

                if not final_result:
                    final_result = "No agent outputs generated."

                logger.info("Workflow completed with final output generated")

                # Return ONLY modified fields
                return {
                    "final_output": final_result,
                    "execution_path": state.execution_path + ["completion"]
                }

            except Exception as e:
                logger.error(f"Completion function failed: {e}")
                return {
                    "final_output": "Error during completion",
                    "errors": state.errors + [{
                        "agent": "completion",
                        "error": str(e)
                    }]
                }

        return completion_function

    def _create_router_agent_function(self, router_agent: LangChainAgent) -> Callable:
        """Create a router function using an LLM agent that returns only modified fields."""

        def router_agent_function(state: OrchestratorState) -> Dict[str, Any]:
            """Execute LLM router and return only modified state fields."""
            try:
                # Create routing prompt
                routing_prompt = f"""
                Analyze this user request and decide the best route:

                USER REQUEST: {state.input_prompt}

                AVAILABLE ROUTES:
                - quick: Simple responses, greetings
                - research: Information gathering, web search
                - analysis: Deep analysis, reasoning
                - standards: Compliance, regulations

                Respond with JSON: {{"route": "route_name", "confidence": 0.9, "reasoning": "why this route"}}
                """

                logger.debug("[RouterLLM] Sending routing prompt: %s", state.input_prompt)
                result = router_agent.execute(routing_prompt)
                logger.debug("[RouterLLM] Raw router output: %s", result)

                # Parse routing decision (simplified - could use JSON parsing)
                route = "analysis"  # Default fallback
                if "quick" in result.lower():
                    route = "quick"
                elif "research" in result.lower():
                    route = "research"
                elif "standards" in result.lower():
                    route = "standards"

                logger.info("Router agent selected route: %s (keyword extraction)", route)

                # Return ONLY modified fields
                return {
                    "current_route": route,
                    "router_decision": {
                        "route": route,
                        "agent_reasoning": result,
                    },
                    "messages": [AIMessage(content=f"Router LLM selected: {route}")]
                }

            except Exception as e:
                # Fallback to simple routing
                logger.exception("Router agent execution failed", exc_info=e)
                return {
                    "current_route": "analysis",
                    "errors": state.errors + [{
                        "agent": "router_agent",
                        "error": str(e)
                    }],
                    "error_state": f"Router agent failed: {str(e)}"
                }

        return router_agent_function

    def _create_simple_router_function(self) -> Callable:
        """Create a simple keyword-based router."""
        return self._create_router_function()  # Reuse the simple router

    def _create_chat_agent_function(
        self, agent: LangChainAgent, config: AgentConfig
    ) -> Callable:
        """Create a chat-style agent function."""
        return self._create_agent_function(agent, config)  # Reuse agent function

    def _create_chat_completion_function(self) -> Callable:
        """Create a chat completion function."""
        return self._create_completion_function()  # Reuse completion function

    def _create_sequential_agent_function(
        self, agent: LangChainAgent, config: AgentConfig, position: int
    ) -> Callable:
        """Create a sequential agent function that returns only modified fields."""

        def sequential_agent_function(state: OrchestratorState) -> Dict[str, Any]:
            """Execute sequential agent and return only modified state fields."""
            try:
                # Build contextual information
                context = f"You are agent {position + 1} in a sequential workflow."
                if state.completed_agents:
                    context += f" Previous agents: {', '.join(state.completed_agents)}"

                # Get contextual prompt WITHOUT mutating state
                contextual_prompt = state.get_contextual_prompt(context)

                # Execute agent with contextual prompt
                task_description = contextual_prompt
                if state.memory_context:
                    task_description = f"{state.memory_context}\n\n{task_description}"

                result = agent.execute(task_description, {"state": asdict(state)})

                # Return ONLY modified fields
                return {
                    "agent_outputs": {**state.agent_outputs, config.name: result},
                    "completed_agents": state.completed_agents + [config.name],
                    "current_iteration": state.current_iteration + 1,
                    "messages": [AIMessage(content=result)],
                    "execution_path": state.execution_path + [config.name]
                }

            except Exception as e:
                logger.error(f"Sequential agent {config.name} failed: {e}")
                return {
                    "errors": state.errors + [{
                        "agent": config.name,
                        "error": str(e),
                        "position": position
                    }]
                }

        return sequential_agent_function

    def _add_workflow_edges(
        self,
        workflow: StateGraph,
        config: OrchestratorConfig,
        agent_nodes: Dict[str, str],
        custom_routing: Optional[Dict[str, str]],
    ) -> None:
        """Add edges to the workflow graph."""

        # Router to agents based on route
        def route_condition(state: OrchestratorState) -> str:
            route = state.current_route or "analysis"

            # Map routes to agents
            route_mapping = custom_routing or {
                "quick": (
                    "quickresponder"
                    if "QuickResponder" in agent_nodes
                    else "orchestrator"
                ),
                "research": (
                    "researcher" if "Researcher" in agent_nodes else "orchestrator"
                ),
                "analysis": "analyst" if "Analyst" in agent_nodes else "orchestrator",
                "standards": (
                    "standardsagent"
                    if "StandardsAgent" in agent_nodes
                    else "orchestrator"
                ),
            }

            target_agent = route_mapping.get(route, "orchestrator")
            # Convert to node name
            for agent_name, node_name in agent_nodes.items():
                if agent_name.lower().replace(" ", "") == target_agent.lower().replace(
                    "_", ""
                ):
                    return node_name

            # Fallback to first available agent
            return list(agent_nodes.values())[0] if agent_nodes else "final_output"

        workflow.add_conditional_edges("router", route_condition)

        # All agents to final output
        for node_name in agent_nodes.values():
            workflow.add_edge(node_name, "final_output")

        # Final output to END
        workflow.add_edge("final_output", END)

    def _add_chat_edges(
        self, workflow: StateGraph, agent_nodes: Dict[str, str]
    ) -> None:
        """Add chat-style edges with dynamic routing."""

        # Router to agents
        def chat_route_condition(state: OrchestratorState) -> str:
            # Simple routing based on current route
            route = state.current_route or "analysis"

            # Map to available agents
            if route == "quick" and "quickresponder" in agent_nodes.values():
                return "quickresponder"
            elif route == "research" and "researcher" in agent_nodes.values():
                return "researcher"
            elif route == "standards" and "standardsagent" in agent_nodes.values():
                return "standardsagent"
            else:
                # Default to first available agent
                return list(agent_nodes.values())[0] if agent_nodes else "complete"

        workflow.add_conditional_edges("router", chat_route_condition)

        # All agents to completion
        for node_name in agent_nodes.values():
            workflow.add_edge(node_name, "complete")

        # Complete to END
        workflow.add_edge("complete", END)

    def get_created_graphs(self) -> Dict[str, StateGraph]:
        """Get all created graphs."""
        return self._created_graphs.copy()

    def clear_cache(self) -> None:
        """Clear the graph cache."""
        self._created_graphs.clear()
        logger.info("Graph cache cleared")


def create_default_workflow_graph(
    orchestrator_config: OrchestratorConfig,
    agent_factory: Optional[AgentFactory] = None,
) -> StateGraph:
    """
    Convenience function to create a default workflow graph.

    Args:
        orchestrator_config: Configuration for the orchestrator
        agent_factory: Optional custom agent factory

    Returns:
        Compiled StateGraph ready for execution
    """
    factory = GraphFactory(agent_factory)
    graph = factory.create_workflow_graph(orchestrator_config)
    return graph.compile()


def create_chat_workflow_graph(
    agent_configs: List[AgentConfig],
    routing_agent: Optional[AgentConfig] = None,
    agent_factory: Optional[AgentFactory] = None,
) -> StateGraph:
    """
    Convenience function to create a chat workflow graph.

    Args:
        agent_configs: List of agent configurations
        routing_agent: Optional routing agent configuration
        agent_factory: Optional custom agent factory

    Returns:
        Compiled StateGraph ready for execution
    """
    factory = GraphFactory(agent_factory)
    graph = factory.create_chat_graph(agent_configs, routing_agent)
    return graph.compile()
