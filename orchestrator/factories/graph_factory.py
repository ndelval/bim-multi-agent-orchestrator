"""
Graph factory for creating LangGraph StateGraphs from agent configurations.

This factory builds dynamic StateGraphs
with more controlled, observable, and debuggable agent orchestration.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import asdict
import json

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
from ..core.exceptions import GraphCreationError, BudgetExceededError
from .agent_factory import AgentFactory
from .route_classifier import RouteClassifier, RouteDecision
from .routing_config import get_routing_strategy, RoutingStrategy
from ..cli.events import (
    emit_node_start,
    emit_node_complete,
    emit_tool_invocation,
    emit_tool_complete,
    emit_token_usage,
)

logger = logging.getLogger(__name__)


class GraphFactory:
    """Factory for creating LangGraph StateGraphs from orchestrator configurations."""

    def __init__(
        self,
        agent_factory: Optional[AgentFactory] = None,
        route_classifier: Optional[RouteClassifier] = None,
        routing_strategy: Optional[RoutingStrategy] = None,
        token_tracker: Optional[Any] = None,
    ):
        """
        Initialize graph factory with agent factory and routing components.

        Args:
            agent_factory: Optional custom agent factory
            route_classifier: Optional custom route classifier
            routing_strategy: Optional custom routing strategy
            token_tracker: Optional TokenTracker for per-agent cost tracking
        """
        if not is_available():
            raise GraphCreationError(
                "LangChain components not available for graph creation"
            )

        self.agent_factory = agent_factory or AgentFactory()
        self.route_classifier = route_classifier or RouteClassifier()
        self.routing_strategy = routing_strategy or get_routing_strategy("default")
        self.token_tracker = token_tracker
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
                # Ensure OpenAI router always returns valid JSON (BP-STRUCT-04)
                if routing_agent.llm_provider == "openai":
                    routing_agent.llm_kwargs.setdefault(
                        "response_format", {"type": "json_object"}
                    )

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

    def _create_agent_from_config(self, config: AgentConfig) -> LangChainAgent:
        """Create a LangChain agent from configuration with dynamic tool support.

        Args:
            config: Agent configuration with tools list

        Returns:
            LangChainAgent with tools properly attached
        """
        if not config.tools:
            return self.agent_factory.create_agent(config)

        from copy import deepcopy

        enriched_config = deepcopy(config)
        resolved_tools: List[Any] = []

        for tool_entry in config.tools:
            # Preserve callables/imported tools
            if callable(tool_entry) and not isinstance(tool_entry, str):
                resolved_tools.append(tool_entry)
                continue

            if isinstance(tool_entry, str) and tool_entry in self._dynamic_tools:
                resolved_tools.append(self._dynamic_tools[tool_entry])
            else:
                resolved_tools.append(tool_entry)

        enriched_config.tools = resolved_tools
        return self.agent_factory.create_agent(enriched_config)

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

                # Create per-call token tracking callback
                callbacks = None
                token_handler = None
                if self.token_tracker and getattr(self.token_tracker, "enabled", False):
                    token_handler = self.token_tracker.create_callback(
                        agent_name=config.name,
                        model=getattr(agent, "llm_name", ""),
                    )
                    callbacks = [token_handler]

                # Get task description
                task_description = state.input_prompt
                if state.memory_context:
                    task_description = f"{state.memory_context}\n\n{task_description}"

                # Execute agent
                result = agent.execute(
                    task_description, {"state": asdict(state)}, callbacks=callbacks
                )

                # Record token usage and check budgets
                if self.token_tracker and token_handler is not None:
                    self.token_tracker.record(config.name, token_handler.usage)
                    emit_token_usage(
                        agent_name=config.name,
                        input_tokens=token_handler.usage.input_tokens,
                        output_tokens=token_handler.usage.output_tokens,
                        reasoning_tokens=token_handler.usage.reasoning_tokens,
                        total_tokens=token_handler.usage.total_tokens,
                        estimated_cost=token_handler.usage.estimated_cost,
                        model=token_handler.usage.model,
                    )
                    self.token_tracker.check_budgets(config.name)

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
                # Note: current_iteration removed - use state.execution_depth or state.completed_count instead
                return {
                    "agent_outputs": {**state.agent_outputs, config.name: result},
                    "completed_agents": state.completed_agents + [config.name],
                    "messages": [AIMessage(content=result)],  # add_messages will append
                }

            except BudgetExceededError:
                raise  # Propagate without catching to halt the workflow
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
                    "errors": state.errors
                    + [
                        {
                            "agent": config.name,
                            "error": str(e),
                            "timestamp": str(datetime.now()),
                        }
                    ],
                    "error_state": f"Agent {config.name} failed: {str(e)}",
                }

        return agent_function

    def _create_router_function(self) -> Callable:
        """
        Create a keyword-based router function using RouteClassifier.

        Returns:
            Callable router function that returns only modified state fields
        """

        def router_function(state: OrchestratorState) -> Dict[str, Any]:
            """Execute router and return only modified state fields."""
            try:
                logger.info("[Router] Evaluating prompt: %s", state.input_prompt)

                # Use RouteClassifier for keyword-based routing
                decision = self.route_classifier.classify_by_keywords(
                    state.input_prompt
                )

                logger.info(
                    "Router selected route: %s (source: %s)",
                    decision.route,
                    decision.source,
                )

                # Return ONLY modified fields
                return {
                    "current_route": decision.route,
                    "router_decision": {
                        "route": decision.route,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                    },
                    "messages": [AIMessage(content=f"Routing to: {decision.route}")],
                }

            except Exception as e:
                logger.error(f"Router function failed: {e}")
                return {
                    "current_route": self.route_classifier.default_route,
                    "errors": state.errors + [{"agent": "router", "error": str(e)}],
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
                    "execution_path": state.execution_path + ["completion"],
                }

            except Exception as e:
                logger.error(f"Completion function failed: {e}")
                return {
                    "final_output": "Error during completion",
                    "errors": state.errors + [{"agent": "completion", "error": str(e)}],
                }

        return completion_function

    def _create_router_agent_function(self, router_agent: LangChainAgent) -> Callable:
        """
        Create a router function using an LLM agent with RouteClassifier.

        Includes single-retry logic when all parsing strategies fail
        (BP-PROMPT-05).

        Returns:
            Callable router function that returns only modified state fields
        """

        def router_agent_function(state: OrchestratorState) -> Dict[str, Any]:
            """Execute LLM router and return only modified state fields."""
            try:
                # Create routing prompt
                routing_prompt = self._build_routing_prompt(state.input_prompt)

                logger.debug(
                    "[RouterLLM] Sending routing prompt: %s", state.input_prompt
                )

                # Execute router agent
                result = router_agent.execute(routing_prompt)
                logger.debug("[RouterLLM] Raw router output: %s", result)

                # Use RouteClassifier to parse LLM response
                decision = self.route_classifier.classify_from_llm_response(result)

                # Retry once if all parsing strategies failed (BP-PROMPT-05)
                if decision.source == "default" and (
                    decision.confidence is None or decision.confidence <= 0.3
                ):
                    logger.info(
                        "[RouterLLM] All parsing failed, retrying with stricter prompt"
                    )
                    retry_prompt = (
                        f"{routing_prompt}\n\n"
                        "IMPORTANT: Respond ONLY with valid JSON, no other text.\n"
                        'Example: {"route": "quick", "confidence": 0.9, '
                        '"reasoning": "Simple greeting"}'
                    )
                    retry_result = router_agent.execute(retry_prompt)
                    retry_decision = self.route_classifier.classify_from_llm_response(
                        retry_result
                    )
                    if retry_decision.source != "default":
                        decision = retry_decision

                logger.info(
                    "Router agent selected route: %s (source: %s)",
                    decision.route,
                    decision.source,
                )

                # Return ONLY modified fields
                return {
                    "current_route": decision.route,
                    "router_decision": {
                        "route": decision.route,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "agent_reasoning": result,
                    },
                    "messages": [
                        AIMessage(content=f"Router LLM selected: {decision.route}")
                    ],
                }

            except Exception as e:
                # Fallback to simple routing
                logger.exception("Router agent execution failed", exc_info=e)
                return {
                    "current_route": self.route_classifier.default_route,
                    "errors": state.errors
                    + [{"agent": "router_agent", "error": str(e)}],
                    "error_state": f"Router agent failed: {str(e)}",
                }

        return router_agent_function

    def _build_routing_prompt(self, user_request: str) -> str:
        """
        Build routing prompt for LLM router agent.

        Uses centralized prompt template (BP-MCP-05, BP-PROMPT-08) with
        XML tag structure (BP-STRUCT-02/03) and few-shot examples (BP-PROMPT-06).

        Args:
            user_request: The user's input prompt

        Returns:
            Formatted routing prompt
        """
        from ..prompts import get_prompt

        return get_prompt("routing.classify", user_request=user_request)

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
                # Note: current_iteration removed - use state.execution_depth or state.completed_count instead
                return {
                    "agent_outputs": {**state.agent_outputs, config.name: result},
                    "completed_agents": state.completed_agents + [config.name],
                    "messages": [AIMessage(content=result)],
                    "execution_path": state.execution_path + [config.name],
                }

            except Exception as e:
                logger.error(f"Sequential agent {config.name} failed: {e}")
                return {
                    "errors": state.errors
                    + [{"agent": config.name, "error": str(e), "position": position}]
                }

        return sequential_agent_function

    def _add_workflow_edges(
        self,
        workflow: StateGraph,
        config: OrchestratorConfig,
        agent_nodes: Dict[str, str],
        custom_routing: Optional[Dict[str, str]],
    ) -> None:
        """
        Add edges to the workflow graph using routing strategy.

        Args:
            workflow: StateGraph to add edges to
            config: Orchestrator configuration
            agent_nodes: Mapping of agent names to node names
            custom_routing: Optional custom route-to-agent mappings
        """
        # Create routing strategy (custom or default)
        from .routing_config import DefaultRoutingStrategy

        strategy = (
            DefaultRoutingStrategy(route_mapping=custom_routing)
            if custom_routing
            else self.routing_strategy
        )

        # Router to agents based on route
        def route_condition(state: OrchestratorState) -> str:
            """Determine target agent node based on current route."""
            route = state.current_route or "analysis"
            return strategy.get_target_agent(route, agent_nodes)

        workflow.add_conditional_edges("router", route_condition)

        # All agents to final output
        for node_name in agent_nodes.values():
            workflow.add_edge(node_name, "final_output")

        # Final output to END
        workflow.add_edge("final_output", END)

    def _add_chat_edges(
        self, workflow: StateGraph, agent_nodes: Dict[str, str]
    ) -> None:
        """
        Add chat-style edges with dynamic routing using routing strategy.

        Args:
            workflow: StateGraph to add edges to
            agent_nodes: Mapping of agent names to node names
        """
        # Use chat-specific routing strategy
        from .routing_config import get_routing_strategy

        chat_strategy = get_routing_strategy("chat")

        # Router to agents
        def chat_route_condition(state: OrchestratorState) -> str:
            """Determine target agent node for chat routing."""
            route = state.current_route or "analysis"
            return chat_strategy.get_target_agent(route, agent_nodes)

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
