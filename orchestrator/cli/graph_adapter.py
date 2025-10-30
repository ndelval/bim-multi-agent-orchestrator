"""
CLI Backend Adapter for StateGraph integration.

This module provides a unified interface for StateGraph-based agent orchestration
with dynamic planning and execution capabilities.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import asdict

from ..core.config import OrchestratorConfig, AgentConfig, MemoryConfig
from ..core.orchestrator import Orchestrator
from ..core.value_objects import ExecutionContext
from .events import (
    emit_node_start,
    emit_node_complete,
    emit_tool_invocation,
    emit_tool_complete,
)
from .mermaid_utils import save_mermaid_diagram, get_graph_info

logger = logging.getLogger(__name__)


def calculate_safe_max_iterations(
    graph_spec=None, agent_count=None, buffer_multiplier=3, minimum_buffer=15
):
    """
    Calculate safe max_iterations based on graph topology.

    This prevents the max_iterations validation error by accounting for:
    - LangGraph internal nodes (routing, coordination, state management)
    - State coercion overhead (multiple __post_init__ calls per node)
    - Framework-specific execution patterns

    Args:
        graph_spec: Optional StateGraphSpec with node information
        agent_count: Number of agents if graph_spec not available
        buffer_multiplier: Multiplier for safety buffer (default: 3x nodes)
        minimum_buffer: Minimum buffer to add (default: 15)

    Returns:
        Safe max_iterations value that accounts for framework overhead

    Examples:
        >>> calculate_safe_max_iterations(agent_count=3)
        24  # 3 * 3 + 15 = 24

        >>> calculate_safe_max_iterations(graph_spec=spec_with_5_nodes)
        30  # 5 * 3 + 15 = 30
    """
    if graph_spec and hasattr(graph_spec, "nodes"):
        node_count = len(graph_spec.nodes)
    elif agent_count is not None:
        node_count = agent_count
    else:
        # Fallback: assume small workflow
        node_count = 3

    # Formula: nodes * multiplier + buffer
    # This accounts for:
    # - Each user node (1x)
    # - Internal framework nodes (~2x overhead)
    # - Buffer for state coercion and retries
    safe_iterations = node_count * buffer_multiplier + minimum_buffer

    logger.debug(
        f"Calculated safe max_iterations: {safe_iterations} "
        f"(nodes={node_count}, multiplier={buffer_multiplier}, buffer={minimum_buffer})"
    )

    return safe_iterations


class CLIBackendAdapter:
    """
    Intelligent adapter that automatically selects and manages backend systems.

    This adapter provides a seamless interface between the CLI and the underlying
    agent orchestration systems, ensuring the user experience remains identical
    while leveraging the most advanced available technology.
    """

    def __init__(self, memory_manager=None, llm="gpt-4o-mini", enable_parallel=True):
        """Initialize the adapter with automatic backend detection.

        Args:
            memory_manager: Memory manager instance for agent tools
            llm: LLM model name to use
            enable_parallel: Whether to enable parallel execution
        """
        self.memory_manager = memory_manager
        self.llm = llm
        self.enable_parallel = enable_parallel

        self.backend_info = self._detect_available_backends()
        self.use_stategraph = self.backend_info["stategraph"]["available"]
        self.backend_type = "stategraph"

        logger.info(f"CLI Backend Adapter initialized: using {self.backend_type}")
        if self.use_stategraph:
            logger.info(
                "StateGraph pipeline fully available - enhanced orchestration enabled"
            )
        else:
            logger.error("StateGraph dependencies not available - cannot proceed")

    def _detect_available_backends(self) -> Dict[str, Any]:
        """Detect which backend systems are available."""
        backend_info = {
            "stategraph": {"available": False, "components": {}, "errors": []}
        }

        # Test StateGraph availability
        try:
            # Test LangGraph integration
            from ..integrations.langchain_integration import (
                is_available as langchain_available,
            )

            backend_info["stategraph"]["components"][
                "langchain"
            ] = langchain_available()

            # Test graph planning pipeline
            from ..planning import (
                is_graph_planning_available,
                validate_planning_environment,
            )

            backend_info["stategraph"]["components"][
                "graph_planning"
            ] = is_graph_planning_available()

            # Validate environment
            env_errors = validate_planning_environment()
            backend_info["stategraph"]["errors"] = env_errors

            # StateGraph is available if LangChain works and no critical errors
            critical_errors = [e for e in env_errors if "not available" in e.lower()]
            backend_info["stategraph"]["available"] = (
                langchain_available()
                and is_graph_planning_available()
                and len(critical_errors) == 0
            )

        except Exception as e:
            backend_info["stategraph"]["errors"].append(f"Detection failed: {str(e)}")
            backend_info["stategraph"]["available"] = False

        return backend_info

    def get_backend_info(self) -> Dict[str, Any]:
        """Get detailed information about available backends."""
        return {
            "current_backend": self.backend_type,
            "stategraph_available": self.backend_info["stategraph"]["available"],
            "backend_details": self.backend_info,
        }

    def execute_router(
        self, router_config: OrchestratorConfig, timeout: Optional[float] = None
    ) -> Any:
        """
        Execute router decision with the best available backend.

        Args:
            router_config: Router orchestrator configuration
            timeout: Optional execution timeout

        Returns:
            Router execution result in consistent format
        """
        try:
            if self.use_stategraph:
                return self._execute_stategraph_router(router_config, timeout)
            else:
                raise RuntimeError("StateGraph backend not available")

        except Exception as e:
            logger.error(f"Router execution failed with {self.backend_type}: {e}")
            raise

    def execute_route(
        self, agent_sequence: Sequence[str], context: ExecutionContext
    ) -> Any:
        """
        Execute route with the best available backend.

        Args:
            agent_sequence: Sequence of agents to use
            context: ExecutionContext with prompt, user_id, recall_items, etc.

        Returns:
            Route execution result in consistent format
        """
        try:
            if self.use_stategraph:
                return self._execute_stategraph_route(agent_sequence, context)
            else:
                return self._execute_legacy_route(agent_sequence, context)

        except Exception as e:
            logger.error(f"StateGraph route execution failed: {e}")
            from ..core.exceptions import GraphCreationError

            raise GraphCreationError(
                f"StateGraph execution failed. Please check graph configuration and ToT planning output. "
                f"Original error: {str(e)}"
            ) from e

    def generate_planning(
        self,
        prompt: str,
        recall_snippets: Sequence[str],
        agent_catalog: Sequence[AgentConfig],
        memory_config: Optional[MemoryConfig] = None,
        backend_preference: str = "gpt-4o-mini",
        max_steps: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate planning using the best available method.

        Args:
            prompt: User prompt
            recall_snippets: Memory context
            agent_catalog: Available agents
            memory_config: Memory configuration
            backend_preference: Preferred LLM backend
            max_steps: Maximum planning steps

        Returns:
            Planning result with assignments and optional graph_spec
        """
        try:
            if self.use_stategraph:
                return self._generate_stategraph_planning(
                    prompt,
                    recall_snippets,
                    agent_catalog,
                    memory_config,
                    backend_preference,
                    max_steps,
                )
            else:
                return self._generate_legacy_planning(
                    prompt,
                    recall_snippets,
                    agent_catalog,
                    memory_config,
                    backend_preference,
                    max_steps,
                )

        except Exception as e:
            logger.error(f"Planning generation failed with {self.backend_type}: {e}")
            raise

    # StateGraph Implementation Methods

    def _execute_stategraph_router(
        self, router_config: OrchestratorConfig, timeout: Optional[float]
    ) -> Any:
        """Execute router using StateGraph system."""
        try:
            from ..integrations.langchain_integration import (
                OrchestratorState,
                HumanMessage,
            )
            from ..factories.graph_factory import GraphFactory

            # Create StateGraph for router
            graph_factory = GraphFactory()
            router_agent_config = (
                router_config.agents[0] if router_config.agents else None
            )

            if not router_agent_config:
                raise ValueError("No router agent configuration provided")

            # Create chat graph with router agent (pass as routing_agent parameter)
            router_graph = graph_factory.create_chat_graph(
                [], routing_agent=router_agent_config
            )

            # Compile the StateGraph for execution
            compiled_router = router_graph.compile()

            # Extract user prompt from task description
            user_prompt = self._extract_user_prompt_from_config(router_config)

            # Create initial state
            initial_state = OrchestratorState(
                messages=[HumanMessage(content=user_prompt)],
                input_prompt=user_prompt,
                max_iterations=router_config.max_iterations,
            )

            # Execute StateGraph
            logger.debug("Executing router with StateGraph backend")
            result = compiled_router.invoke(initial_state)

            # Extract result in format compatible with legacy system
            return self._extract_result_from_state(result)

        except Exception as e:
            logger.error(f"StateGraph router execution failed: {e}")
            raise

    def _execute_stategraph_route(
        self, agent_sequence: Sequence[str], context: ExecutionContext
    ) -> Any:
        """Execute route using StateGraph system with ToT planning."""
        try:
            from ..planning import generate_graph_with_tot, PlanningSettings
            from ..planning.graph_compiler import compile_tot_graph
            from ..integrations.langchain_integration import (
                OrchestratorState,
                HumanMessage,
            )

            # Create agent configurations from sequence
            agent_configs = self._create_agent_configs_from_sequence(agent_sequence)

            # Use ToT→StateGraph pipeline
            logger.info("🚀 Using advanced ToT→StateGraph planning (CLI v6.4)")
            planning_result = generate_graph_with_tot(
                prompt=context.prompt,
                recall_snippets=context.recall_items,
                agent_catalog=agent_configs,
                settings=PlanningSettings(
                    backend="gpt-4o-mini",  # Use efficient model
                    max_steps=max(len(agent_sequence), 3),
                    n_generate_sample=2,
                    n_evaluate_sample=1,
                    n_select_sample=1,
                ),
                memory_config=context.base_memory_config,
                enable_graph_planning=True,
            )

            graph_spec = planning_result.get("graph_spec")

            if graph_spec:
                # Compile and execute StateGraph
                logger.debug("Compiling and executing StateGraph")
                compiled_graph = compile_tot_graph(graph_spec, agent_configs)

                # PHASE 2: Generate and save Mermaid diagram
                try:
                    graph_info = get_graph_info(compiled_graph)

                    # Check for errors in graph info extraction
                    if "error" in graph_info:
                        logger.warning(
                            f"Failed to extract graph info: {graph_info['error']}"
                        )
                    else:
                        logger.info(
                            f"📊 Graph structure: {graph_info['node_count']} nodes, {graph_info['edge_count']} edges"
                        )

                    mermaid_path = save_mermaid_diagram(
                        compiled_graph, filename=f"workflow_{graph_spec.name}"
                    )
                    if mermaid_path:
                        logger.info(f"📈 Mermaid diagram saved: {mermaid_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate Mermaid diagram: {e}")

                # Calculate safe max_iterations based on graph topology
                safe_max_iter = calculate_safe_max_iterations(
                    graph_spec=graph_spec, agent_count=len(agent_configs)
                )
                logger.info(
                    f"Using dynamic max_iterations={safe_max_iter} "
                    f"(graph has {len(graph_spec.nodes)} nodes)"
                )

                # Create initial state with context
                context_content = f"{context.prompt}\n\nMemory Context:\n" + "\n".join(
                    f"- {item}" for item in context.recall_items
                )
                initial_state = OrchestratorState(
                    messages=[HumanMessage(content=context_content)],
                    input_prompt=context.prompt,
                    memory_context="\n".join(context.recall_items),
                    max_iterations=safe_max_iter,  # Use calculated value instead of hardcoded
                    recall_items=list(context.recall_items),
                )

                # Execute compiled StateGraph
                # NOTE: LangGraph's StateGraph.invoke() returns a dict, not an OrchestratorState object
                result = compiled_graph.invoke(initial_state)

                # Post-execution validation: detect abnormal execution patterns
                # DEFENSIVE FIX: Extract execution_depth from dict state (mimics OrchestratorState.execution_depth property)
                # OrchestratorState.execution_depth is computed as: len(self.execution_path)
                if isinstance(result, dict):
                    execution_path = result.get("execution_path", [])
                    final_depth = len(execution_path) if execution_path else 0
                else:
                    # Fallback for object-based state (future-proof)
                    final_depth = getattr(result, "execution_depth", 0)

                from ..core.constants import DEPTH_WARNING_THRESHOLD

                if final_depth > DEPTH_WARNING_THRESHOLD:
                    logger.warning(
                        f"⚠️  High execution depth detected: {final_depth} steps. "
                        f"This may indicate inefficient graph topology or potential loops."
                    )
                elif final_depth > safe_max_iter * 0.8:
                    logger.info(
                        f"Execution depth ({final_depth}) approaching max_iterations ({safe_max_iter}). "
                        f"Consider optimizing graph structure if this occurs frequently."
                    )

                logger.debug(
                    f"Workflow completed successfully in {final_depth} execution steps"
                )

                return self._extract_result_from_state(result)

            else:
                # Fallback to assignments if no graph generated
                logger.info(
                    "⚠️  No StateGraph generated, falling back to assignment execution"
                )
                fallback_assignments = planning_result.get(
                    "assignments", context.assignments or []
                )
                # Create updated context with fallback assignments
                fallback_context = context.with_assignments(fallback_assignments)
                return self._execute_legacy_route(agent_sequence, fallback_context)

        except Exception as e:
            logger.error(f"StateGraph route execution failed: {e}")
            raise

    def _generate_stategraph_planning(
        self,
        prompt: str,
        recall_snippets: Sequence[str],
        agent_catalog: Sequence[AgentConfig],
        memory_config: Optional[MemoryConfig],
        backend_preference: str,
        max_steps: int,
    ) -> Dict[str, Any]:
        """Generate planning using StateGraph ToT system."""
        try:
            from ..planning import generate_graph_with_tot, PlanningSettings

            # Use advanced graph planning
            logger.info(
                "🧠 ToT Graph Planning: generating StateGraph specification (CLI v6.4)"
            )
            result = generate_graph_with_tot(
                prompt=prompt,
                recall_snippets=recall_snippets,
                agent_catalog=agent_catalog,
                settings=PlanningSettings(
                    backend=backend_preference,
                    max_steps=max_steps,
                    n_generate_sample=3,
                    n_evaluate_sample=2,
                    n_select_sample=2,
                ),
                memory_config=memory_config,
                enable_graph_planning=True,
            )

            planning_method = result.get("planning_method", "unknown")
            assignments_count = len(result.get("assignments", []))
            has_graph = result.get("graph_spec") is not None
            logger.info(
                f"✅ StateGraph planning completed: method={planning_method}, assignments={assignments_count}, graph_spec={'YES' if has_graph else 'NO'}"
            )
            return result

        except Exception as e:
            logger.error(f"StateGraph planning failed: {e}")
            raise

    # CLI-Compatible Methods

    def run_single_agent(self, agent_config: AgentConfig, user_query: str) -> Any:
        """
        Run a single agent with the given query.

        Args:
            agent_config: Agent configuration
            user_query: User query to process

        Returns:
            Agent execution result
        """
        from ..core.config import TaskConfig, ProcessType

        agent_name_lower = (agent_config.name or "").lower()
        agent_role_lower = (agent_config.role or "").lower()
        is_router_agent = "router" in agent_name_lower or "router" in agent_role_lower

        if is_router_agent:
            # Preserve dedicated router path
            task_config = TaskConfig(
                name=f"{agent_config.name}_task",
                description=f"USER PROMPT: {user_query}",
                expected_output="Routing decision",
                agent=agent_config.name,
            )

            config = OrchestratorConfig(
                name=f"Router::{agent_config.name}",
                process=ProcessType.SEQUENTIAL.value,
                agents=[agent_config],
                tasks=[task_config],
                verbose=True,
            )

            return self.execute_router(config)

        # Non-router agents: leverage route execution pipeline
        recall_items: List[str] = []
        base_memory_config = None

        if self.memory_manager:
            base_memory_config = getattr(self.memory_manager, "config", None)
            try:
                memory_results = self.memory_manager.retrieve_with_graph(
                    query=user_query,
                    limit=5,
                )
                recall_items = [
                    (item.get("content") or "")[:200]
                    for item in memory_results
                    if item.get("content")
                ]
            except Exception as exc:
                logger.warning(
                    "Memory retrieval failed for single agent '%s': %s",
                    agent_config.name,
                    exc,
                )

        context = ExecutionContext(
            prompt=user_query,
            recall_items=recall_items,
            base_memory_config=base_memory_config,
        )

        try:
            from ..integrations.langchain_integration import (
                OrchestratorState,
                HumanMessage,
            )
            from ..factories.agent_factory import AgentFactory
            from ..factories.graph_factory import GraphFactory

            # Build lightweight sequential graph for the single agent
            agent_factory = AgentFactory()
            graph_factory = GraphFactory(agent_factory)
            sequential_graph = graph_factory.create_sequential_graph([agent_config])
            compiled_graph = sequential_graph.compile()

            memory_context = "\n".join(recall_items) if recall_items else None

            initial_state = OrchestratorState(
                messages=[HumanMessage(content=user_query)],
                input_prompt=user_query,
                memory_context=memory_context,
                recall_items=list(recall_items),
                max_iterations=context.max_iterations,
            )

            result = compiled_graph.invoke(initial_state)
            return self._extract_result_from_state(result)

        except Exception as exc:
            logger.error(
                "Single-agent execution via sequential graph failed (%s). Falling back to route execution.",
                exc,
                exc_info=True,
            )
            return self.execute_route([agent_config.name], context)

    def run_multi_agent_workflow(
        self, agent_sequence: List[str], user_query: str, display_adapter=None
    ) -> Any:
        """
        Run a multi-agent workflow with the given agent sequence.

        Args:
            agent_sequence: List of agent names to execute in sequence
            user_query: User query to process
            display_adapter: Optional display adapter for progress updates

        Returns:
            Workflow execution result
        """
        from .main import _get_agent_template

        # Create agent configurations from sequence
        agent_configs = []
        for agent_name in agent_sequence:
            try:
                agent_config = _get_agent_template(agent_name.lower())
                agent_configs.append(agent_config)
            except ValueError:
                logger.warning(f"Unknown agent template: {agent_name}, skipping")

        if not agent_configs:
            raise ValueError(f"No valid agents found in sequence: {agent_sequence}")

        # Execute workflow using route execution
        recall_items = []
        if self.memory_manager:
            # PRIORITY 2 FIX: Retrieve relevant context from memory (now properly delegated)
            memory_results = self.memory_manager.retrieve_with_graph(
                query=user_query, limit=5
            )
            recall_items = [
                f"{item.get('content', '')[:200]}" for item in memory_results
            ]
            if not recall_items:
                logger.debug("No relevant memories found in graph search")

        # Emit progress events if display provided
        if display_adapter:
            for agent_name in agent_sequence:
                display_adapter.show_agent_start(
                    agent_name, f"Processing with {agent_name}"
                )

        # Calculate safe max_iterations for this workflow
        safe_max_iter = calculate_safe_max_iterations(agent_count=len(agent_sequence))
        logger.debug(
            f"Multi-agent workflow using max_iterations={safe_max_iter} "
            f"for {len(agent_sequence)} agents"
        )

        # Create execution context
        execution_context = ExecutionContext(
            prompt=user_query,
            recall_items=recall_items,
            assignments=None,
            base_memory_config=None,
            user_id="default_user",
            verbose=1,
            max_iterations=safe_max_iter,  # Use calculated value instead of hardcoded 6
        )

        # Execute the route
        result = self.execute_route(
            agent_sequence=agent_sequence, context=execution_context
        )

        # Emit completion events
        if display_adapter:
            for agent_name in agent_sequence:
                display_adapter.show_agent_complete(
                    agent_name, f"{agent_name} completed"
                )

        # PRIORITY 1 FIX: Extract final output from state before returning
        if isinstance(result, dict) and any(
            key in result for key in ("messages", "final_output", "agent_outputs")
        ):
            final_output = self._extract_result_from_state(result)
        elif isinstance(result, str):
            final_output = result
        else:
            # Already-normalized payload (e.g., {"output": ..., "current_route": ...})
            final_output = result

        # Show final answer via display adapter if available
        if display_adapter:
            from orchestrator.cli.main import _extract_text

            display_text = _extract_text(final_output)
            display_adapter.show_final_answer(display_text)

        # Store workflow result in memory if memory manager is available
        if self.memory_manager and final_output:
            self._store_workflow_result(user_query, final_output, agent_sequence)

        return final_output  # Return clean text, not state object

    # Helper Methods

    def _extract_user_prompt_from_config(self, config: OrchestratorConfig) -> str:
        """Extract user prompt from orchestrator configuration."""
        if config.tasks and config.tasks[0].description:
            # Extract USER PROMPT from task description
            description = config.tasks[0].description
            if "USER PROMPT:" in description:
                lines = description.split("\n")
                for line in lines:
                    if "USER PROMPT:" in line:
                        return line.replace("USER PROMPT:", "").strip()
            return description
        return "No prompt provided"

    def _extract_result_from_state(self, state: Any) -> Union[str, Dict[str, Any]]:
        """
        Extract both output text AND routing metadata from StateGraph state.

        PHASE 3 FIX: Returns routing metadata when present to support proper decision extraction.
        """

        # Preserve already-normalized payloads and literal strings
        if isinstance(state, str):
            return state.strip()

        if isinstance(state, dict) and "output" in state and not any(
            key in state for key in ("messages", "final_output", "agent_outputs")
        ):
            return state

        def _from_mapping(mapping: Dict[str, Any]) -> Optional[str]:
            """Best-effort extraction when StateGraph returns a dict."""
            if not mapping:
                return None

            # Prefer the explicit final output if present
            raw = mapping.get("final_output")
            if raw:
                return raw

            # Fall back to the latest AI message
            messages = mapping.get("messages") or []
            for message in reversed(list(messages)):
                if hasattr(message, "content"):
                    if message.content:
                        return message.content
                elif isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        return content

            # Finally, rely on the most recent agent output
            agent_outputs = mapping.get("agent_outputs")
            if isinstance(agent_outputs, dict) and agent_outputs:
                return list(agent_outputs.values())[-1]

            return None

        # PHASE 3 FIX: Extract routing metadata from state
        routing_metadata = {}
        if hasattr(state, "current_route"):
            routing_metadata["decision"] = getattr(state, "current_route")
        if hasattr(state, "router_decision"):
            routing_metadata["router_decision"] = getattr(state, "router_decision")
        if isinstance(state, dict):
            if "current_route" in state:
                routing_metadata["decision"] = state["current_route"]
            if "router_decision" in state:
                routing_metadata["router_decision"] = state["router_decision"]

        # Get raw output from state depending on the container type (EXISTING CODE)
        raw_output: Optional[str] = None

        if isinstance(state, dict):
            raw_output = _from_mapping(state)
        else:
            if hasattr(state, "final_output") and getattr(state, "final_output"):
                raw_output = getattr(state, "final_output")
            elif hasattr(state, "messages") and getattr(state, "messages"):
                for message in reversed(list(getattr(state, "messages"))):
                    if hasattr(message, "content") and message.content:
                        raw_output = message.content
                        break
                    if isinstance(message, dict) and message.get("content"):
                        raw_output = message["content"]
                        break
            elif hasattr(state, "agent_outputs") and getattr(state, "agent_outputs"):
                outputs = list(getattr(state, "agent_outputs").values())
                raw_output = outputs[-1] if outputs else None

        if not raw_output:
            raw_output = "No output generated"

        # Clean the output: remove agent name prefix (e.g., "**Orchestrator**: ")
        clean_output = re.sub(r"^\*\*\w+\*\*:\s*", "", raw_output.strip())

        # Try to parse as JSON to extract the "response" field
        try:
            parsed = json.loads(clean_output)
            if isinstance(parsed, dict):
                # Extract response field, fallback to content, then original
                clean_output = parsed.get(
                    "response", parsed.get("content", clean_output)
                )
        except (json.JSONDecodeError, ValueError):
            # Not JSON, use cleaned text
            pass

        # PHASE 3 FIX: Flatten routing metadata to top-level keys for _extract_decision() compatibility
        if routing_metadata:
            return {
                "output": clean_output,
                "current_route": routing_metadata.get("decision"),
                "router_decision": routing_metadata.get("router_decision"),
            }

        return clean_output  # Legacy format for backward compatibility

    def _store_workflow_result(
        self,
        user_query: str,
        final_output: Any,
        agent_sequence: List[str]
    ) -> None:
        """
        Store multi-agent workflow result in memory.

        Args:
            user_query: The original user query
            final_output: The workflow's final output
            agent_sequence: List of agent names involved in the workflow
        """
        if not self.memory_manager:
            return

        try:
            from orchestrator.memory.document_schema import current_timestamp
            from orchestrator.cli.main import _extract_text

            # Extract clean text from result
            result_text = _extract_text(final_output)

            if not result_text:
                logger.debug("No workflow result to store")
                return

            # Create workflow metadata
            workflow_metadata = {
                "content_type": "workflow_result",
                "user_id": "default_user",
                "agent_id": "multi_agent_workflow",
                "agent_sequence": ",".join(agent_sequence),
                "workflow_type": "sequential",
                "timestamp": current_timestamp(),
            }

            # Store workflow interaction
            workflow_content = f"""Multi-Agent Workflow Result:
User Query: {user_query}
Agent Sequence: {" → ".join(agent_sequence)}
Result: {result_text}"""

            doc_id = self.memory_manager.store(
                content=workflow_content,
                metadata=workflow_metadata
            )

            logger.info(f"Workflow result stored in memory: {doc_id}")

        except Exception as e:
            logger.warning(f"Failed to store workflow result: {e}", exc_info=True)

    def _create_agent_configs_from_sequence(
        self, agent_sequence: Sequence[str]
    ) -> List[AgentConfig]:
        """Create agent configurations from agent sequence."""
        from .main import _get_agent_template

        agent_configs = []
        for agent_name in agent_sequence:
            try:
                # Normalize to lowercase for case-insensitive lookup
                agent_config = _get_agent_template(agent_name.lower())
                agent_configs.append(agent_config)
            except Exception as e:
                logger.warning(f"Failed to get agent template for {agent_name}: {e}")

        # Validate that we successfully retrieved agent configurations
        if not agent_configs and agent_sequence:
            raise ValueError(
                f"Failed to create agent configurations for sequence: {agent_sequence}. "
                f"Available templates: {list(_get_agent_template.__globals__.get('templates', {}).keys())}"
            )

        return agent_configs


# Singleton instance for global use
_cli_adapter: Optional[CLIBackendAdapter] = None


def get_cli_adapter() -> CLIBackendAdapter:
    """Get the singleton CLI backend adapter instance."""
    global _cli_adapter
    if _cli_adapter is None:
        _cli_adapter = CLIBackendAdapter()
    return _cli_adapter


def reset_cli_adapter() -> None:
    """Reset the CLI adapter (useful for testing)."""
    global _cli_adapter
    _cli_adapter = None


# PHASE 1 FIX: Add GraphAgentAdapter as alias for backward compatibility with main.py
GraphAgentAdapter = CLIBackendAdapter


# Export main classes and functions
__all__ = [
    "CLIBackendAdapter",
    "GraphAgentAdapter",  # PHASE 1 FIX: Export alias
    "get_cli_adapter",
    "reset_cli_adapter",
]
