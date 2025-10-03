"""
CLI Backend Adapter for seamless StateGraph/PraisonAI integration.

This module provides a unified interface that automatically detects the best
available backend (StateGraph vs PraisonAI) and provides an identical CLI
experience regardless of the underlying system.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import asdict

from ..core.config import OrchestratorConfig, AgentConfig, MemoryConfig
from ..core.orchestrator import Orchestrator
from .events import (
    emit_node_start,
    emit_node_complete,
    emit_tool_invocation,
    emit_tool_complete,
)

logger = logging.getLogger(__name__)


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
        self.backend_type = "stategraph" if self.use_stategraph else "praisonai"

        logger.info(f"CLI Backend Adapter initialized: using {self.backend_type}")
        if self.use_stategraph:
            logger.info("StateGraph pipeline fully available - enhanced orchestration enabled")
        else:
            logger.info("Using PraisonAI legacy system - full compatibility maintained")

    def _detect_available_backends(self) -> Dict[str, Any]:
        """Detect which backend systems are available."""
        backend_info = {
            "stategraph": {"available": False, "components": {}, "errors": []},
            "praisonai": {"available": False, "components": {}, "errors": []}
        }

        # Test StateGraph availability
        try:
            # Test LangGraph integration
            from ..integrations.langchain_integration import is_available as langchain_available
            backend_info["stategraph"]["components"]["langchain"] = langchain_available()

            # Test graph planning pipeline
            from ..planning import is_graph_planning_available, validate_planning_environment
            backend_info["stategraph"]["components"]["graph_planning"] = is_graph_planning_available()

            # Validate environment
            env_errors = validate_planning_environment()
            backend_info["stategraph"]["errors"] = env_errors

            # StateGraph is available if LangChain works and no critical errors
            critical_errors = [e for e in env_errors if "not available" in e.lower()]
            backend_info["stategraph"]["available"] = (
                langchain_available() and
                is_graph_planning_available() and
                len(critical_errors) == 0
            )

        except Exception as e:
            backend_info["stategraph"]["errors"].append(f"Detection failed: {str(e)}")
            backend_info["stategraph"]["available"] = False

        # Test PraisonAI availability (always available as fallback)
        try:
            from ..integrations.praisonai import is_available as praisonai_available
            backend_info["praisonai"]["available"] = praisonai_available()
            backend_info["praisonai"]["components"]["praisonai"] = praisonai_available()
        except Exception as e:
            backend_info["praisonai"]["errors"].append(f"Detection failed: {str(e)}")
            backend_info["praisonai"]["available"] = False

        return backend_info

    def get_backend_info(self) -> Dict[str, Any]:
        """Get detailed information about available backends."""
        return {
            "current_backend": self.backend_type,
            "stategraph_available": self.backend_info["stategraph"]["available"],
            "praisonai_available": self.backend_info["praisonai"]["available"],
            "backend_details": self.backend_info
        }

    def execute_router(
        self,
        router_config: OrchestratorConfig,
        timeout: Optional[float] = None
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
                return self._execute_legacy_router(router_config, timeout)

        except Exception as e:
            logger.error(f"Router execution failed with {self.backend_type}: {e}")
            # Fallback to the other backend if possible
            if self.use_stategraph and self.backend_info["praisonai"]["available"]:
                logger.info("Falling back to PraisonAI for router execution")
                return self._execute_legacy_router(router_config, timeout)
            else:
                raise

    def execute_route(
        self,
        prompt: str,
        agent_sequence: Sequence[str],
        recall_items: Sequence[str],
        assignments: Optional[List[Dict[str, Any]]] = None,
        base_memory_config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        verbose: int = 1,
        max_iter: int = 6
    ) -> Any:
        """
        Execute route with the best available backend.

        Args:
            prompt: User prompt to process
            agent_sequence: Sequence of agents to use
            recall_items: Memory recall context
            assignments: Optional pre-generated assignments
            base_memory_config: Memory configuration
            user_id: User identifier
            verbose: Verbosity level
            max_iter: Maximum iterations

        Returns:
            Route execution result in consistent format
        """
        try:
            if self.use_stategraph:
                return self._execute_stategraph_route(
                    prompt, agent_sequence, recall_items, assignments,
                    base_memory_config, user_id, verbose, max_iter
                )
            else:
                return self._execute_legacy_route(
                    prompt, agent_sequence, recall_items, assignments,
                    base_memory_config, user_id, verbose, max_iter
                )

        except Exception as e:
            logger.error(f"StateGraph route execution failed: {e}")
            # Fail fast - no PraisonAI fallback (legacy system being phased out)
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
        max_steps: int = 3
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
                    prompt, recall_snippets, agent_catalog, memory_config,
                    backend_preference, max_steps
                )
            else:
                return self._generate_legacy_planning(
                    prompt, recall_snippets, agent_catalog, memory_config,
                    backend_preference, max_steps
                )

        except Exception as e:
            logger.error(f"Planning generation failed with {self.backend_type}: {e}")
            # Fallback to legacy planning
            if self.use_stategraph and self.backend_info["praisonai"]["available"]:
                logger.info("Falling back to PraisonAI for planning generation")
                return self._generate_legacy_planning(
                    prompt, recall_snippets, agent_catalog, memory_config,
                    backend_preference, max_steps
                )
            else:
                raise

    # StateGraph Implementation Methods

    def _execute_stategraph_router(
        self,
        router_config: OrchestratorConfig,
        timeout: Optional[float]
    ) -> Any:
        """Execute router using StateGraph system."""
        try:
            from ..integrations.langchain_integration import (
                OrchestratorState, HumanMessage
            )
            from ..factories.graph_factory import GraphFactory

            # Create StateGraph for router
            graph_factory = GraphFactory()
            router_agent_config = router_config.agents[0] if router_config.agents else None

            if not router_agent_config:
                raise ValueError("No router agent configuration provided")


            # Create chat graph with router agent (pass as routing_agent parameter)
            router_graph = graph_factory.create_chat_graph([], routing_agent=router_agent_config)

            # Compile the StateGraph for execution
            compiled_router = router_graph.compile()

            # Extract user prompt from task description
            user_prompt = self._extract_user_prompt_from_config(router_config)

            # Create initial state
            initial_state = OrchestratorState(
                messages=[HumanMessage(content=user_prompt)],
                input_prompt=user_prompt,
                max_iterations=router_config.max_iter
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
        self,
        prompt: str,
        agent_sequence: Sequence[str],
        recall_items: Sequence[str],
        assignments: Optional[List[Dict[str, Any]]],
        base_memory_config: Optional[MemoryConfig],
        user_id: str,
        verbose: int,
        max_iter: int
    ) -> Any:
        """Execute route using StateGraph system with ToT planning."""
        try:
            from ..planning import generate_graph_with_tot, PlanningSettings
            from ..planning.graph_compiler import compile_tot_graph
            from ..integrations.langchain_integration import (
                OrchestratorState, HumanMessage
            )

            # Create agent configurations from sequence
            agent_configs = self._create_agent_configs_from_sequence(agent_sequence)

            # Use ToT→StateGraph pipeline
            logger.info("🚀 Using advanced ToT→StateGraph planning (CLI v6.4)")
            planning_result = generate_graph_with_tot(
                prompt=prompt,
                recall_snippets=recall_items,
                agent_catalog=agent_configs,
                settings=PlanningSettings(
                    backend="gpt-4o-mini",  # Use efficient model
                    max_steps=max(len(agent_sequence), 3),
                    n_generate_sample=2,
                    n_evaluate_sample=1,
                    n_select_sample=1
                ),
                memory_config=base_memory_config,
                enable_graph_planning=True
            )

            graph_spec = planning_result.get("graph_spec")

            if graph_spec:
                # Compile and execute StateGraph
                logger.debug("Compiling and executing StateGraph")
                compiled_graph = compile_tot_graph(graph_spec, agent_configs)

                # Create initial state with context
                context_content = f"{prompt}\n\nMemory Context:\n" + "\n".join(f"- {item}" for item in recall_items)
                initial_state = OrchestratorState(
                    messages=[HumanMessage(content=context_content)],
                    input_prompt=prompt,
                    memory_context="\n".join(recall_items),
                    max_iterations=max_iter,
                    recall_items=list(recall_items)
                )

                # Execute compiled StateGraph
                result = compiled_graph.invoke(initial_state)
                return self._extract_result_from_state(result)

            else:
                # Fallback to assignments if no graph generated
                logger.info("⚠️  No StateGraph generated, falling back to assignment execution")
                fallback_assignments = planning_result.get("assignments", assignments or [])
                return self._execute_legacy_route(
                    prompt, agent_sequence, recall_items, fallback_assignments,
                    base_memory_config, user_id, verbose, max_iter
                )

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
        max_steps: int
    ) -> Dict[str, Any]:
        """Generate planning using StateGraph ToT system."""
        try:
            from ..planning import generate_graph_with_tot, PlanningSettings

            # Use advanced graph planning
            logger.info("🧠 ToT Graph Planning: generating StateGraph specification (CLI v6.4)")
            result = generate_graph_with_tot(
                prompt=prompt,
                recall_snippets=recall_snippets,
                agent_catalog=agent_catalog,
                settings=PlanningSettings(
                    backend=backend_preference,
                    max_steps=max_steps,
                    n_generate_sample=3,
                    n_evaluate_sample=2,
                    n_select_sample=2
                ),
                memory_config=memory_config,
                enable_graph_planning=True
            )

            planning_method = result.get('planning_method', 'unknown')
            assignments_count = len(result.get('assignments', []))
            has_graph = result.get('graph_spec') is not None
            logger.info(f"✅ StateGraph planning completed: method={planning_method}, assignments={assignments_count}, graph_spec={'YES' if has_graph else 'NO'}")
            return result

        except Exception as e:
            logger.error(f"StateGraph planning failed: {e}")
            raise

    # Legacy Implementation Methods

    def _execute_legacy_router(
        self,
        router_config: OrchestratorConfig,
        timeout: Optional[float]
    ) -> Any:
        """Execute router using legacy PraisonAI system."""
        logger.debug("Executing router with PraisonAI legacy backend")
        orchestrator = Orchestrator(router_config)
        return orchestrator.run_sync()

    def _execute_legacy_route(
        self,
        prompt: str,
        agent_sequence: Sequence[str],
        recall_items: Sequence[str],
        assignments: Optional[List[Dict[str, Any]]],
        base_memory_config: Optional[MemoryConfig],
        user_id: str,
        verbose: int,
        max_iter: int
    ) -> Any:
        """Execute route using legacy PraisonAI system."""
        from ..core.config import ProcessType

        logger.debug("Executing route with PraisonAI legacy backend")

        # Create agent configurations
        agent_configs = self._create_agent_configs_from_sequence(agent_sequence)

        # Build route configuration
        route_config = OrchestratorConfig(
            name="CliRoute::legacy",
            process=ProcessType.WORKFLOW.value,
            agents=agent_configs,
            tasks=[],
            memory=base_memory_config,
            verbose=verbose >= 2,
        )

        # Create and execute orchestrator
        route_orchestrator = Orchestrator(route_config)

        # Use dynamic planning
        route_orchestrator.plan_from_prompt(
            prompt,
            agent_sequence,
            recall_snippets=recall_items,
            assignments=assignments
        )

        return route_orchestrator.run_sync()

    def _generate_legacy_planning(
        self,
        prompt: str,
        recall_snippets: Sequence[str],
        agent_catalog: Sequence[AgentConfig],
        memory_config: Optional[MemoryConfig],
        backend_preference: str,
        max_steps: int
    ) -> Dict[str, Any]:
        """Generate planning using legacy ToT system."""
        try:
            from ..planning import generate_plan_with_tot, PlanningSettings

            # Use traditional assignment planning
            logger.info("📋 Legacy ToT Planning: using assignment-based planning (fallback mode)")
            result = generate_plan_with_tot(
                prompt=prompt,
                recall_snippets=recall_snippets,
                agent_catalog=agent_catalog,
                settings=PlanningSettings(
                    backend=backend_preference,
                    max_steps=max_steps
                ),
                memory_config=memory_config
            )

            # Add planning method for consistency
            result["planning_method"] = "assignments"
            logger.info(f"Legacy planning completed: {len(result.get('assignments', []))} assignments")
            return result

        except Exception as e:
            logger.error(f"Legacy planning failed: {e}")
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

        # Create orchestrator config for single agent
        task_config = TaskConfig(
            name=f"{agent_config.name}_task",
            description=f"USER PROMPT: {user_query}",
            expected_output="Analysis result",
            agent=agent_config.name
        )

        config = OrchestratorConfig(
            name=f"SingleAgent::{agent_config.name}",
            process=ProcessType.SEQUENTIAL.value,
            agents=[agent_config],
            tasks=[task_config],
            verbose=True,
        )

        # Execute using the router execution path
        return self.execute_router(config)

    def run_multi_agent_workflow(
        self,
        agent_sequence: List[str],
        user_query: str,
        rich_display=None
    ) -> Any:
        """
        Run a multi-agent workflow with the given agent sequence.

        Args:
            agent_sequence: List of agent names to execute in sequence
            user_query: User query to process
            rich_display: Optional Rich display for progress updates

        Returns:
            Workflow execution result
        """
        from .main import _get_agent_template
        from .events import emit_agent_start, emit_agent_complete

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
                query=user_query,
                limit=5
            )
            recall_items = [
                f"{item.get('content', '')[:200]}"
                for item in memory_results
            ]
            if not recall_items:
                logger.debug("No relevant memories found in graph search")

        # Emit progress events if display provided
        if rich_display:
            for agent_name in agent_sequence:
                emit_agent_start(agent_name, f"Processing with {agent_name}")

        # Execute the route
        result = self.execute_route(
            prompt=user_query,
            agent_sequence=agent_sequence,
            recall_items=recall_items,
            assignments=None,
            base_memory_config=None,
            user_id="default_user",
            verbose=1,
            max_iter=6
        )

        # Emit completion events
        if rich_display:
            for agent_name in agent_sequence:
                emit_agent_complete(agent_name, f"{agent_name} completed")

        # PRIORITY 1 FIX: Extract final output from state before returning
        final_output = self._extract_result_from_state(result)

        # Emit final answer to Rich display if available
        if rich_display:
            from .events import emit_final_answer
            emit_final_answer(final_output)

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
        if hasattr(state, 'current_route'):
            routing_metadata['decision'] = getattr(state, 'current_route')
        if hasattr(state, 'router_decision'):
            routing_metadata['router_decision'] = getattr(state, 'router_decision')
        if isinstance(state, dict):
            if 'current_route' in state:
                routing_metadata['decision'] = state['current_route']
            if 'router_decision' in state:
                routing_metadata['router_decision'] = state['router_decision']

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
        clean_output = re.sub(r'^\*\*\w+\*\*:\s*', '', raw_output.strip())

        # Try to parse as JSON to extract the "response" field
        try:
            parsed = json.loads(clean_output)
            if isinstance(parsed, dict):
                # Extract response field, fallback to content, then original
                clean_output = parsed.get('response', parsed.get('content', clean_output))
        except (json.JSONDecodeError, ValueError):
            # Not JSON, use cleaned text
            pass

        # PHASE 3 FIX: Flatten routing metadata to top-level keys for _extract_decision() compatibility
        if routing_metadata:
            return {
                'output': clean_output,
                'current_route': routing_metadata.get('decision'),
                'router_decision': routing_metadata.get('router_decision')
            }

        return clean_output  # Legacy format for backward compatibility

    def _create_agent_configs_from_sequence(self, agent_sequence: Sequence[str]) -> List[AgentConfig]:
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
    "reset_cli_adapter"
]
