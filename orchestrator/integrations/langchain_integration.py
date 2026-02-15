"""
LangChain integration module for Orchestrator.

This module provides clean imports for LangChain/LangGraph components
"""

import logging
import os
from typing import Optional, Any, Dict, List, Union, TYPE_CHECKING, Annotated
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# LangGraph reducer functions for concurrent state updates


def merge_dicts(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
    """
    Merge two dicts for LangGraph reducer channel.

    Right dict values overwrite left on key collision.
    Compatible with Python 3.5+.

    This is used by Annotated fields in OrchestratorState to enable
    multiple concurrent writes to dict state fields (parallel execution).

    Args:
        left: Current accumulated dict value
        right: New dict value to merge

    Returns:
        Merged dict with combined keys
    """
    return {**left, **right}


def merge_lists(left: List, right: List) -> List:
    """
    Concatenate two lists for LangGraph reducer channel.

    This is used for list fields that can receive concurrent writes
    during parallel node execution (e.g., completed_agents, execution_path).

    Args:
        left: Current accumulated list
        right: New list to append

    Returns:
        Concatenated list
    """
    return left + right


# ---------------------------------------------------------------------------
# Context windowing constants and helpers (BP-COST-07, BP-COST-08)
# ---------------------------------------------------------------------------

MSG_WINDOW_SIZE: int = int(os.environ.get("ORCHESTRATOR_MSG_WINDOW_SIZE", "20"))
SUMMARY_PREFIX: str = "SUMMARY OF PRIOR CONTEXT:"

_SUMMARIZER_INSTANCE = None  # lazy singleton


def _get_summarizer():
    """Return a lazily-initialised Summarizer singleton."""
    global _SUMMARIZER_INSTANCE
    if _SUMMARIZER_INSTANCE is None:
        from orchestrator.memory.summarizer import Summarizer

        _SUMMARIZER_INSTANCE = Summarizer()
    return _SUMMARIZER_INSTANCE


def summarizing_add_messages(
    left: List,
    right: List,
) -> List:
    """Custom LangGraph reducer: add_messages + sliding window with summarization.

    Delegates to LangGraph's ``add_messages`` for merge semantics (ID-based
    deduplication, concurrent-write safety), then compresses old messages when
    the combined list exceeds ``MSG_WINDOW_SIZE``.

    Fallback behaviour: if the summarisation LLM call fails for any reason,
    fall back to simple truncation (keep the last N messages) so graph
    execution is never interrupted.
    """
    # 1. Delegate to the original LangGraph add_messages reducer
    if add_messages is None:
        # LangChain not available — simple concatenation fallback
        merged = list(left or []) + list(right or [])
    else:
        merged = add_messages(left, right)

    # 2. Under threshold → nothing to do
    if len(merged) <= MSG_WINDOW_SIZE:
        return merged

    # 3. Split into old (to summarise) and recent (to keep raw)
    recent = merged[-MSG_WINDOW_SIZE:]
    old = merged[:-MSG_WINDOW_SIZE]

    # 4. Summarise the old messages
    try:
        summarizer = _get_summarizer()
        summary = summarizer.summarize_old_messages(old)
    except Exception:
        logger.warning(
            "Message summarization failed; falling back to simple truncation",
            exc_info=True,
        )
        return list(recent)

    if not summary:
        return list(recent)

    # 5. Build the windowed message list
    #    Import here to handle the LANGCHAIN_AVAILABLE=False case gracefully
    try:
        from langchain_core.messages import HumanMessage as _HM, AIMessage as _AM

        summary_messages = [
            _HM(content=f"{SUMMARY_PREFIX} {summary}"),
            _AM(content="Understood, continuing with prior context noted."),
        ]
    except ImportError:
        # Cannot create typed messages — return raw truncation
        return list(recent)

    return summary_messages + list(recent)


# Core LangChain imports
try:
    from langchain.schema import AgentAction, AgentFinish, BaseMessage
    from langchain.agents import AgentExecutor
    from langchain.tools import Tool, BaseTool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_community.tools import DuckDuckGoSearchRun

    # LangGraph imports
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain components successfully imported")

except ImportError as e:
    logger.error(f"Failed to import LangChain components: {e}")
    # Provide fallback None values to prevent import errors
    AgentAction = None
    AgentFinish = None
    BaseMessage = None
    AgentExecutor = None
    Tool = None
    BaseTool = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    Runnable = None
    RunnableLambda = None
    RunnablePassthrough = None
    BaseOutputParser = None
    StrOutputParser = None
    ChatOpenAI = None
    DuckDuckGoSearchRun = None
    StateGraph = None
    END = None
    START = None
    add_messages = None
    create_react_agent = None
    MemorySaver = None

    LANGCHAIN_AVAILABLE = False


@dataclass
class OrchestratorState:
    """
    State schema for LangGraph StateGraph orchestration.


    All mutable fields use field(default_factory=...) to prevent shared reference issues.
    The __post_init__ method validates state consistency after initialization.

    IMPORTANT - Concurrent Write Safety Guidelines:
    - Fields written by multiple parallel nodes MUST have Annotated reducers
    - Collection types (List, Dict) that aggregate from parallel nodes need merge_lists/merge_dicts
    - Scalar fields (int, str, bool) should be read-only OR proven single-writer via graph topology
    - See ADR for state field design principles
    """

    # Input/Output
    messages: Annotated[List[BaseMessage], summarizing_add_messages] = field(
        default_factory=list
    )
    input_prompt: str = (
        ""  # Immutable user input (read-only) - SAFE: never written by nodes
    )
    final_output: Optional[str] = (
        None  # SAFE: only END node writes this (single-writer)
    )

    # Routing and Decision Making
    current_route: Optional[str] = (
        None  # "quick", "research", "analysis", "standards" - SAFE: only router writes (single-writer)
    )
    router_decision: Optional[Dict[str, Any]] = (
        None  # Structured routing decision - SAFE: only router writes (single-writer)
    )
    assignments: Optional[List[Dict[str, Any]]] = (
        None  # SAFE: only router/planner writes (single-writer)
    )

    # Agent Execution
    current_agent: Optional[str] = (
        None  # SAFE: only sequential agent nodes write (single-writer pattern)
    )
    agent_outputs: Annotated[Dict[str, str], merge_dicts] = field(
        default_factory=dict
    )  # PARALLEL-SAFE: reducer handles concurrent writes
    completed_agents: Annotated[List[str], merge_lists] = field(
        default_factory=list
    )  # PARALLEL-SAFE: reducer handles concurrent writes

    # Graph Execution State
    # NOTE: current_node field REMOVED - caused concurrent write errors in parallel execution
    # In parallel graphs, "current node" is ambiguous (multiple nodes execute simultaneously)
    # Use execution_path[-1] for last executed node in sequential flows
    execution_path: Annotated[List[str], merge_lists] = field(
        default_factory=list
    )  # PARALLEL-SAFE: reducer handles concurrent writes
    node_outputs: Annotated[Dict[str, str], merge_dicts] = field(
        default_factory=dict
    )  # PARALLEL-SAFE: reducer handles concurrent writes
    condition_results: Annotated[Dict[str, bool], merge_dicts] = field(
        default_factory=dict
    )  # PARALLEL-SAFE: reducer handles concurrent writes from parallel conditions
    parallel_execution_active: bool = (
        False  # SAFE: read-only flag set by graph framework
    )

    # Memory and Context
    recall_items: List[str] = field(
        default_factory=list
    )  # SAFE: read-only from initialization, never written by nodes
    memory_context: Optional[str] = (
        None  # SAFE: only router/memory system writes (single-writer)
    )

    # Execution Control
    max_iterations: int = 10  # SAFE: configuration constant, never written

    # Error Handling (improved from single error_state string)
    errors: Annotated[List[Dict[str, Any]], merge_lists] = field(
        default_factory=list
    )  # PARALLEL-SAFE: reducer handles concurrent error reports

    @property
    def execution_depth(self) -> int:
        """
        Number of execution steps completed (derived from execution_path).

        This replaces the removed current_iteration field which was incompatible
        with parallel execution semantics. In parallel graphs, "iteration" is ambiguous,
        but execution depth (steps in the graph) is well-defined.

        Returns:
            Number of nodes in the execution path
        """
        return len(self.execution_path)

    @property
    def completed_count(self) -> int:
        """
        Number of agents that have completed execution (derived from completed_agents).

        This provides the same information as the removed current_iteration field
        but with clearer semantics that work correctly in parallel execution.

        Returns:
            Number of completed agents
        """
        return len(self.completed_agents)

    def __post_init__(self):
        """
        Validate state consistency after initialization.

        Raises:
            ValueError: If state validation fails
        """
        # SAFE: Relaxed validation - only detect infinite loops (>100 steps)
        # Previous strict validation (execution_depth > max_iterations) was too restrictive
        # because it didn't account for LangGraph internal nodes and state coercion overhead.
        # max_iterations is now advisory, not a hard limit.
        from ..core.constants import MAX_EXECUTION_DEPTH

        if self.execution_depth > MAX_EXECUTION_DEPTH:
            raise ValueError(
                f"Possible infinite loop detected: execution_depth ({self.execution_depth}) "
                f"exceeds safety threshold ({MAX_EXECUTION_DEPTH}). This may indicate a cycle in the graph."
            )

        # Validate route consistency
        if self.current_route and self.router_decision:
            decision_route = self.router_decision.get("route")
            if decision_route and decision_route != self.current_route:
                logger.warning(
                    f"Route mismatch: current_route='{self.current_route}' but "
                    f"router_decision.route='{decision_route}'"
                )

        # NOTE: Validation for current_node removed - field eliminated to fix concurrent write bug
        # execution_path contains complete node execution history; use execution_path[-1] for last node in sequential flows

    def add_error(
        self, node: str, error_message: str, recoverable: bool = True
    ) -> None:
        """
        Add a structured error record to the state.

        Args:
            node: The node name where the error occurred
            error_message: Description of the error
            recoverable: Whether the error is recoverable (default: True)
        """
        self.errors.append(
            {
                "node": node,
                "message": error_message,
                "recoverable": recoverable,
                "execution_depth": self.execution_depth,  # Use derived property instead of current_iteration
            }
        )

    def has_fatal_errors(self) -> bool:
        """
        Check if any non-recoverable errors exist in the state.

        Returns:
            True if any error is marked as non-recoverable
        """
        return any(not error.get("recoverable", True) for error in self.errors)

    def get_contextual_prompt(self, additional_context: str = "") -> str:
        """
        Get prompt with added context WITHOUT mutating state.

        This replaces the anti-pattern of modifying input_prompt in-place.
        Agents should use this method to get enriched prompts without state mutation.

        Args:
            additional_context: Additional context to prepend to the prompt

        Returns:
            Contextually enriched prompt string
        """
        base_prompt = self.input_prompt

        if additional_context:
            return f"{additional_context}\n\nOriginal request: {base_prompt}"

        if self.memory_context:
            return f"{self.memory_context}\n\n{base_prompt}"

        return base_prompt


class LangChainAgent:
    """
    LangChain-based agent wrapper

    This provides backward compatibility while transitioning to LangChain.
    """

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        instructions: str = "",
        llm: Optional[str] = None,
        llm_provider: str = "openai",
        tools: Optional[List[BaseTool]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.instructions = instructions
        self.llm_name = llm or "gpt-4o-mini"
        self.llm_provider = llm_provider
        self.tools = tools or []
        self.llm_kwargs = llm_kwargs or {}
        self.kwargs = kwargs

        # Initialize LangChain components
        self._initialize_langchain_agent()

    def _initialize_langchain_agent(self):
        """Initialize the underlying LangChain agent."""
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain components not available")

        # Create LLM via provider-agnostic factory
        from ..core.llm_factory import LLMFactory

        self.llm = LLMFactory.create(
            provider=self.llm_provider,
            model=self.llm_name,
            temperature=0.1,
            **self.llm_kwargs,
        )

        # Create system prompt from centralized template (BP-MCP-05)
        from ..prompts import get_prompt

        system_prompt = get_prompt(
            "system.agent",
            name=self.name,
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            instructions=self.instructions,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Create the agent chain
        if self.tools:
            # Use ReAct agent if tools are provided
            self.agent = create_react_agent(self.llm, self.tools)
        else:
            # Simple conversational chain
            self.agent = self.prompt | self.llm | StrOutputParser()

    def execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List] = None,
    ) -> str:
        """Execute task using LangChain agent with detailed logging.

        Args:
            task_description: The task to execute
            context: Optional execution context
            callbacks: Optional LangChain callbacks (e.g. for token tracking)

        Raises:
            AgentExecutionError: If the agent execution fails.
        """
        from ..core.exceptions import AgentExecutionError

        if not LANGCHAIN_AVAILABLE:
            raise AgentExecutionError(
                agent_name=self.name,
                original_error=RuntimeError("LangChain not available"),
            )

        try:
            # Log agent execution details
            tool_names = []
            if hasattr(self, "tools") and self.tools:
                tool_names = [
                    tool.name if hasattr(tool, "name") else str(tool)
                    for tool in self.tools
                ]
            logger.info(
                "Agent '%s' executing task (tools: %s)",
                self.name,
                tool_names or "none",
            )

            # Build invoke config for callbacks (token tracking, etc.)
            invoke_config = {"callbacks": callbacks} if callbacks else None

            # Execute agent
            if hasattr(self.agent, "invoke"):
                # For StateGraph-based agents
                if context and context.get("messages"):
                    result = self.agent.invoke(
                        {
                            "messages": context["messages"]
                            + [HumanMessage(content=task_description)]
                        },
                        config=invoke_config,
                    )
                else:
                    result = self.agent.invoke(
                        {"messages": [HumanMessage(content=task_description)]},
                        config=invoke_config,
                    )
            else:
                # For simple chains
                result = self.agent.invoke(
                    {"messages": [HumanMessage(content=task_description)]},
                    config=invoke_config,
                )

            # Extract string response
            output = None
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                output = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
            elif hasattr(result, "content"):
                output = result.content
            else:
                output = str(result)

            logger.info("Agent '%s' completed (%d chars)", self.name, len(output))

            return output

        except Exception as e:
            logger.error(f"Agent {self.name} execution failed: {e}")
            logger.exception("Full exception traceback:")
            raise AgentExecutionError(
                agent_name=self.name,
                original_error=e,
            ) from e


class LangChainTask:
    """
    LangChain-based task wrapper

    This provides backward compatibility while transitioning to LangChain.
    """

    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: LangChainAgent,
        context: Optional[List["LangChainTask"]] = None,
        async_execution: bool = False,
        **kwargs,
    ):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context or []
        self.async_execution = async_execution
        self.kwargs = kwargs

        # Execution state
        self.result = None
        self.status = "pending"  # pending, running, completed, failed

    def execute(self, context_data: Optional[Dict[str, Any]] = None) -> str:
        """Execute the task using the assigned agent."""
        try:
            self.status = "running"

            # Build context from dependent tasks
            task_context = ""
            if self.context:
                context_results = []
                for ctx_task in self.context:
                    if hasattr(ctx_task, "result") and ctx_task.result:
                        context_results.append(f"Previous result: {ctx_task.result}")
                if context_results:
                    task_context = "\n".join(context_results) + "\n\n"

            # Add external context
            if context_data:
                task_context += f"Additional context: {context_data}\n\n"

            # Execute task
            full_description = f"{task_context}Task: {self.description}\n\nExpected output: {self.expected_output}"
            self.result = self.agent.execute(full_description, context_data)
            self.status = "completed"

            return self.result

        except Exception as e:
            self.status = "failed"
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)
            self.result = error_msg
            return self.result


def create_default_tools() -> List[BaseTool]:
    """Create default tools for LangChain agents."""
    if not LANGCHAIN_AVAILABLE:
        return []

    tools = []

    # Add DuckDuckGo search tool
    try:
        search_tool = DuckDuckGoSearchRun()
        tools.append(search_tool)
    except Exception as e:
        logger.warning(f"Failed to create search tool: {e}")

    return tools


def is_available() -> bool:
    """Check if LangChain components are available."""
    return LANGCHAIN_AVAILABLE


def get_langchain_version() -> Optional[str]:
    """Get LangChain version if available."""
    try:
        import langchain

        return getattr(langchain, "__version__", "unknown")
    except (ImportError, AttributeError):
        return None


def get_langgraph_version() -> Optional[str]:
    """Get LangGraph version if available."""
    try:
        import langgraph

        return getattr(langgraph, "__version__", "unknown")
    except (ImportError, AttributeError):
        return None


# Re-export key classes for clean imports
__all__ = [
    "OrchestratorState",
    "LangChainAgent",
    "LangChainTask",
    "create_default_tools",
    "is_available",
    "get_langchain_version",
    "get_langgraph_version",
    # LangChain core classes
    "AgentAction",
    "AgentFinish",
    "BaseMessage",
    "AgentExecutor",
    "Tool",
    "BaseTool",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "Runnable",
    "RunnableLambda",
    "RunnablePassthrough",
    "BaseOutputParser",
    "StrOutputParser",
    "ChatOpenAI",
    "DuckDuckGoSearchRun",
    # LangGraph classes
    "StateGraph",
    "END",
    "START",
    "create_react_agent",
    "MemorySaver",
]
