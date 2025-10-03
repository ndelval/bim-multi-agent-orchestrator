"""
LangChain integration module for Orchestrator.

This module provides clean imports for LangChain/LangGraph components and handles
the migration from PraisonAI to LangChain-based agent orchestration.
"""

import logging
from typing import Optional, Any, Dict, List, Union, TYPE_CHECKING, TypedDict, Annotated
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

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


class RouterDecision(TypedDict, total=False):
    """
    Structured type for router decision results.

    Attributes:
        route: The selected route name
        confidence: Confidence score (0.0-1.0)
        method: Routing method used (e.g., 'rule_based', 'llm_based')
        reasoning: Optional explanation for the routing decision
    """
    route: str
    confidence: float
    method: str
    reasoning: Optional[str]


@dataclass
class OrchestratorState:
    """
    State schema for LangGraph StateGraph orchestration.

    This replaces the PraisonAI-based workflow with a structured state
    that can be passed between nodes in a LangGraph StateGraph.

    All mutable fields use field(default_factory=...) to prevent shared reference issues.
    The __post_init__ method validates state consistency after initialization.
    """
    # Input/Output
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    input_prompt: str = ""  # Immutable user input (read-only)
    final_output: Optional[str] = None

    # Routing and Decision Making
    current_route: Optional[str] = None  # "quick", "research", "analysis", "standards"
    router_decision: Optional[RouterDecision] = None  # Structured routing decision
    assignments: Optional[List[Dict[str, Any]]] = None

    # Agent Execution
    current_agent: Optional[str] = None
    agent_outputs: Dict[str, str] = field(default_factory=dict)  # Agent name -> result string
    completed_agents: List[str] = field(default_factory=list)

    # Graph Execution State (added fields previously missing)
    current_node: Optional[str] = None  # Currently executing node name
    execution_path: List[str] = field(default_factory=list)  # Sequence of executed nodes
    node_outputs: Dict[str, str] = field(default_factory=dict)  # Node name -> output string
    condition_results: Dict[str, bool] = field(default_factory=dict)  # Condition name -> boolean result
    parallel_execution_active: bool = False  # Whether parallel execution is in progress

    # Memory and Context
    recall_items: List[str] = field(default_factory=list)
    memory_context: Optional[str] = None

    # Execution Control
    max_iterations: int = 10
    current_iteration: int = 0

    # Error Handling (improved from single error_state string)
    errors: List[Dict[str, Any]] = field(default_factory=list)  # List of error records

    def __post_init__(self):
        """
        Validate state consistency after initialization.

        Raises:
            ValueError: If state validation fails (e.g., iteration bounds violated)
        """
        # Validate iteration bounds
        if self.current_iteration > self.max_iterations:
            raise ValueError(
                f"current_iteration ({self.current_iteration}) cannot exceed "
                f"max_iterations ({self.max_iterations})"
            )

        # Validate route consistency
        if self.current_route and self.router_decision:
            decision_route = self.router_decision.get("route")
            if decision_route and decision_route != self.current_route:
                logger.warning(
                    f"Route mismatch: current_route='{self.current_route}' but "
                    f"router_decision.route='{decision_route}'"
                )

        # Validate execution path consistency
        if self.current_node and self.current_node not in self.execution_path:
            # Current node should typically be the last in execution path
            logger.debug(
                f"current_node '{self.current_node}' not found in execution_path. "
                f"This may be expected at initialization."
            )

    def add_error(self, node: str, error_message: str, recoverable: bool = True) -> None:
        """
        Add a structured error record to the state.

        Args:
            node: The node name where the error occurred
            error_message: Description of the error
            recoverable: Whether the error is recoverable (default: True)
        """
        self.errors.append({
            "node": node,
            "message": error_message,
            "recoverable": recoverable,
            "iteration": self.current_iteration
        })

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
    LangChain-based agent wrapper that mimics PraisonAI Agent interface.
    
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
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.instructions = instructions
        self.llm_name = llm or "gpt-4o-mini"
        self.tools = tools or []
        self.kwargs = kwargs
        
        # Initialize LangChain components
        self._initialize_langchain_agent()
    
    def _initialize_langchain_agent(self):
        """Initialize the underlying LangChain agent."""
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain components not available")
        
        # Create LLM
        self.llm = ChatOpenAI(model=self.llm_name, temperature=0.1)
        
        # Create system prompt combining role, goal, backstory, and instructions
        system_prompt = f"""You are {self.name}, a {self.role}.

GOAL: {self.goal}

BACKSTORY: {self.backstory}

INSTRUCTIONS:
{self.instructions}

Always provide clear, actionable responses based on your role and expertise."""
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create the agent chain
        if self.tools:
            # Use ReAct agent if tools are provided
            self.agent = create_react_agent(self.llm, self.tools)
        else:
            # Simple conversational chain
            self.agent = self.prompt | self.llm | StrOutputParser()
    
    def execute(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute task using LangChain agent."""
        if not LANGCHAIN_AVAILABLE:
            return f"Error: LangChain not available for agent {self.name}"
        
        try:
            if hasattr(self.agent, 'invoke'):
                # For StateGraph-based agents
                if context and context.get("messages"):
                    result = self.agent.invoke({
                        "messages": context["messages"] + [HumanMessage(content=task_description)]
                    })
                else:
                    result = self.agent.invoke({
                        "messages": [HumanMessage(content=task_description)]
                    })
            else:
                # For simple chains
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=task_description)]
                })
            
            # Extract string response
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                return last_message.content if hasattr(last_message, 'content') else str(last_message)
            elif hasattr(result, 'content'):
                return result.content
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Agent {self.name} execution failed: {e}")
            return f"Error: Agent execution failed - {str(e)}"


class LangChainTask:
    """
    LangChain-based task wrapper that mimics PraisonAI Task interface.
    
    This provides backward compatibility while transitioning to LangChain.
    """
    
    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: LangChainAgent,
        context: Optional[List["LangChainTask"]] = None,
        async_execution: bool = False,
        **kwargs
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
                    if hasattr(ctx_task, 'result') and ctx_task.result:
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
        return getattr(langchain, '__version__', 'unknown')
    except (ImportError, AttributeError):
        return None


def get_langgraph_version() -> Optional[str]:
    """Get LangGraph version if available."""
    try:
        import langgraph
        return getattr(langgraph, '__version__', 'unknown')
    except (ImportError, AttributeError):
        return None


# Re-export key classes for clean imports
__all__ = [
    'OrchestratorState',
    'RouterDecision',
    'LangChainAgent', 
    'LangChainTask',
    'create_default_tools',
    'is_available',
    'get_langchain_version',
    'get_langgraph_version',
    # LangChain core classes
    'AgentAction',
    'AgentFinish', 
    'BaseMessage',
    'AgentExecutor',
    'Tool',
    'BaseTool',
    'HumanMessage',
    'AIMessage',
    'SystemMessage',
    'ChatPromptTemplate',
    'MessagesPlaceholder',
    'Runnable',
    'RunnableLambda',
    'RunnablePassthrough',
    'BaseOutputParser',
    'StrOutputParser',
    'ChatOpenAI',
    'DuckDuckGoSearchRun',
    # LangGraph classes
    'StateGraph',
    'END',
    'START',
    'create_react_agent',
    'MemorySaver',
]