"""
Agent backend strategies for mode-based agent creation.

This module implements the Strategy Pattern to support multiple agent
creation backends (LangChain, PraisonAI, etc.) with runtime selection.

Architecture:
- AgentBackend: Abstract base class defining the strategy interface
- Concrete backends: LangChainBackend, PraisonAIBackend
- BackendRegistry: Singleton registry for backend management
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class AgentBackend(ABC):
    """Abstract base class for agent creation backends.

    This defines the interface that all backend strategies must implement
    to support pluggable agent creation systems.
    """

    @abstractmethod
    def create_agent(self, config: 'AgentConfig', **kwargs) -> Any:
        """Create an agent using backend-specific implementation.

        Args:
            config: Agent configuration with name, role, goal, etc.
            **kwargs: Additional backend-specific parameters

        Returns:
            Agent instance from the backend

        Raises:
            Exception: If agent creation fails
        """
        pass

    @abstractmethod
    def get_tools(self, tool_names: List[str]) -> List[Any]:
        """Map tool names to backend-specific tool instances.

        Args:
            tool_names: List of tool identifiers (e.g., ["duckduckgo"])

        Returns:
            List of backend-specific tool instances
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are available.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Unique identifier for this backend.

        Returns:
            Backend name (e.g., 'langchain', 'praisonai')
        """
        pass

    def validate_config(self, config: 'AgentConfig', **kwargs) -> None:
        """Validate configuration for this backend.

        Args:
            config: Agent configuration to validate
            **kwargs: Additional parameters to validate

        Raises:
            ValueError: If configuration is invalid for this backend
        """
        # Default implementation - subclasses can override
        if not config.name:
            raise ValueError("Agent name is required")
        if not config.role:
            raise ValueError("Agent role is required")


class LangChainBackend(AgentBackend):
    """LangChain agent creation backend.

    This backend creates agents using the LangChain/LangGraph framework
    via the langchain_integration module.
    """

    def __init__(self):
        """Initialize LangChain backend with lazy imports."""
        self._agent_class = None
        self._tools = {}
        self._available = None

    def is_available(self) -> bool:
        """Check if LangChain dependencies are available."""
        if self._available is not None:
            return self._available

        try:
            from ..integrations.langchain_integration import LangChainAgent
            self._agent_class = LangChainAgent
            self._available = True
            logger.debug("LangChain backend is available")
            return True
        except ImportError as e:
            logger.debug(f"LangChain backend not available: {e}")
            self._available = False
            return False

    def _ensure_imports(self):
        """Ensure LangChain modules are imported."""
        if self._agent_class is None:
            if not self.is_available():
                raise ImportError(
                    "LangChain backend not available. "
                    "Install with: pip install langchain langgraph langchain-openai"
                )
            from ..integrations.langchain_integration import LangChainAgent
            self._agent_class = LangChainAgent

    def create_agent(self, config: 'AgentConfig', **kwargs) -> Any:
        """Create a LangChain agent.

        Args:
            config: Agent configuration
            **kwargs: Additional parameters (llm, tools, etc.)

        Returns:
            LangChainAgent instance
        """
        self._ensure_imports()

        # Extract LangChain-specific parameters
        llm = kwargs.get('llm', 'gpt-4o-mini')
        tools = kwargs.get('tools', [])

        # Filter out kwargs that shouldn't be passed to LangChainAgent
        agent_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ['llm', 'tools']}

        try:
            agent = self._agent_class(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions or "",
                llm=llm,
                tools=tools,
                **agent_kwargs
            )
            logger.info(f"Created LangChain agent: {config.name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create LangChain agent: {e}")
            raise

    def get_tools(self, tool_names: List[str]) -> List[Any]:
        """Get LangChain-specific tools with LangChain Tool wrapper.

        Args:
            tool_names: List of tool identifiers or callable instances

        Returns:
            List of LangChain tool instances compatible with create_react_agent
        """
        self._ensure_imports()

        tools = []
        for item in tool_names:
            if callable(item) and not isinstance(item, str):
                # PRIORITY 3 FIX: Wrap dynamic tool callable with LangChain Tool
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
                except ImportError:
                    logger.debug("LangChain Tool not available for wrapping dynamic tools")
                except Exception as e:
                    logger.debug(f"Failed to wrap dynamic tool: {e}")

            elif item == "duckduckgo":
                try:
                    from ..integrations.langchain_integration import DuckDuckGoSearchRun
                    if DuckDuckGoSearchRun:
                        tools.append(DuckDuckGoSearchRun())
                        logger.debug(f"Added DuckDuckGo tool for LangChain")
                except ImportError:
                    logger.debug(f"DuckDuckGo tool not available for LangChain")
            else:
                if isinstance(item, str):
                    # PRIORITY 3 FIX: Only debug for actual string tool names that aren't recognized
                    logger.debug(f"Unknown tool name for LangChain: {item}")

        return tools

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "langchain"


class PraisonAIBackend(AgentBackend):
    """PraisonAI agent creation backend.

    This backend creates agents using the PraisonAI framework
    via the praisonai integration module.
    """

    def __init__(self):
        """Initialize PraisonAI backend with lazy imports."""
        self._agent_class = None
        self._tools_map = {}
        self._available = None

    def is_available(self) -> bool:
        """Check if PraisonAI dependencies are available."""
        if self._available is not None:
            return self._available

        try:
            from ..integrations.praisonai import Agent
            self._agent_class = Agent
            self._available = True
            logger.debug("PraisonAI backend is available")
            return True
        except ImportError as e:
            logger.debug(f"PraisonAI backend not available: {e}")
            self._available = False
            return False

    def _ensure_imports(self):
        """Ensure PraisonAI modules are imported."""
        if self._agent_class is None:
            if not self.is_available():
                raise ImportError(
                    "PraisonAI backend not available. "
                    "Install with: pip install praisonaiagents"
                )
            from ..integrations.praisonai import Agent
            self._agent_class = Agent

    def create_agent(self, config: 'AgentConfig', **kwargs) -> Any:
        """Create a PraisonAI agent.

        Args:
            config: Agent configuration
            **kwargs: Additional parameters

        Returns:
            PraisonAI Agent instance
        """
        self._ensure_imports()

        try:
            agent = self._agent_class(
                name=config.name,
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                instructions=config.instructions,
                **kwargs
            )
            logger.info(f"Created PraisonAI agent: {config.name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create PraisonAI agent: {e}")
            raise

    def get_tools(self, tool_names: List[str]) -> List[Any]:
        """Get PraisonAI-specific tools.

        Args:
            tool_names: List of tool identifiers

        Returns:
            List of PraisonAI tool instances
        """
        self._ensure_imports()

        # Lazy load tools
        if not self._tools_map:
            try:
                from ..integrations.praisonai import duckduckgo
                self._tools_map["duckduckgo"] = duckduckgo
                logger.debug("Loaded PraisonAI tools")
            except ImportError:
                logger.warning("Failed to import PraisonAI tools")

        tools = []
        for name in tool_names:
            if name in self._tools_map:
                tools.append(self._tools_map[name])
                logger.debug(f"Added {name} tool for PraisonAI")
            else:
                logger.warning(f"Unknown tool for PraisonAI: {name}")

        return tools

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "praisonai"


class BackendRegistry:
    """Registry for agent backends with singleton pattern.

    This registry manages available backends and provides runtime
    selection based on mode parameter.

    Example:
        registry = BackendRegistry()
        backend = registry.get("langchain")
        agent = backend.create_agent(config)
    """

    _instance = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._backends = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only once due to singleton)."""
        if not self._initialized:
            self._register_default_backends()
            self._initialized = True

    def _register_default_backends(self):
        """Register built-in backends."""
        # Try to register LangChain backend
        try:
            langchain_backend = LangChainBackend()
            if langchain_backend.is_available():
                self.register(langchain_backend)
                logger.info("Registered LangChain backend")
        except Exception as e:
            logger.debug(f"Could not register LangChain backend: {e}")

        # Try to register PraisonAI backend
        try:
            praisonai_backend = PraisonAIBackend()
            if praisonai_backend.is_available():
                self.register(praisonai_backend)
                logger.info("Registered PraisonAI backend")
        except Exception as e:
            logger.debug(f"Could not register PraisonAI backend: {e}")

    def register(self, backend: AgentBackend) -> None:
        """Register a backend implementation.

        Args:
            backend: Backend instance to register

        Raises:
            ValueError: If backend name conflicts with existing registration
        """
        name = backend.backend_name
        if name in self._backends:
            logger.warning(f"Overwriting existing backend: {name}")

        self._backends[name] = backend
        logger.debug(f"Registered backend: {name}")

    def unregister(self, backend_name: str) -> None:
        """Unregister a backend.

        Args:
            backend_name: Name of backend to unregister
        """
        if backend_name in self._backends:
            del self._backends[backend_name]
            logger.debug(f"Unregistered backend: {backend_name}")

    def get(self, mode: str) -> Optional[AgentBackend]:
        """Get backend by mode name.

        Args:
            mode: Backend mode name (e.g., 'langchain', 'praisonai')

        Returns:
            Backend instance or None if not found
        """
        backend = self._backends.get(mode)
        if backend is None:
            logger.warning(
                f"Backend '{mode}' not registered. "
                f"Available: {list(self._backends.keys())}"
            )
        return backend

    def available_backends(self) -> List[str]:
        """List available backend modes.

        Returns:
            List of registered backend names
        """
        return list(self._backends.keys())

    def get_default_backend(self) -> Optional[AgentBackend]:
        """Get default backend (prefers LangChain, falls back to PraisonAI).

        Returns:
            Default backend instance or None if none available
        """
        # Prefer LangChain
        if "langchain" in self._backends:
            return self._backends["langchain"]

        # Fallback to PraisonAI
        if "praisonai" in self._backends:
            return self._backends["praisonai"]

        # No backends available
        available = self.available_backends()
        if available:
            # Return first available backend
            return self._backends[available[0]]

        logger.error("No agent backends available")
        return None

    def reset(self):
        """Reset registry (mainly for testing)."""
        self._backends = {}
        self._initialized = False
        self._register_default_backends()
        self._initialized = True