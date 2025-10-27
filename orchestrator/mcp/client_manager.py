"""
MCP client connection manager with lazy initialization support.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None

from .stdio_custom import custom_stdio_client

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Manages MCP server connections with lazy initialization.

    Connections are created on-demand when first accessed and cleaned up
    when no longer needed. Supports both stdio (local subprocess) and
    HTTP/SSE (remote server) transports.

    Example:
        >>> manager = MCPClientManager()
        >>> config = MCPServerConfig(
        ...     name="fs-server",
        ...     transport="stdio",
        ...     command="npx",
        ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        ... )
        >>> # Connection created on first use
        >>> session = await manager.get_session(config)
        >>> # Use session for tool calls
        >>> await manager.cleanup()  # Close all connections
    """

    def __init__(self):
        """Initialize the client manager."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK not installed. Install with: pip install mcp"
            )

        self._sessions: Dict[str, ClientSession] = {}
        self._read_streams: Dict[str, Any] = {}
        self._write_streams: Dict[str, Any] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._initialized_servers: Dict[str, MCPServerConfig] = {}
        self._context_managers: Dict[str, Any] = {}  # Store context managers

        logger.info("MCPClientManager initialized")

    async def get_session(self, config: MCPServerConfig) -> ClientSession:
        """
        Get or create a client session for the given MCP server.

        Uses lazy initialization - creates connection on first access.

        Args:
            config: MCP server configuration

        Returns:
            Active ClientSession for the server

        Raises:
            RuntimeError: If connection fails
        """
        if not config.enabled:
            raise RuntimeError(f"MCP server '{config.name}' is disabled")

        # Check if already initialized
        if config.name in self._sessions:
            logger.debug(f"Reusing existing session for '{config.name}'")
            return self._sessions[config.name]

        # Get or create lock for this server
        if config.name not in self._session_locks:
            self._session_locks[config.name] = asyncio.Lock()

        # Ensure thread-safe initialization
        async with self._session_locks[config.name]:
            # Double-check after acquiring lock
            if config.name in self._sessions:
                return self._sessions[config.name]

            logger.info(f"Initializing MCP session for '{config.name}' ({config.transport})")

            try:
                if config.transport == "stdio":
                    session = await self._create_stdio_session(config)
                elif config.transport == "http":
                    session = await self._create_http_session(config)
                else:
                    raise ValueError(f"Unsupported transport: {config.transport}")

                self._sessions[config.name] = session
                self._initialized_servers[config.name] = config
                logger.info(f"Successfully initialized MCP session for '{config.name}'")

                return session

            except Exception as e:
                logger.error(f"Failed to initialize MCP session for '{config.name}': {e}")
                raise RuntimeError(f"Failed to connect to MCP server '{config.name}': {e}")

    def _prepare_env(self, config: MCPServerConfig) -> Dict[str, str]:
        """
        Prepare environment variables with proper inheritance.

        Copies parent process environment and updates with config.env.
        This ensures servers have access to all necessary environment variables.

        Args:
            config: Server configuration

        Returns:
            Complete environment dictionary
        """
        import os

        # Start with parent environment
        full_env = os.environ.copy()

        # Update with config-specific environment
        if config.env:
            full_env.update(config.env)

        logger.debug(
            f"Prepared environment for '{config.name}' with "
            f"{len(full_env)} variables ({len(config.env or {})} from config)"
        )

        return full_env

    async def _create_stdio_session(self, config: MCPServerConfig) -> ClientSession:
        """
        Create a stdio-based MCP session (local subprocess).

        Args:
            config: Server configuration

        Returns:
            Initialized ClientSession
        """
        # Use official stdio client with enhanced environment handling
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args or [],
            env=self._prepare_env(config)  # Inherit parent env + config env
        )

        # Create and store stdio client context manager
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()

        # Store context manager for proper cleanup
        self._context_managers[config.name] = stdio_ctx
        self._read_streams[config.name] = read
        self._write_streams[config.name] = write

        # Create and initialize session with timeout
        # Note: Older MCP servers (like .NET implementation) may use protocol version "2024-11-05"
        session = ClientSession(read, write)

        # Override protocol version if specified in config
        original_version = None
        if config.protocol_version:
            try:
                import mcp.types
                original_version = mcp.types.LATEST_PROTOCOL_VERSION
                mcp.types.LATEST_PROTOCOL_VERSION = config.protocol_version
                logger.info(
                    f"Using custom protocol version '{config.protocol_version}' "
                    f"for server '{config.name}' (default was '{original_version}')"
                )
            except Exception as e:
                logger.warning(f"Failed to override protocol version: {e}")

        try:
            # Use timeout for initialization (60 seconds for .NET servers)
            await asyncio.wait_for(session.initialize(), timeout=config.timeout)
        except (asyncio.TimeoutError, TimeoutError) as timeout_err:
            # Cleanup on timeout
            await stdio_ctx.__aexit__(None, None, None)
            self._context_managers.pop(config.name, None)
            raise RuntimeError(
                f"Timeout ({config.timeout}s) connecting to MCP server '{config.name}'. "
                f"Server process may not be starting correctly or not responding to MCP protocol. "
                f"Command: {config.command} {' '.join(config.args or [])}"
            ) from timeout_err
        except Exception as e:
            # Cleanup on any error
            await stdio_ctx.__aexit__(None, None, None)
            self._context_managers.pop(config.name, None)
            # Extract nested exception from ExceptionGroup if present
            error_msg = str(e)
            if hasattr(e, 'exceptions') and e.exceptions:
                error_msg = str(e.exceptions[0])
            raise RuntimeError(
                f"Error connecting to MCP server '{config.name}': {error_msg}. "
                f"Command: {config.command} {' '.join(config.args or [])}"
            ) from e
        finally:
            # Restore original protocol version
            if original_version is not None:
                try:
                    import mcp.types
                    mcp.types.LATEST_PROTOCOL_VERSION = original_version
                    logger.debug(f"Restored protocol version to '{original_version}'")
                except Exception as e:
                    logger.warning(f"Failed to restore protocol version: {e}")

        logger.debug(f"Stdio session created for '{config.name}' - command: {config.command}")
        return session

    async def _create_http_session(self, config: MCPServerConfig) -> ClientSession:
        """
        Create an HTTP/SSE-based MCP session (remote server).

        Args:
            config: Server configuration

        Returns:
            Initialized ClientSession
        """
        # Create and store SSE client context manager
        sse_ctx = sse_client(config.url)
        read, write = await sse_ctx.__aenter__()

        # Store context manager for proper cleanup
        self._context_managers[config.name] = sse_ctx
        self._read_streams[config.name] = read
        self._write_streams[config.name] = write

        # Create and initialize session with timeout
        session = ClientSession(read, write)

        # Override protocol version if specified in config
        original_version = None
        if config.protocol_version:
            try:
                import mcp.types
                original_version = mcp.types.LATEST_PROTOCOL_VERSION
                mcp.types.LATEST_PROTOCOL_VERSION = config.protocol_version
                logger.info(
                    f"Using custom protocol version '{config.protocol_version}' "
                    f"for server '{config.name}' (default was '{original_version}')"
                )
            except Exception as e:
                logger.warning(f"Failed to override protocol version: {e}")

        try:
            await asyncio.wait_for(session.initialize(), timeout=config.timeout)
        except asyncio.TimeoutError:
            # Cleanup on timeout
            await sse_ctx.__aexit__(None, None, None)
            self._context_managers.pop(config.name, None)
            raise RuntimeError(
                f"Timeout connecting to MCP server '{config.name}' at {config.url}. "
                f"Server may be unreachable or not responding."
            )
        finally:
            # Restore original protocol version
            if original_version is not None:
                try:
                    import mcp.types
                    mcp.types.LATEST_PROTOCOL_VERSION = original_version
                    logger.debug(f"Restored protocol version to '{original_version}'")
                except Exception as e:
                    logger.warning(f"Failed to restore protocol version: {e}")

        logger.debug(f"HTTP/SSE session created for '{config.name}' - url: {config.url}")
        return session

    async def list_tools(self, config: MCPServerConfig) -> List[Dict[str, Any]]:
        """
        List available tools from an MCP server.

        Args:
            config: MCP server configuration

        Returns:
            List of tool definitions

        Raises:
            RuntimeError: If connection or listing fails
        """
        try:
            session = await self.get_session(config)
            response = await session.list_tools()

            tools = []
            if hasattr(response, 'tools'):
                for tool in response.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description if hasattr(tool, 'description') else "",
                        "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    })

            logger.info(f"Listed {len(tools)} tools from '{config.name}'")
            return tools

        except Exception as e:
            logger.error(f"Failed to list tools from '{config.name}': {e}")
            raise RuntimeError(f"Failed to list tools from MCP server '{config.name}': {e}")

    async def call_tool(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            config: MCP server configuration
            tool_name: Name of the tool to call
            arguments: Tool arguments (optional)

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If tool call fails
        """
        try:
            session = await self.get_session(config)

            logger.debug(f"Calling tool '{tool_name}' on '{config.name}'")
            result = await session.call_tool(tool_name, arguments=arguments or {})

            logger.debug(f"Tool '{tool_name}' completed successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}' on '{config.name}': {e}")
            raise RuntimeError(
                f"Failed to call tool '{tool_name}' on MCP server '{config.name}': {e}"
            )

    async def cleanup_server(self, server_name: str) -> None:
        """
        Close connection to a specific MCP server.

        Args:
            server_name: Name of the server to disconnect
        """
        if server_name in self._sessions:
            logger.info(f"Cleaning up MCP session for '{server_name}'")

            try:
                # Close session
                session = self._sessions.pop(server_name)

                # Close the context manager (stdio_client or sse_client)
                if server_name in self._context_managers:
                    ctx = self._context_managers.pop(server_name)
                    try:
                        await ctx.__aexit__(None, None, None)
                    except Exception as ctx_error:
                        logger.debug(f"Error closing context manager for '{server_name}': {ctx_error}")

                # Clean up streams
                self._read_streams.pop(server_name, None)
                self._write_streams.pop(server_name, None)
                self._initialized_servers.pop(server_name, None)

                logger.debug(f"Successfully cleaned up session for '{server_name}'")

            except Exception as e:
                logger.warning(f"Error cleaning up session for '{server_name}': {e}")

    async def cleanup(self) -> None:
        """Close all active MCP connections."""
        logger.info(f"Cleaning up {len(self._sessions)} MCP session(s)")

        server_names = list(self._sessions.keys())
        for server_name in server_names:
            await self.cleanup_server(server_name)

        # Clear all data structures
        self._sessions.clear()
        self._read_streams.clear()
        self._write_streams.clear()
        self._session_locks.clear()
        self._initialized_servers.clear()
        self._context_managers.clear()

        logger.info("All MCP sessions cleaned up")

    def is_connected(self, server_name: str) -> bool:
        """
        Check if a server is currently connected.

        Args:
            server_name: Name of the server

        Returns:
            True if connected, False otherwise
        """
        return server_name in self._sessions

    def get_active_servers(self) -> List[str]:
        """
        Get list of currently active server connections.

        Returns:
            List of server names with active connections
        """
        return list(self._sessions.keys())

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
