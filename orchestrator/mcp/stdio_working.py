"""
Working stdio transport implementation for MCP servers.

This is a workaround for the Python MCP SDK stdio_client bug that causes
BrokenResourceError when connecting to certain MCP servers.

The official SDK's stdio_client has a stream management bug where the stdout_reader
task crashes with BrokenResourceError. This implementation uses simple subprocess
communication that works reliably.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StdioMCPClient:
    """
    Working stdio MCP client implementation.

    This client bypasses the buggy Python MCP SDK stdio_client and implements
    direct subprocess communication that actually works with MCP servers.
    """

    def __init__(self, command: str, args: list[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize stdio MCP client.

        Args:
            command: Executable to run
            args: Command arguments
            env: Environment variables (None = inherit from parent)
        """
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._initialized = False

    async def start(self) -> None:
        """Start the MCP server subprocess."""
        logger.info(f"Starting MCP server: {self.command} {' '.join(self.args)}")

        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env
        )

        logger.info(f"MCP server process started (PID: {self.process.pid})")

    async def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self.process:
            logger.info(f"Stopping MCP server process (PID: {self.process.pid})")
            self.process.kill()
            await self.process.wait()
            self.process = None

    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send JSON-RPC request and get response.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            JSON-RPC response result

        Raises:
            RuntimeError: If server communication fails
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP server process not started")

        # Build request
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method
        }
        if params:
            request["params"] = params

        # Send request
        request_str = json.dumps(request) + "\n"
        logger.debug(f"Sending request: {method}")

        try:
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()

            # Read response with timeout
            response_bytes = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=60.0
            )

            # Parse response
            response = json.loads(response_bytes.decode())
            logger.debug(f"Received response for: {method}")

            # Check for error
            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"MCP error: {error.get('message', str(error))}")

            return response.get("result", {})

        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout waiting for MCP response to {method}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")

    async def initialize(self, protocol_version: str = "2024-11-05") -> Dict[str, Any]:
        """
        Initialize MCP session.

        Args:
            protocol_version: MCP protocol version to use

        Returns:
            Server initialization result
        """
        logger.info(f"Initializing MCP session with protocol version {protocol_version}")

        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": "orchestrator-mcp-client",
                    "version": "1.0.0"
                }
            }
        )

        self._initialized = True
        logger.info(f"MCP session initialized: {result.get('serverInfo', {}).get('name', 'unknown')}")

        return result

    async def list_tools(self) -> list[Dict[str, Any]]:
        """
        List available tools from MCP server.

        Returns:
            List of tool definitions
        """
        if not self._initialized:
            raise RuntimeError("MCP session not initialized. Call initialize() first.")

        result = await self._send_request("tools/list")

        tools = result.get("tools", [])
        logger.info(f"Listed {len(tools)} tools from MCP server")

        return tools

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._initialized:
            raise RuntimeError("MCP session not initialized. Call initialize() first.")

        logger.debug(f"Calling tool: {tool_name}")

        result = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments or {}
            }
        )

        return result


@asynccontextmanager
async def working_stdio_client(
    command: str,
    args: list[str],
    env: Optional[Dict[str, str]] = None,
    protocol_version: str = "2024-11-05"
):
    """
    Working stdio MCP client context manager.

    Usage:
        async with working_stdio_client("node", ["server.js"]) as client:
            tools = await client.list_tools()
            result = await client.call_tool("myTool", {"arg": "value"})

    Args:
        command: Executable to run
        args: Command arguments
        env: Environment variables (None = inherit from parent)
        protocol_version: MCP protocol version

    Yields:
        Initialized StdioMCPClient instance
    """
    client = StdioMCPClient(command, args, env)

    try:
        # Start subprocess
        await client.start()

        # Initialize MCP session
        await client.initialize(protocol_version)

        yield client

    finally:
        # Cleanup
        await client.stop()
