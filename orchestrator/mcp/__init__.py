"""
MCP (Model Context Protocol) integration for orchestrator agents.

This package provides MCP client functionality, allowing agents to connect to
and use tools from MCP servers (both local via stdio and remote via HTTP/SSE).
"""

from .config import MCPServerConfig, MCPTransportType
from .client_manager import MCPClientManager
from .tool_adapter import MCPToolAdapter

__all__ = [
    "MCPServerConfig",
    "MCPTransportType",
    "MCPClientManager",
    "MCPToolAdapter",
]
