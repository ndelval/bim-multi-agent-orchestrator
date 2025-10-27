"""
Configuration dataclasses for MCP server connections.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class MCPTransportType(str, Enum):
    """Supported MCP transport mechanisms."""
    STDIO = "stdio"  # Local subprocess communication
    HTTP = "http"    # Remote HTTP/SSE communication
    SSE = "sse"      # Alias for HTTP (Server-Sent Events)


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP (Model Context Protocol) server connection.

    Supports both local (stdio) and remote (HTTP/SSE) MCP servers.

    Attributes:
        name: Unique identifier for this MCP server
        transport: Transport mechanism ("stdio" or "http"/"sse")
        command: Command to run for stdio transport (e.g., "node", "npx", "python")
        args: Command arguments for stdio transport (e.g., ["server.js"])
        url: Server URL for HTTP/SSE transport
        env: Environment variables for the server process
        tools: List of tool names to expose (manual configuration)
        timeout: Connection timeout in seconds (default: 30)
        enabled: Whether this server is enabled (default: True)
        protocol_version: Optional MCP protocol version override (e.g., "2024-11-05")

    Examples:
        Local stdio server:
        >>> config = MCPServerConfig(
        ...     name="filesystem",
        ...     transport="stdio",
        ...     command="npx",
        ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ...     tools=["read_file", "list_directory"]
        ... )

        Remote HTTP server:
        >>> config = MCPServerConfig(
        ...     name="api-server",
        ...     transport="http",
        ...     url="https://api.example.com/mcp",
        ...     tools=["fetch_weather", "geocode"]
        ... )

        .NET server with custom protocol version:
        >>> config = MCPServerConfig(
        ...     name="autodesk-aec",
        ...     transport="stdio",
        ...     command="dotnet",
        ...     args=["run", "--project", "server.csproj"],
        ...     protocol_version="2024-11-05",
        ...     tools=["GetToken", "GetHubs"]
        ... )
    """

    name: str
    transport: Literal["stdio", "http", "sse"]

    # Stdio transport fields
    command: Optional[str] = None
    args: Optional[List[str]] = None

    # HTTP/SSE transport fields
    url: Optional[str] = None

    # Common fields
    env: Optional[Dict[str, str]] = None
    tools: List[str] = field(default_factory=list)
    timeout: int = 30
    enabled: bool = True
    protocol_version: Optional[str] = None  # MCP protocol version override

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Normalize transport type
        if self.transport in ("sse",):
            self.transport = "http"

        # Validate stdio configuration
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"MCP server '{self.name}': 'command' is required for stdio transport")
            if not self.args:
                raise ValueError(f"MCP server '{self.name}': 'args' is required for stdio transport")

        # Validate HTTP configuration
        elif self.transport == "http":
            if not self.url:
                raise ValueError(f"MCP server '{self.name}': 'url' is required for HTTP transport")

        else:
            raise ValueError(
                f"MCP server '{self.name}': Invalid transport '{self.transport}'. "
                f"Must be 'stdio' or 'http'/'sse'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": self.args,
            "url": self.url,
            "env": self.env,
            "tools": self.tools,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary."""
        return cls(**data)
