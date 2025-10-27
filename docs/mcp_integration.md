# MCP (Model Context Protocol) Integration

## Overview

The orchestrator now supports MCP (Model Context Protocol) integration, allowing agents to connect to and use tools from MCP servers. This enables agents to access external data sources, APIs, and services through a standardized protocol.

## Features

- ✅ **Stdio Transport**: Connect to local MCP servers via subprocess (npm, python, node)
- ✅ **HTTP/SSE Transport**: Connect to remote MCP servers via HTTP with Server-Sent Events
- ✅ **Lazy Initialization**: Connections created on-demand, minimizing resource usage
- ✅ **Manual Tool Configuration**: Explicitly specify which tools each agent can access
- ✅ **Multiple Servers**: Agents can connect to multiple MCP servers simultaneously
- ✅ **Proper Cleanup**: Automatic cleanup of connections on orchestrator shutdown

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgentFactory                                │
│  ┌──────────────────┐     ┌──────────────────┐                 │
│  │ MCPClientManager │────▶│  MCPToolAdapter  │                 │
│  └──────────────────┘     └──────────────────┘                 │
│           │                        │                            │
│           │                        ▼                            │
│           │                  Convert tools                      │
│           ▼                        │                            │
│  Connect to servers               ▼                            │
└───────────────────────────────────│─────────────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │  Agent Tools  │
                            └───────────────┘
```

## Quick Start

### 1. Install MCP SDK

The MCP SDK is already included in the dependencies:

```bash
# Install or upgrade dependencies
uv pip install -e .
```

### 2. Configure an MCP Server

#### Stdio (Local Server)

```python
from orchestrator.mcp.config import MCPServerConfig

# Filesystem MCP server (local)
filesystem_server = MCPServerConfig(
    name="filesystem",
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    tools=["read_file", "list_directory", "write_file"]
)
```

#### HTTP/SSE (Remote Server)

```python
# API MCP server (remote)
api_server = MCPServerConfig(
    name="api-server",
    transport="http",
    url="https://api.example.com/mcp",
    tools=["fetch_data", "process_request"]
)
```

### 3. Attach MCP Server to Agent

```python
from orchestrator.core.config import AgentConfig

agent_config = AgentConfig(
    name="FileManager",
    role="File System Specialist",
    goal="Manage files using MCP tools",
    backstory="Expert at file operations",
    instructions="Use MCP filesystem tools efficiently",
    mcp_servers=[filesystem_server],  # Attach MCP server(s)
    tools=[]  # Can also include native tools
)
```

### 4. Run the Orchestrator

```python
from orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    name="MCP Demo",
    agents=[agent_config],
    tasks=[task_config]
)

orchestrator = Orchestrator(config)
try:
    result = orchestrator.run_sync()
    print(result)
finally:
    orchestrator.cleanup()  # Clean up MCP connections
```

## Configuration Options

### MCPServerConfig

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Unique identifier for the server |
| `transport` | "stdio" \| "http" | Yes | Transport mechanism |
| `command` | str | Stdio only | Command to run (e.g., "npx", "python") |
| `args` | List[str] | Stdio only | Command arguments |
| `url` | str | HTTP only | Server URL |
| `env` | Dict[str, str] | No | Environment variables |
| `tools` | List[str] | No | Tool names to expose (empty = all) |
| `timeout` | int | No | Connection timeout (default: 30s) |
| `enabled` | bool | No | Enable/disable server (default: True) |

## Usage Patterns

### Pattern 1: Single MCP Server

```python
agent = AgentConfig(
    name="Analyst",
    role="Data Analyst",
    goal="Analyze data",
    backstory="Expert analyst",
    instructions="Use tools to analyze data",
    mcp_servers=[filesystem_server]
)
```

### Pattern 2: Multiple MCP Servers

```python
agent = AgentConfig(
    name="DataExpert",
    role="Multi-Source Data Expert",
    goal="Analyze data from multiple sources",
    backstory="Versatile data specialist",
    instructions="Use all available tools",
    mcp_servers=[
        filesystem_server,
        sqlite_server,
        api_server
    ]
)
```

### Pattern 3: Mix MCP and Native Tools

```python
agent = AgentConfig(
    name="Researcher",
    role="Research Specialist",
    goal="Research and document findings",
    backstory="Skilled researcher",
    instructions="Use web search and file tools",
    mcp_servers=[filesystem_server],
    tools=["duckduckgo", "wikipedia"]  # Native tools
)
```

### Pattern 4: JSON Configuration

```json
{
  "agents": [
    {
      "name": "FileAgent",
      "role": "File Specialist",
      "goal": "Manage files",
      "backstory": "File expert",
      "instructions": "Use filesystem tools",
      "mcp_servers": [
        {
          "name": "filesystem",
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "tools": ["read_file", "list_directory"]
        }
      ]
    }
  ]
}
```

## Available MCP Servers

### Official Servers (by Anthropic)

- **@modelcontextprotocol/server-filesystem**: File operations
- **@modelcontextprotocol/server-sqlite**: SQLite database access
- **@modelcontextprotocol/server-postgres**: PostgreSQL database access
- **@modelcontextprotocol/server-github**: GitHub API integration
- **@modelcontextprotocol/server-slack**: Slack API integration

### Installation

```bash
# Install via npx (no permanent installation needed)
npx -y @modelcontextprotocol/server-filesystem /path/to/directory

# Or install globally
npm install -g @modelcontextprotocol/server-filesystem
```

## Best Practices

### 1. Explicit Tool Lists

Always specify which tools agents should have access to:

```python
# ✅ Good: Explicit tool list
mcp_servers=[
    MCPServerConfig(
        name="filesystem",
        tools=["read_file", "list_directory"]  # Only these tools
    )
]

# ❌ Avoid: Empty tool list (exposes all tools)
mcp_servers=[
    MCPServerConfig(
        name="filesystem",
        tools=[]  # All tools exposed
    )
]
```

### 2. Proper Cleanup

Always clean up MCP connections:

```python
orchestrator = Orchestrator(config)
try:
    result = orchestrator.run_sync()
finally:
    orchestrator.cleanup()  # Critical for connection cleanup
```

### 3. Error Handling

MCP integration is designed to fail gracefully:

```python
# If MCP SDK not installed, agents work with native tools only
# If MCP server fails to start, agent continues with remaining tools
# Connection failures are logged but don't crash the orchestrator
```

### 4. Environment Variables

Use environment variables for sensitive configuration:

```python
import os

mcp_server = MCPServerConfig(
    name="api-server",
    transport="http",
    url=os.getenv("MCP_SERVER_URL"),
    env={
        "API_KEY": os.getenv("API_KEY")
    }
)
```

## Troubleshooting

### MCP SDK Not Installed

```
ImportError: MCP SDK not installed. Install with: pip install mcp
```

**Solution**: Install the MCP SDK:
```bash
uv pip install mcp>=1.18.0
```

### Server Not Found

```
Failed to connect to MCP server 'filesystem': Command not found
```

**Solution**: Install the MCP server:
```bash
npm install -g @modelcontextprotocol/server-filesystem
```

### Connection Timeout

```
RuntimeError: Failed to connect to MCP server 'api-server': Connection timeout
```

**Solution**: Increase timeout or check server availability:
```python
mcp_server = MCPServerConfig(
    name="api-server",
    timeout=60,  # Increase timeout
    ...
)
```

### Tool Not Found

```
Warning: No tools found for 'filesystem' with filter: ['nonexistent_tool']
```

**Solution**: List available tools first or check tool names:
```bash
# Check what tools the server provides
npx -y @modelcontextprotocol/server-filesystem --help
```

## Examples

See `examples/mcp_example.py` for comprehensive examples:

- Stdio MCP server (local filesystem)
- HTTP/SSE MCP server (remote API)
- Multiple MCP servers on one agent
- JSON configuration
- Programmatic configuration

## API Reference

### orchestrator.mcp.config

- `MCPServerConfig`: MCP server configuration dataclass
- `MCPTransportType`: Enum for transport types

### orchestrator.mcp.client_manager

- `MCPClientManager`: Manages MCP server connections
  - `get_session(config)`: Get or create client session
  - `list_tools(config)`: List available tools
  - `call_tool(config, tool_name, arguments)`: Call a tool
  - `cleanup()`: Close all connections

### orchestrator.mcp.tool_adapter

- `MCPToolAdapter`: Converts MCP tools to agent tools
  - `create_tools(config)`: Create callable tools
  - `clear_cache()`: Clear tool cache

## Future Enhancements

- [ ] Async tool invocation from sync contexts
- [ ] Tool caching with TTL
- [ ] Connection pooling and reuse
- [ ] Health checks for MCP servers
- [ ] Tool usage metrics and monitoring
- [ ] WebSocket transport support
