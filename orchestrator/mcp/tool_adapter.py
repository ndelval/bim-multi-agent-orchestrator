"""
Adapter for converting MCP tools to agent-compatible callable tools.
"""

import asyncio
import logging
from typing import Any, Dict, List, Callable, Optional
from functools import wraps
import json

from .config import MCPServerConfig
from .client_manager import MCPClientManager

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """
    Converts MCP server tools to agent-compatible callable functions.

    This adapter creates Python functions that wrap MCP tool calls,
    making them compatible with agent tool systems.

    Example:
        >>> manager = MCPClientManager()
        >>> adapter = MCPToolAdapter(manager)
        >>> config = MCPServerConfig(...)
        >>> tools = await adapter.create_tools(config)
        >>> # tools is a list of callable functions
        >>> result = tools[0](arg1="value1")
    """

    def __init__(self, client_manager: MCPClientManager):
        """
        Initialize the tool adapter.

        Args:
            client_manager: MCP client manager instance
        """
        self.client_manager = client_manager
        self._tool_cache: Dict[str, List[Callable]] = {}

        logger.debug("MCPToolAdapter initialized")

    async def create_tools(
        self, config: MCPServerConfig, filter_tools: Optional[List[str]] = None
    ) -> List[Callable]:
        """
        Create agent-compatible tools from an MCP server.

        Args:
            config: MCP server configuration
            filter_tools: Optional list of tool names to include (uses config.tools if None)

        Returns:
            List of callable tool functions

        Raises:
            RuntimeError: If tool creation fails
        """
        # Use configured tools or filter
        tool_filter = filter_tools or config.tools

        # Check cache
        cache_key = f"{config.name}:{','.join(sorted(tool_filter))}"
        if cache_key in self._tool_cache:
            logger.debug(f"Using cached tools for '{config.name}'")
            return self._tool_cache[cache_key]

        try:
            # List available tools from server
            logger.info(f"Listing tools from MCP server '{config.name}'")
            available_tools = await self.client_manager.list_tools(config)

            # Filter tools if configured
            if tool_filter:
                available_tools = [
                    tool for tool in available_tools if tool["name"] in tool_filter
                ]

            if not available_tools:
                logger.warning(
                    f"No tools found for '{config.name}' with filter: {tool_filter}"
                )
                return []

            # Create callable wrappers
            tools = []
            for tool_def in available_tools:
                tool_func = self._create_tool_function(config, tool_def)
                tools.append(tool_func)

            # Cache result
            self._tool_cache[cache_key] = tools

            logger.info(f"Created {len(tools)} tool(s) from MCP server '{config.name}'")
            return tools

        except Exception as e:
            logger.error(f"Failed to create tools from '{config.name}': {e}")
            raise RuntimeError(f"Failed to create MCP tools from '{config.name}': {e}")

    def _enrich_description(
        self, description: str, input_schema: Dict[str, Any]
    ) -> str:
        """Append parameter documentation from MCP input_schema to description."""
        properties = input_schema.get("properties", {})
        if not properties:
            return description

        required = set(input_schema.get("required", []))
        lines = [description, "", "Parameters:"]
        for name, prop in properties.items():
            ptype = prop.get("type", "any")
            req = "required" if name in required else "optional"
            desc = prop.get("description", "")
            constraints: List[str] = []
            if "enum" in prop:
                constraints.append(f"one of {prop['enum']}")
            if "minimum" in prop:
                constraints.append(f"min={prop['minimum']}")
            if "maximum" in prop:
                constraints.append(f"max={prop['maximum']}")
            if "minLength" in prop:
                constraints.append(f"minLength={prop['minLength']}")
            suffix = f" ({', '.join(constraints)})" if constraints else ""
            lines.append(f"- {name} ({ptype}, {req}): {desc}{suffix}")

        return "\n".join(lines)

    def _create_tool_function(
        self, config: MCPServerConfig, tool_def: Dict[str, Any]
    ) -> Callable:
        """
        Create a callable Python function wrapping an MCP tool.

        Returns an async function that works correctly in async contexts.

        Args:
            config: MCP server configuration
            tool_def: Tool definition from MCP server

        Returns:
            Async callable function that invokes the MCP tool
        """
        tool_name = tool_def["name"]
        input_schema = tool_def.get("input_schema", {})
        tool_description = self._enrich_description(
            tool_def.get("description", f"MCP tool: {tool_name}"),
            input_schema,
        )

        # Create async wrapper (primary version)
        async def mcp_tool_async(**kwargs) -> str:
            """Async MCP tool invocation."""
            try:
                logger.debug(
                    f"Calling MCP tool '{tool_name}' on '{config.name}' "
                    f"with args: {kwargs}"
                )

                # Call the MCP tool
                result = await self.client_manager.call_tool(
                    config=config, tool_name=tool_name, arguments=kwargs
                )

                # Format result
                formatted_result = self._format_tool_result(result)

                logger.debug(
                    f"MCP tool '{tool_name}' returned: {formatted_result[:200]}"
                )
                return formatted_result

            except Exception as e:
                error_msg = f"Error calling MCP tool '{tool_name}': {str(e)}"
                logger.error(error_msg)
                return error_msg

        # Attach metadata for tool introspection
        mcp_tool_async.__name__ = tool_name
        mcp_tool_async.__doc__ = tool_description
        mcp_tool_async.mcp_server = config.name
        mcp_tool_async.mcp_tool_name = tool_name
        mcp_tool_async.mcp_input_schema = input_schema
        mcp_tool_async.is_mcp_tool = True
        mcp_tool_async.is_async = True

        logger.debug(
            f"Created async tool function for '{tool_name}' from '{config.name}'"
        )

        # Return async version as primary
        return mcp_tool_async

    def _format_tool_result(self, result: Any) -> str:
        """
        Format MCP tool result for agent consumption.

        Args:
            result: Raw MCP tool result

        Returns:
            Formatted string result
        """
        try:
            # Handle different result types from MCP
            if hasattr(result, "content"):
                # Result has content attribute (common MCP response)
                content = result.content

                if isinstance(content, list):
                    # Multiple content items
                    parts = []
                    for item in content:
                        if hasattr(item, "text"):
                            parts.append(item.text)
                        elif hasattr(item, "data"):
                            parts.append(str(item.data))
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)

                elif hasattr(content, "text"):
                    return content.text

                elif hasattr(content, "data"):
                    return str(content.data)

                else:
                    return str(content)

            elif isinstance(result, (dict, list)):
                # JSON-serializable result
                return json.dumps(result, indent=2)

            else:
                # Fallback to string conversion
                return str(result)

        except Exception as e:
            logger.warning(f"Error formatting MCP result: {e}")
            return str(result)

    def clear_cache(self) -> None:
        """Clear the tool cache."""
        self._tool_cache.clear()
        logger.debug("Tool cache cleared")

    def get_cached_tools(self, config: MCPServerConfig) -> Optional[List[Callable]]:
        """
        Get cached tools for a server if available.

        Args:
            config: MCP server configuration

        Returns:
            Cached tools or None if not cached
        """
        tool_filter = config.tools
        cache_key = f"{config.name}:{','.join(sorted(tool_filter))}"
        return self._tool_cache.get(cache_key)
