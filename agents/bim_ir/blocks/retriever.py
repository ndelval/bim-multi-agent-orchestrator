"""
Retriever (Block 4) for BIM-IR Agent.

Executes BIM queries via bim.query MCP tool and returns structured results.
Integrates with Autodesk APS Model Derivative API via MCP for element retrieval.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from orchestrator.mcp.client_manager import MCPClientManager
from orchestrator.mcp.config import MCPServerConfig

from ..models.resolved_values import ResolvedValues
from ..models.retriever_result import (
    RetrieverResult,
    BIMElement,
    QueryMetadata,
    QuerySummary,
    MCPConnectionError,
    MCPToolError,
    BIMQueryError,
    ResponseParseError
)

logger = logging.getLogger(__name__)


class Retriever:
    """
    Block 4: BIM Element Retrieval

    Executes BIM queries using the bim.query MCP tool to retrieve matching
    elements from BIM models. Integrates normalized values from Block 3 with
    MCP infrastructure to query Autodesk Model Derivative API.

    The retriever:
    1. Builds structured query from normalized values
    2. Calls bim.query MCP tool with filter and projection parameters
    3. Parses response into structured RetrieverResult
    4. Handles errors and validates results

    Example:
        >>> from orchestrator.mcp.client_manager import MCPClientManager
        >>> from orchestrator.mcp.config import MCPServerConfig
        >>>
        >>> # Configure APS MCP server
        >>> aps_config = MCPServerConfig(
        ...     name="aps-mcp",
        ...     transport="stdio",
        ...     command="node",
        ...     args=["/path/to/aps-mcp-server/server.js"],
        ...     env={"APS_CLIENT_ID": "...", "APS_CLIENT_SECRET": "..."}
        ... )
        >>>
        >>> # Initialize retriever
        >>> mcp_manager = MCPClientManager()
        >>> retriever = Retriever(
        ...     mcp_manager=mcp_manager,
        ...     aps_config=aps_config,
        ...     model_urn="urn:adsk.wipprod:fs.file:vf.xyz"
        ... )
        >>>
        >>> # Retrieve elements
        >>> result = await retriever.retrieve(
        ...     resolved_values=resolved,  # From Block 3
        ...     projection_parameters=["Name", "Volume", "Area"]
        ... )
        >>> print(f"Found {result.element_count} elements")
    """

    DEFAULT_TOOL_NAME = "bimQueryTool"
    DEFAULT_LIMIT = 100
    DEFAULT_PROJECTION = ["Name", "Category"]
    DEFAULT_OPERATOR = "="

    def __init__(
        self,
        mcp_manager: MCPClientManager,
        aps_config: MCPServerConfig,
        model_urn: str
    ):
        """
        Initialize Retriever.

        Args:
            mcp_manager: MCP client manager instance for server connections
            aps_config: Configuration for APS MCP server
            model_urn: URN of the BIM model to query

        Raises:
            ValueError: If model_urn is empty or invalid
        """
        if not model_urn or not model_urn.strip():
            raise ValueError("model_urn cannot be empty")

        self.mcp_manager = mcp_manager
        self.aps_config = aps_config
        self.model_urn = model_urn

        logger.info(f"Initialized Retriever for model: {model_urn}")
        logger.info(f"Using MCP server: {aps_config.name}")

    async def retrieve(
        self,
        resolved_values: ResolvedValues,
        projection_parameters: List[str],
        limit: int = DEFAULT_LIMIT,
        max_retries: int = 3
    ) -> RetrieverResult:
        """
        Retrieve BIM elements matching query criteria.

        Builds a structured BIM query from normalized values and executes it
        via the bim.query MCP tool.

        Args:
            resolved_values: Normalized filter values from Block 3
            projection_parameters: Properties to retrieve (from Block 2)
            limit: Maximum number of elements to return (default: 100)
            max_retries: Maximum retry attempts for MCP calls (default: 3)

        Returns:
            RetrieverResult with matching elements and metadata

        Raises:
            MCPConnectionError: If MCP server connection fails
            MCPToolError: If MCP tool call fails
            BIMQueryError: If BIM query execution fails
            ResponseParseError: If response parsing fails
        """
        logger.info(
            f"Retrieving BIM elements: {len(resolved_values.filter_values)} filters, "
            f"{len(projection_parameters)} projections, limit={limit}"
        )

        # Build query arguments
        query_args = self._build_query_arguments(
            resolved_values,
            projection_parameters,
            limit
        )

        logger.debug(f"Query arguments: {json.dumps(query_args, indent=2)}")

        # Call MCP tool with retry logic
        mcp_response = await self._call_mcp_tool_with_retry(
            query_args,
            max_retries
        )

        # Parse response into RetrieverResult
        result = self._parse_mcp_response(mcp_response, query_args)

        logger.info(
            f"Retrieved {result.element_count} elements "
            f"({result.summary.total_matched} total matched)"
        )

        return result

    def _build_query_arguments(
        self,
        resolved_values: ResolvedValues,
        projection_parameters: List[str],
        limit: int
    ) -> Dict[str, Any]:
        """
        Build MCP tool arguments from resolved values and projections.

        Args:
            resolved_values: Normalized filter values
            projection_parameters: Properties to retrieve
            limit: Maximum results

        Returns:
            Dictionary of arguments for bim.query MCP tool
        """
        # Build filter_parameters from resolved values
        filter_parameters = []
        for norm_value in resolved_values.filter_values:
            filter_parameters.append({
                "property_name": norm_value.property_name,
                "operator": self.DEFAULT_OPERATOR,
                "value": norm_value.normalized_value
            })

        # Use provided projections or defaults
        proj_params = projection_parameters if projection_parameters else self.DEFAULT_PROJECTION

        # Build complete query
        query_args = {
            "model_urn": self.model_urn,
            "filter_parameters": filter_parameters,
            "projection_parameters": proj_params,
            "limit": limit
        }

        return query_args

    async def _call_mcp_tool_with_retry(
        self,
        query_args: Dict[str, Any],
        max_retries: int
    ) -> Any:
        """
        Call MCP tool with exponential backoff retry logic.

        Args:
            query_args: Arguments for the MCP tool
            max_retries: Maximum number of retry attempts

        Returns:
            MCP tool response

        Raises:
            MCPConnectionError: If connection fails after all retries
            MCPToolError: If tool call fails after all retries
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"MCP tool call attempt {attempt}/{max_retries}")

                result = await self.mcp_manager.call_tool(
                    config=self.aps_config,
                    tool_name=self.DEFAULT_TOOL_NAME,
                    arguments=query_args
                )

                logger.debug(f"MCP tool call succeeded on attempt {attempt}")
                return result

            except RuntimeError as e:
                last_error = e
                error_msg = str(e).lower()

                # Check if connection error
                if "connect" in error_msg or "timeout" in error_msg:
                    logger.warning(f"MCP connection error on attempt {attempt}: {e}")

                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # Non-retryable error
                    logger.error(f"Non-retryable MCP error: {e}")
                    raise MCPToolError(f"MCP tool call failed: {e}") from e

            except Exception as e:
                logger.error(f"Unexpected error calling MCP tool: {e}")
                raise MCPToolError(f"Unexpected MCP error: {e}") from e

        # All retries exhausted
        logger.error(f"MCP tool call failed after {max_retries} attempts")
        raise MCPConnectionError(
            f"Failed to call MCP tool after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_mcp_response(
        self,
        mcp_response: Any,
        query_args: Dict[str, Any]
    ) -> RetrieverResult:
        """
        Parse MCP tool response into RetrieverResult.

        The MCP tool returns a response with .content attribute containing
        a list of TextContent objects. The first TextContent.text contains
        JSON-formatted query results.

        Args:
            mcp_response: Response from MCP tool call
            query_args: Original query arguments for context

        Returns:
            Parsed RetrieverResult

        Raises:
            ResponseParseError: If parsing fails
        """
        try:
            # Extract text content from MCP response
            if not hasattr(mcp_response, 'content'):
                raise ResponseParseError("MCP response missing 'content' attribute")

            if not mcp_response.content or len(mcp_response.content) == 0:
                raise ResponseParseError("MCP response content is empty")

            # Get first text content
            text_content = mcp_response.content[0]

            if not hasattr(text_content, 'text'):
                raise ResponseParseError("MCP content missing 'text' attribute")

            # Parse JSON
            try:
                data = json.loads(text_content.text)
            except json.JSONDecodeError as e:
                raise ResponseParseError(f"Invalid JSON in MCP response: {e}") from e

            # Validate required fields
            required_fields = ["query_metadata", "elements", "summary"]
            for field in required_fields:
                if field not in data:
                    raise ResponseParseError(f"Missing required field: {field}")

            # Parse elements
            elements = []
            for elem_data in data["elements"]:
                try:
                    element = BIMElement(
                        element_id=elem_data["element_id"],
                        name=elem_data["name"],
                        properties=elem_data.get("properties", {})
                    )
                    elements.append(element)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid element: {e}")
                    continue

            # Parse metadata
            metadata_data = data["query_metadata"]
            query_metadata = QueryMetadata(
                model_urn=metadata_data["model_urn"],
                viewable_guid=metadata_data["viewable_guid"],
                viewable_name=metadata_data["viewable_name"],
                filter_count=metadata_data["filter_count"],
                projection_count=metadata_data["projection_count"],
                limit=metadata_data["limit"]
            )

            # Parse summary
            summary_data = data["summary"]
            query_summary = QuerySummary(
                total_matched=summary_data["total_matched"],
                returned_count=summary_data["returned_count"],
                filter_conditions=summary_data["filter_conditions"],
                requested_properties=summary_data["requested_properties"]
            )

            # Create result
            result = RetrieverResult(
                elements=elements,
                query_metadata=query_metadata,
                summary=query_summary
            )

            logger.debug(f"Successfully parsed MCP response: {result}")
            return result

        except ResponseParseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing MCP response: {e}")
            raise ResponseParseError(f"Failed to parse MCP response: {e}") from e

    async def validate_mcp_connection(self) -> bool:
        """
        Validate MCP server connection and tool availability.

        Returns:
            True if connection successful and tool available

        Raises:
            MCPConnectionError: If connection fails
            MCPToolError: If tool not found
        """
        try:
            logger.info(f"Validating MCP connection to '{self.aps_config.name}'")

            # List available tools
            tools = await self.mcp_manager.list_tools(self.aps_config)

            # Check if bimQueryTool exists
            tool_names = [tool["name"] for tool in tools]
            if self.DEFAULT_TOOL_NAME not in tool_names:
                raise MCPToolError(
                    f"Tool '{self.DEFAULT_TOOL_NAME}' not found on server. "
                    f"Available tools: {tool_names}"
                )

            logger.info(
                f"MCP connection validated. Found {len(tools)} tools including "
                f"'{self.DEFAULT_TOOL_NAME}'"
            )

            return True

        except RuntimeError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e
        except Exception as e:
            raise MCPConnectionError(f"Unexpected error validating connection: {e}") from e


class RetrieverFactory:
    """Factory for creating Retriever instances with configuration."""

    @staticmethod
    async def create_from_env(
        model_urn: str,
        mcp_manager: Optional[MCPClientManager] = None
    ) -> Retriever:
        """
        Create Retriever from environment variables.

        Expects the following environment variables:
        - APS_MCP_COMMAND: Command to run MCP server (default: "node")
        - APS_MCP_SERVER_PATH: Path to server.js
        - APS_CLIENT_ID: Autodesk APS client ID
        - APS_CLIENT_SECRET: Autodesk APS client secret

        Args:
            model_urn: URN of BIM model to query
            mcp_manager: Optional MCPClientManager instance (creates new if None)

        Returns:
            Configured Retriever instance

        Raises:
            EnvironmentError: If required environment variables missing
        """
        import os

        # Get environment variables
        server_path = os.getenv("APS_MCP_SERVER_PATH")
        if not server_path:
            raise EnvironmentError("APS_MCP_SERVER_PATH environment variable required")

        client_id = os.getenv("APS_CLIENT_ID")
        client_secret = os.getenv("APS_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise EnvironmentError(
                "APS_CLIENT_ID and APS_CLIENT_SECRET environment variables required"
            )

        # Build APS config
        aps_config = MCPServerConfig(
            name="aps-mcp",
            transport="stdio",
            command=os.getenv("APS_MCP_COMMAND", "node"),
            args=[server_path],
            env={
                "APS_CLIENT_ID": client_id,
                "APS_CLIENT_SECRET": client_secret
            }
        )

        # Create manager if not provided
        if mcp_manager is None:
            mcp_manager = MCPClientManager()

        # Create retriever
        retriever = Retriever(
            mcp_manager=mcp_manager,
            aps_config=aps_config,
            model_urn=model_urn
        )

        # Validate connection
        await retriever.validate_mcp_connection()

        logger.info(f"Created Retriever from environment for model: {model_urn}")

        return retriever
