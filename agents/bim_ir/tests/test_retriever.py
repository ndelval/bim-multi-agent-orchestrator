"""
Test suite for Retriever (Block 4).

Tests BIM element retrieval via bim.query MCP tool integration.
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.bim_ir.blocks.retriever import Retriever, RetrieverFactory
from agents.bim_ir.models.resolved_values import ResolvedValues, NormalizedValue
from agents.bim_ir.models.retriever_result import (
    RetrieverResult,
    BIMElement,
    QueryMetadata,
    QuerySummary,
    MCPConnectionError,
    MCPToolError,
    ResponseParseError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MockTextContent:
    """Mock text content for MCP response."""
    text: str


@dataclass
class MockMCPResponse:
    """Mock MCP tool response."""
    content: List[MockTextContent]


class MockMCPManager:
    """
    Mock MCP client manager for testing without real MCP server.

    Simulates bim.query MCP tool responses for testing.
    """

    def __init__(self, mock_response_data: Optional[Dict[str, Any]] = None):
        """
        Initialize mock MCP manager.

        Args:
            mock_response_data: Optional dict of response data to return
        """
        self.calls = []  # Track all calls
        self.mock_response_data = mock_response_data or self._default_response()

    def _default_response(self) -> Dict[str, Any]:
        """Default mock response with sample BIM elements."""
        return {
            "query_metadata": {
                "model_urn": "urn:adsk.wipprod:fs.file:vf.test123",
                "viewable_guid": "abc-def-123",
                "viewable_name": "3D View",
                "filter_count": 2,
                "projection_count": 3,
                "limit": 100
            },
            "elements": [
                {
                    "element_id": 12345,
                    "name": "Basic Wall: Exterior - Brick on CMU",
                    "properties": {
                        "Category": "Walls",
                        "Level": "Level 1",
                        "Volume": 15.5,
                        "Area": 45.2,
                        "Material": "Concrete"
                    }
                },
                {
                    "element_id": 12346,
                    "name": "Basic Wall: Interior - Partition",
                    "properties": {
                        "Category": "Walls",
                        "Level": "Level 1",
                        "Volume": 8.3,
                        "Area": 28.9,
                        "Material": "Gypsum"
                    }
                }
            ],
            "summary": {
                "total_matched": 2,
                "returned_count": 2,
                "filter_conditions": "Category = 'Walls' AND Level = 'Level 1'",
                "requested_properties": ["Name", "Volume", "Area"]
            }
        }

    async def call_tool(
        self,
        config: Any,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> MockMCPResponse:
        """
        Simulate MCP tool call.

        Args:
            config: MCP server config (ignored in mock)
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Mock MCP response
        """
        # Track call
        self.calls.append({
            "tool_name": tool_name,
            "arguments": arguments
        })

        # Return mock response
        response_json = json.dumps(self.mock_response_data)
        return MockMCPResponse(
            content=[MockTextContent(text=response_json)]
        )

    async def list_tools(self, config: Any) -> List[Dict[str, str]]:
        """List available tools (mock)."""
        return [
            {
                "name": "bimQueryTool",
                "description": "Query BIM elements",
                "input_schema": {}
            }
        ]


class MockAPSConfig:
    """Mock APS MCP server config."""

    def __init__(self):
        self.name = "mock-aps-mcp"
        self.transport = "stdio"
        self.command = "node"
        self.args = ["server.js"]


def test_build_query_arguments():
    """Test query argument building from resolved values."""
    print("\n" + "="*70)
    print("Testing Query Argument Building")
    print("="*70)

    # Create mock inputs
    resolved_values = ResolvedValues(
        filter_values=[
            NormalizedValue("Category", "wall", "Walls", "synonym", 0.95),
            NormalizedValue("Level", "ground floor", "Level 1", "synonym", 0.95)
        ]
    )

    projection_parameters = ["Name", "Volume", "Area"]

    # Create retriever
    mock_manager = MockMCPManager()
    mock_config = MockAPSConfig()
    retriever = Retriever(mock_manager, mock_config, "urn:test")

    # Build arguments
    query_args = retriever._build_query_arguments(
        resolved_values,
        projection_parameters,
        limit=100
    )

    # Validate
    assert query_args["model_urn"] == "urn:test"
    assert len(query_args["filter_parameters"]) == 2
    assert query_args["filter_parameters"][0]["property_name"] == "Category"
    assert query_args["filter_parameters"][0]["value"] == "Walls"
    assert query_args["projection_parameters"] == projection_parameters
    assert query_args["limit"] == 100

    print("✅ Query arguments built correctly")
    print(f"   Filters: {len(query_args['filter_parameters'])}")
    print(f"   Projections: {len(query_args['projection_parameters'])}")


async def test_successful_retrieval():
    """Test successful BIM element retrieval."""
    print("\n" + "="*70)
    print("Testing Successful Retrieval")
    print("="*70)

    # Create mock inputs
    resolved_values = ResolvedValues(
        filter_values=[
            NormalizedValue("Category", "wall", "Walls", "synonym", 0.95),
            NormalizedValue("Level", "level 1", "Level 1", "case_insensitive", 0.99)
        ]
    )

    projection_parameters = ["Name", "Volume", "Area"]

    # Create retriever with mock
    mock_manager = MockMCPManager()
    mock_config = MockAPSConfig()
    retriever = Retriever(mock_manager, mock_config, "urn:test123")

    # Retrieve elements
    result = await retriever.retrieve(
        resolved_values=resolved_values,
        projection_parameters=projection_parameters,
        limit=100
    )

    # Validate result
    assert isinstance(result, RetrieverResult)
    assert result.element_count == 2
    assert not result.is_empty
    assert result.summary.total_matched == 2
    assert result.summary.returned_count == 2
    assert not result.summary.is_truncated

    # Validate elements
    assert all(isinstance(elem, BIMElement) for elem in result.elements)
    assert result.elements[0].name == "Basic Wall: Exterior - Brick on CMU"
    assert result.elements[0].get_property("Category") == "Walls"
    assert result.elements[0].get_property("Volume") == 15.5

    # Check MCP call was made
    assert len(mock_manager.calls) == 1
    assert mock_manager.calls[0]["tool_name"] == "bimQueryTool"

    print("✅ Retrieval successful")
    print(f"   Elements: {result.element_count}")
    print(f"   Total matched: {result.summary.total_matched}")
    print(f"   First element: {result.elements[0].name}")


async def test_empty_results():
    """Test handling of empty results (no matching elements)."""
    print("\n" + "="*70)
    print("Testing Empty Results")
    print("="*70)

    # Create mock with empty results
    empty_response = {
        "query_metadata": {
            "model_urn": "urn:test",
            "viewable_guid": "abc",
            "viewable_name": "3D",
            "filter_count": 1,
            "projection_count": 2,
            "limit": 100
        },
        "elements": [],
        "summary": {
            "total_matched": 0,
            "returned_count": 0,
            "filter_conditions": "Category = 'NonExistent'",
            "requested_properties": ["Name"]
        }
    }

    mock_manager = MockMCPManager(mock_response_data=empty_response)
    mock_config = MockAPSConfig()
    retriever = Retriever(mock_manager, mock_config, "urn:test")

    # Retrieve with filters that match nothing
    resolved_values = ResolvedValues(
        filter_values=[
            NormalizedValue("Category", "nonexistent", "NonExistent", "exact", 1.0)
        ]
    )

    result = await retriever.retrieve(
        resolved_values=resolved_values,
        projection_parameters=["Name"],
        limit=100
    )

    # Validate
    assert result.is_empty
    assert result.element_count == 0
    assert result.summary.total_matched == 0

    print("✅ Empty results handled correctly")
    print(f"   Elements: {result.element_count}")


async def test_truncated_results():
    """Test handling of truncated results (limit < total_matched)."""
    print("\n" + "="*70)
    print("Testing Truncated Results")
    print("="*70)

    # Create mock with truncated results
    truncated_response = {
        "query_metadata": {
            "model_urn": "urn:test",
            "viewable_guid": "abc",
            "viewable_name": "3D",
            "filter_count": 1,
            "projection_count": 2,
            "limit": 2
        },
        "elements": [
            {
                "element_id": 1,
                "name": "Wall 1",
                "properties": {"Category": "Walls"}
            },
            {
                "element_id": 2,
                "name": "Wall 2",
                "properties": {"Category": "Walls"}
            }
        ],
        "summary": {
            "total_matched": 100,  # 100 total, but only 2 returned
            "returned_count": 2,
            "filter_conditions": "Category = 'Walls'",
            "requested_properties": ["Name"]
        }
    }

    mock_manager = MockMCPManager(mock_response_data=truncated_response)
    mock_config = MockAPSConfig()
    retriever = Retriever(mock_manager, mock_config, "urn:test")

    resolved_values = ResolvedValues(
        filter_values=[NormalizedValue("Category", "Walls", "Walls", "exact", 1.0)]
    )

    result = await retriever.retrieve(
        resolved_values=resolved_values,
        projection_parameters=["Name"],
        limit=2
    )

    # Validate truncation
    assert result.summary.is_truncated
    assert result.summary.total_matched == 100
    assert result.summary.returned_count == 2
    assert result.element_count == 2
    assert result.summary.truncation_ratio == 0.02  # 2/100

    print("✅ Truncated results handled correctly")
    print(f"   Returned: {result.summary.returned_count}")
    print(f"   Total matched: {result.summary.total_matched}")
    print(f"   Truncation ratio: {result.summary.truncation_ratio:.2%}")


async def test_error_handling():
    """Test error handling for various failure scenarios."""
    print("\n" + "="*70)
    print("Testing Error Handling")
    print("="*70)

    # Test 1: Invalid MCP response (malformed JSON)
    class BadJSONMockManager(MockMCPManager):
        async def call_tool(self, config, tool_name, arguments):
            return MockMCPResponse(
                content=[MockTextContent(text="invalid json {{{")]
            )

    bad_json_manager = BadJSONMockManager()
    retriever = Retriever(bad_json_manager, MockAPSConfig(), "urn:test")

    try:
        await retriever.retrieve(
            ResolvedValues(filter_values=[]),
            ["Name"],
            limit=10
        )
        assert False, "Should have raised ResponseParseError"
    except ResponseParseError as e:
        print("✅ Malformed JSON handled correctly")
        print(f"   Error: {str(e)[:60]}...")

    # Test 2: Missing required fields
    class MissingFieldsMockManager(MockMCPManager):
        async def call_tool(self, config, tool_name, arguments):
            return MockMCPResponse(
                content=[MockTextContent(text='{"elements": []}')]  # Missing other fields
            )

    missing_fields_manager = MissingFieldsMockManager()
    retriever2 = Retriever(missing_fields_manager, MockAPSConfig(), "urn:test")

    try:
        await retriever2.retrieve(
            ResolvedValues(filter_values=[]),
            ["Name"],
            limit=10
        )
        assert False, "Should have raised ResponseParseError"
    except ResponseParseError as e:
        print("✅ Missing fields handled correctly")
        print(f"   Error: {str(e)[:60]}...")

    # Test 3: Empty content
    class EmptyContentMockManager(MockMCPManager):
        async def call_tool(self, config, tool_name, arguments):
            return MockMCPResponse(content=[])

    empty_manager = EmptyContentMockManager()
    retriever3 = Retriever(empty_manager, MockAPSConfig(), "urn:test")

    try:
        await retriever3.retrieve(
            ResolvedValues(filter_values=[]),
            ["Name"],
            limit=10
        )
        assert False, "Should have raised ResponseParseError"
    except ResponseParseError as e:
        print("✅ Empty content handled correctly")
        print(f"   Error: {str(e)[:60]}...")


async def test_result_filtering():
    """Test filtering and querying methods on RetrieverResult."""
    print("\n" + "="*70)
    print("Testing Result Filtering Methods")
    print("="*70)

    # Create retriever and get results
    mock_manager = MockMCPManager()
    retriever = Retriever(mock_manager, MockAPSConfig(), "urn:test")

    result = await retriever.retrieve(
        ResolvedValues(filter_values=[
            NormalizedValue("Category", "Walls", "Walls", "exact", 1.0)
        ]),
        ["Name", "Material"],
        limit=100
    )

    # Test get_elements_by_property
    concrete_walls = result.get_elements_by_property("Material", "Concrete")
    assert len(concrete_walls) == 1
    assert concrete_walls[0].name == "Basic Wall: Exterior - Brick on CMU"

    # Test get_property_values
    materials = result.get_property_values("Material")
    assert set(materials) == {"Concrete", "Gypsum"}

    # Test to_dict
    result_dict = result.to_dict()
    assert "elements" in result_dict
    assert "query_metadata" in result_dict
    assert "summary" in result_dict
    assert len(result_dict["elements"]) == 2

    print("✅ Result filtering methods work correctly")
    print(f"   Concrete walls: {len(concrete_walls)}")
    print(f"   Unique materials: {len(materials)}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Retriever Test Suite (Block 4)")
    print("="*70)

    # Sync tests
    test_build_query_arguments()

    # Async tests
    asyncio.run(test_successful_retrieval())
    asyncio.run(test_empty_results())
    asyncio.run(test_truncated_results())
    asyncio.run(test_error_handling())
    asyncio.run(test_result_filtering())

    print("\n" + "="*70)
    print("All Tests Passed! ✅")
    print("="*70)


if __name__ == "__main__":
    main()
