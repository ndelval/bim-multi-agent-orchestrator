"""
Retriever Result Models (Block 4 Output).

Data models for BIM element retrieval results from bim.query MCP tool.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BIMElement:
    """
    Single BIM element with properties.

    Represents one element returned from BIM query with its ID, name, and properties.

    Attributes:
        element_id: Unique element identifier from BIM model
        name: Element name (e.g., "Basic Wall: Exterior - Brick on CMU")
        properties: Dictionary of property name -> value pairs

    Example:
        >>> element = BIMElement(
        ...     element_id=12345,
        ...     name="Basic Wall: Exterior - Brick on CMU",
        ...     properties={
        ...         "Category": "Walls",
        ...         "Level": "Level 1",
        ...         "Volume": 15.5,
        ...         "Area": 45.2,
        ...         "Material": "Concrete"
        ...     }
        ... )
    """

    element_id: int
    name: str
    properties: Dict[str, Any]

    def __post_init__(self):
        """Validate element data."""
        if self.element_id < 0:
            raise ValueError(f"element_id must be non-negative, got {self.element_id}")

        if not self.name or not self.name.strip():
            raise ValueError("Element name cannot be empty")

        if not isinstance(self.properties, dict):
            raise ValueError("properties must be a dictionary")

    def get_property(self, property_name: str, default: Any = None) -> Any:
        """
        Get property value with optional default.

        Args:
            property_name: Name of property to retrieve
            default: Default value if property not found

        Returns:
            Property value or default
        """
        return self.properties.get(property_name, default)

    def has_property(self, property_name: str) -> bool:
        """Check if element has a specific property."""
        return property_name in self.properties

    def __repr__(self) -> str:
        """Human-readable representation."""
        props_preview = list(self.properties.keys())[:3]
        props_str = ", ".join(props_preview)
        if len(self.properties) > 3:
            props_str += f", ... ({len(self.properties)} total)"

        return (
            f"BIMElement(id={self.element_id}, name='{self.name}', "
            f"properties=[{props_str}])"
        )


@dataclass
class QueryMetadata:
    """
    Metadata about query execution.

    Provides context about the BIM query that was executed.

    Attributes:
        model_urn: URN of the BIM model queried
        viewable_guid: GUID of the viewable within the model
        viewable_name: Human-readable name of the viewable
        filter_count: Number of filter conditions applied
        projection_count: Number of properties requested
        limit: Maximum number of results requested
    """

    model_urn: str
    viewable_guid: str
    viewable_name: str
    filter_count: int
    projection_count: int
    limit: int

    def __post_init__(self):
        """Validate metadata."""
        if not self.model_urn:
            raise ValueError("model_urn cannot be empty")

        if self.filter_count < 0:
            raise ValueError("filter_count must be non-negative")

        if self.projection_count < 0:
            raise ValueError("projection_count must be non-negative")

        if self.limit <= 0:
            raise ValueError("limit must be positive")


@dataclass
class QuerySummary:
    """
    Summary statistics for query results.

    Provides information about query execution and results.

    Attributes:
        total_matched: Total number of elements matching filter conditions
        returned_count: Number of elements actually returned (limited by limit param)
        filter_conditions: Human-readable string describing filter conditions
        requested_properties: List of properties that were requested
    """

    total_matched: int
    returned_count: int
    filter_conditions: str
    requested_properties: List[str]

    def __post_init__(self):
        """Validate summary."""
        if self.total_matched < 0:
            raise ValueError("total_matched must be non-negative")

        if self.returned_count < 0:
            raise ValueError("returned_count must be non-negative")

        if self.returned_count > self.total_matched:
            raise ValueError(
                f"returned_count ({self.returned_count}) cannot exceed "
                f"total_matched ({self.total_matched})"
            )

        if not isinstance(self.requested_properties, list):
            raise ValueError("requested_properties must be a list")

    @property
    def is_truncated(self) -> bool:
        """Check if results were truncated due to limit."""
        return self.returned_count < self.total_matched

    @property
    def truncation_ratio(self) -> float:
        """Calculate ratio of returned to total matched (0.0-1.0)."""
        if self.total_matched == 0:
            return 1.0
        return self.returned_count / self.total_matched


@dataclass
class RetrieverResult:
    """
    Complete result from BIM element retrieval (Block 4 output).

    Contains all information about retrieved BIM elements including the elements
    themselves, query metadata, and result summary.

    Attributes:
        elements: List of retrieved BIM elements
        query_metadata: Metadata about the query execution
        summary: Summary statistics for the results

    Example:
        >>> result = RetrieverResult(
        ...     elements=[
        ...         BIMElement(12345, "Wall 1", {"Category": "Walls"}),
        ...         BIMElement(12346, "Wall 2", {"Category": "Walls"})
        ...     ],
        ...     query_metadata=QueryMetadata(
        ...         model_urn="urn:adsk.wipprod:fs.file:vf.xyz",
        ...         viewable_guid="abc-123",
        ...         viewable_name="3D View",
        ...         filter_count=1,
        ...         projection_count=3,
        ...         limit=100
        ...     ),
        ...     summary=QuerySummary(
        ...         total_matched=2,
        ...         returned_count=2,
        ...         filter_conditions="Category = 'Walls'",
        ...         requested_properties=["Name", "Volume", "Area"]
        ...     )
        ... )
    """

    elements: List[BIMElement]
    query_metadata: QueryMetadata
    summary: QuerySummary

    def __post_init__(self):
        """Validate retriever result."""
        if not isinstance(self.elements, list):
            raise ValueError("elements must be a list")

        if not isinstance(self.query_metadata, QueryMetadata):
            raise ValueError("query_metadata must be a QueryMetadata instance")

        if not isinstance(self.summary, QuerySummary):
            raise ValueError("summary must be a QuerySummary instance")

        # Validate elements count matches summary
        if len(self.elements) != self.summary.returned_count:
            raise ValueError(
                f"Number of elements ({len(self.elements)}) does not match "
                f"summary.returned_count ({self.summary.returned_count})"
            )

    @property
    def element_count(self) -> int:
        """Get number of returned elements."""
        return len(self.elements)

    @property
    def is_empty(self) -> bool:
        """Check if no elements were found."""
        return len(self.elements) == 0

    def get_elements_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[BIMElement]:
        """
        Filter elements by property value.

        Args:
            property_name: Name of property to filter by
            property_value: Value to match

        Returns:
            List of elements with matching property value
        """
        return [
            elem for elem in self.elements
            if elem.get_property(property_name) == property_value
        ]

    def get_property_values(self, property_name: str) -> List[Any]:
        """
        Get all unique values for a property across elements.

        Args:
            property_name: Name of property to collect

        Returns:
            List of unique property values (maintains order)
        """
        values = []
        seen = set()

        for elem in self.elements:
            value = elem.get_property(property_name)
            if value is not None and value not in seen:
                values.append(value)
                seen.add(value)

        return values

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "elements": [
                {
                    "element_id": elem.element_id,
                    "name": elem.name,
                    "properties": elem.properties
                }
                for elem in self.elements
            ],
            "query_metadata": {
                "model_urn": self.query_metadata.model_urn,
                "viewable_guid": self.query_metadata.viewable_guid,
                "viewable_name": self.query_metadata.viewable_name,
                "filter_count": self.query_metadata.filter_count,
                "projection_count": self.query_metadata.projection_count,
                "limit": self.query_metadata.limit
            },
            "summary": {
                "total_matched": self.summary.total_matched,
                "returned_count": self.summary.returned_count,
                "filter_conditions": self.summary.filter_conditions,
                "requested_properties": self.summary.requested_properties,
                "is_truncated": self.summary.is_truncated
            }
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"RetrieverResult(elements={len(self.elements)}, "
            f"matched={self.summary.total_matched}, "
            f"truncated={self.summary.is_truncated})"
        )


class RetrieverError(Exception):
    """Base exception for retriever errors."""
    pass


class MCPConnectionError(RetrieverError):
    """Error connecting to MCP server."""
    pass


class MCPToolError(RetrieverError):
    """Error calling MCP tool."""
    pass


class BIMQueryError(RetrieverError):
    """Error executing BIM query."""
    pass


class ResponseParseError(RetrieverError):
    """Error parsing MCP response."""
    pass
