"""
Summarizer Result Models (Block 5 Output).

Data models for natural language response generation results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ResponseType(str, Enum):
    """Types of responses based on retrieval results."""
    EMPTY = "empty"  # No elements found
    SINGLE = "single"  # Exactly 1 element found
    MULTIPLE = "multiple"  # 2+ elements found
    TRUNCATED = "truncated"  # Results limited by query limit


@dataclass
class PropertyStatistics:
    """
    Statistics for a specific property across elements.

    Provides aggregated values for numerical properties and distributions
    for categorical properties.

    Attributes:
        property_name: Name of the property
        property_type: Type classification (numerical/categorical/text)
        total: Sum of values (numerical properties only)
        average: Mean value (numerical properties only)
        min_value: Minimum value (numerical properties only)
        max_value: Maximum value (numerical properties only)
        unique_values: List of unique values (categorical properties)
        value_counts: Distribution of values (categorical properties)
        count: Number of elements with this property
        null_count: Number of elements without this property
    """

    property_name: str
    property_type: str  # "numerical" | "categorical" | "text"

    # Numerical statistics
    total: Optional[float] = None
    average: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Categorical statistics
    unique_values: Optional[List[str]] = None
    value_counts: Optional[Dict[str, int]] = None

    # General
    count: int = 0  # Elements with this property
    null_count: int = 0  # Elements without this property

    @property
    def coverage_ratio(self) -> float:
        """Calculate ratio of elements with this property (0.0-1.0)."""
        total_elements = self.count + self.null_count
        if total_elements == 0:
            return 0.0
        return self.count / total_elements

    @property
    def is_complete(self) -> bool:
        """Check if all elements have this property."""
        return self.null_count == 0

    def __repr__(self) -> str:
        """Human-readable representation."""
        if self.property_type == "numerical":
            return (
                f"PropertyStatistics('{self.property_name}': "
                f"total={self.total}, avg={self.average}, count={self.count})"
            )
        else:
            unique_count = len(self.unique_values) if self.unique_values else 0
            return (
                f"PropertyStatistics('{self.property_name}': "
                f"{unique_count} unique values, count={self.count})"
            )


@dataclass
class SummarizerResult:
    """
    Result from natural language response generation (Block 5 output).

    Contains the generated natural language response along with metadata
    and structured statistics for programmatic use.

    Attributes:
        response_text: Main natural language response for user
        response_type: Classification of response type
        element_count: Number of elements described in response
        metadata: Additional context about the response
        key_insights: List of key findings (bullet points)
        statistics: Property-level statistics and aggregations

    Example:
        >>> result = SummarizerResult(
        ...     response_text="Found 5 walls on Level 1:\\n  • Wall 1: 15.50 m³\\n  ...",
        ...     response_type=ResponseType.MULTIPLE,
        ...     element_count=5,
        ...     metadata={"query_category": "Walls", "level": "Level 1"},
        ...     key_insights=["5 walls found", "Total volume: 75.50 m³"],
        ...     statistics={"Volume": PropertyStatistics(...)}
        ...  )
    """

    response_text: str
    response_type: ResponseType
    element_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    key_insights: List[str] = field(default_factory=list)
    statistics: Dict[str, PropertyStatistics] = field(default_factory=dict)

    def __post_init__(self):
        """Validate summarizer result."""
        if not self.response_text:
            raise ValueError("response_text cannot be empty")

        if not isinstance(self.response_type, ResponseType):
            # Try to convert string to enum
            if isinstance(self.response_type, str):
                self.response_type = ResponseType(self.response_type)
            else:
                raise ValueError("response_type must be a ResponseType enum")

        if self.element_count < 0:
            raise ValueError("element_count must be non-negative")

    @property
    def has_results(self) -> bool:
        """Check if response contains any elements."""
        return self.element_count > 0

    @property
    def is_empty_response(self) -> bool:
        """Check if this is an empty result response."""
        return self.response_type == ResponseType.EMPTY

    @property
    def is_truncated_response(self) -> bool:
        """Check if results were truncated."""
        return self.response_type == ResponseType.TRUNCATED

    def get_statistic(self, property_name: str) -> Optional[PropertyStatistics]:
        """
        Get statistics for a specific property.

        Args:
            property_name: Name of property to get statistics for

        Returns:
            PropertyStatistics or None if not found
        """
        return self.statistics.get(property_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "response_text": self.response_text,
            "response_type": self.response_type.value,
            "element_count": self.element_count,
            "metadata": self.metadata,
            "key_insights": self.key_insights,
            "statistics": {
                prop_name: {
                    "property_name": stats.property_name,
                    "property_type": stats.property_type,
                    "total": stats.total,
                    "average": stats.average,
                    "min_value": stats.min_value,
                    "max_value": stats.max_value,
                    "unique_values": stats.unique_values,
                    "value_counts": stats.value_counts,
                    "count": stats.count,
                    "null_count": stats.null_count,
                    "coverage_ratio": stats.coverage_ratio,
                    "is_complete": stats.is_complete
                }
                for prop_name, stats in self.statistics.items()
            }
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        preview = self.response_text[:50] + "..." if len(self.response_text) > 50 else self.response_text
        return (
            f"SummarizerResult(type={self.response_type.value}, "
            f"elements={self.element_count}, text='{preview}')"
        )


class SummarizerError(Exception):
    """Base exception for summarizer errors."""
    pass


class InvalidResultError(SummarizerError):
    """Error when input result is invalid."""
    pass


class FormattingError(SummarizerError):
    """Error during response formatting."""
    pass
