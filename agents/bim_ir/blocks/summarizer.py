"""
Summarizer (Block 5) for BIM-IR Agent.

Generates natural language summaries from structured BIM query results.
Final block in the NLU/NLG pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..models.retriever_result import RetrieverResult, BIMElement
from ..models.summarizer_result import (
    SummarizerResult,
    ResponseType,
    PropertyStatistics,
    InvalidResultError,
    FormattingError
)
from ..utils.formatters import (
    PropertyFormatter,
    NumberFormatter,
    TextFormatter,
    BulletFormatter,
    UnitSystem
)

logger = logging.getLogger(__name__)


@dataclass
class FormattingConfig:
    """
    Configuration for response formatting.

    Controls how responses are generated, including units, precision,
    display limits, and formatting preferences.
    """

    # Numerical precision
    decimal_places: int = 2

    # Unit system
    unit_system: UnitSystem = UnitSystem.METRIC

    # Display limits
    max_elements_detailed: int = 10  # Show full details up to this count
    max_properties_per_element: int = 5  # Limit properties shown per element

    # List formatting
    use_bullets: bool = True
    indent_spaces: int = 2

    # Statistics
    include_totals: bool = True
    include_averages: bool = False  # Only for large result sets
    min_elements_for_statistics: int = 5  # Min elements to show statistics

    # Text
    null_text: str = "Not specified"

    @classmethod
    def default(cls) -> "FormattingConfig":
        """Get default configuration."""
        return cls()

    @classmethod
    def compact(cls) -> "FormattingConfig":
        """Get compact configuration for minimal responses."""
        return cls(
            max_elements_detailed=5,
            max_properties_per_element=3,
            include_totals=False,
            use_bullets=False
        )

    @classmethod
    def detailed(cls) -> "FormattingConfig":
        """Get detailed configuration for comprehensive responses."""
        return cls(
            max_elements_detailed=20,
            max_properties_per_element=10,
            include_totals=True,
            include_averages=True,
            min_elements_for_statistics=3
        )


class Summarizer:
    """
    Block 5: Natural Language Response Generation

    Converts structured BIM query results into natural language summaries
    suitable for user presentation. Final block in the BIM-IR pipeline.

    The summarizer:
    1. Determines response type based on result characteristics
    2. Generates appropriate natural language response using templates
    3. Formats property values with units and precision
    4. Calculates aggregated statistics for numerical properties
    5. Provides key insights and metadata

    Example:
        >>> summarizer = Summarizer()
        >>> result = summarizer.summarize(
        ...     retriever_result=retriever_result,  # From Block 4
        ...     original_query="What walls are on Level 1?"
        ... )
        >>> print(result.response_text)
        Found 5 walls on Level 1:
          • Basic Wall: Exterior - Volume: 15.50 m³, Area: 45.20 m²
          • Basic Wall: Interior - Volume: 12.30 m³, Area: 38.10 m²
          ...
        Total Volume: 75.50 m³
    """

    def __init__(self, config: Optional[FormattingConfig] = None):
        """
        Initialize Summarizer.

        Args:
            config: Optional formatting configuration (uses default if None)
        """
        self.config = config or FormattingConfig.default()
        self.property_formatter = PropertyFormatter(
            unit_system=self.config.unit_system,
            decimal_places=self.config.decimal_places,
            null_text=self.config.null_text
        )

        logger.info(
            f"Initialized Summarizer with {self.config.unit_system.value} units, "
            f"{self.config.decimal_places} decimal places"
        )

    def summarize(
        self,
        retriever_result: RetrieverResult,
        original_query: Optional[str] = None,
        intent_category: Optional[str] = None
    ) -> SummarizerResult:
        """
        Generate natural language summary from retrieval results.

        Args:
            retriever_result: Results from Block 4
            original_query: Optional original user query for context
            intent_category: Optional intent category from Block 1

        Returns:
            SummarizerResult with natural language response

        Raises:
            InvalidResultError: If retriever_result is invalid
            FormattingError: If response generation fails
        """
        if not isinstance(retriever_result, RetrieverResult):
            raise InvalidResultError("retriever_result must be a RetrieverResult instance")

        logger.info(
            f"Generating summary for {retriever_result.element_count} elements "
            f"(type: {self._determine_response_type(retriever_result).value})"
        )

        try:
            # Determine response type
            response_type = self._determine_response_type(retriever_result)

            # Generate response based on type
            if response_type == ResponseType.EMPTY:
                response_text = self._generate_empty_response(retriever_result)
                key_insights = ["No matching elements found"]
                statistics = {}

            elif response_type == ResponseType.SINGLE:
                response_text = self._generate_single_element_response(retriever_result)
                key_insights = self._extract_key_insights(retriever_result, response_type)
                statistics = self._calculate_statistics(retriever_result)

            elif response_type == ResponseType.TRUNCATED:
                response_text = self._generate_truncated_response(retriever_result)
                key_insights = self._extract_key_insights(retriever_result, response_type)
                statistics = self._calculate_statistics(retriever_result)

            else:  # MULTIPLE
                response_text = self._generate_multiple_elements_response(retriever_result)
                key_insights = self._extract_key_insights(retriever_result, response_type)
                statistics = self._calculate_statistics(retriever_result)

            # Generate metadata
            metadata = self._generate_metadata(retriever_result, original_query, intent_category)

            result = SummarizerResult(
                response_text=response_text,
                response_type=response_type,
                element_count=retriever_result.element_count,
                metadata=metadata,
                key_insights=key_insights,
                statistics=statistics
            )

            logger.info(f"Summary generated successfully ({len(response_text)} characters)")
            return result

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise FormattingError(f"Failed to generate summary: {e}") from e

    def _determine_response_type(self, result: RetrieverResult) -> ResponseType:
        """Determine response type based on result characteristics."""
        if result.is_empty:
            return ResponseType.EMPTY
        elif result.element_count == 1:
            return ResponseType.SINGLE
        elif result.summary.is_truncated:
            return ResponseType.TRUNCATED
        else:
            return ResponseType.MULTIPLE

    def _generate_empty_response(self, result: RetrieverResult) -> str:
        """Generate response for empty results."""
        filter_conditions = result.summary.filter_conditions

        if filter_conditions and filter_conditions.lower() != "none":
            response = f"No elements were found matching your criteria ({filter_conditions})."
        else:
            response = "No elements were found in the model."

        # Add suggestion
        response += "\n\nTry adjusting your search criteria or checking different levels."

        return response

    def _generate_single_element_response(self, result: RetrieverResult) -> str:
        """Generate response for single element."""
        element = result.elements[0]
        requested_props = result.summary.requested_properties

        # Header
        category = element.get_property("Category", "element")
        response = f"Found 1 {category.lower()}: {element.name}\n"

        # Properties
        if requested_props:
            response += "\nProperties:\n"
            for prop in requested_props[:self.config.max_properties_per_element]:
                value = element.get_property(prop)
                formatted = self.property_formatter.format_property(prop, value)
                response += f"  • {prop}: {formatted}\n"

        return response.rstrip()

    def _generate_multiple_elements_response(self, result: RetrieverResult) -> str:
        """Generate response for multiple elements."""
        element_count = result.element_count
        requested_props = result.summary.requested_properties

        # Extract category from first element or filter conditions
        category = self._extract_category_from_result(result)

        # Header
        plural_category = TextFormatter.pluralize(category.lower(), element_count)
        response = f"Found {element_count} {plural_category}:\n\n"

        # Element list (limited to max_elements_detailed)
        elements_to_show = result.elements[:self.config.max_elements_detailed]

        for elem in elements_to_show:
            response += f"  • {elem.name}"

            # Add key properties inline
            if requested_props:
                props = []
                for prop in requested_props[:3]:  # Show max 3 inline
                    value = elem.get_property(prop)
                    if value is not None:
                        formatted = self.property_formatter.format_property(prop, value)
                        props.append(f"{prop}: {formatted}")

                if props:
                    response += " - " + ", ".join(props)

            response += "\n"

        # If more elements exist but not shown
        if element_count > self.config.max_elements_detailed:
            remaining = element_count - self.config.max_elements_detailed
            response += f"\n  ... and {remaining} more {plural_category}\n"

        # Add statistics if configured
        if self.config.include_totals and element_count >= self.config.min_elements_for_statistics:
            statistics = self._calculate_statistics(result)
            stats_text = self._format_statistics(statistics, requested_props)
            if stats_text:
                response += f"\n{stats_text}"

        return response.rstrip()

    def _generate_truncated_response(self, result: RetrieverResult) -> str:
        """Generate response for truncated results."""
        # Generate normal multiple response
        response = self._generate_multiple_elements_response(result)

        # Add truncation notice
        total_matched = result.summary.total_matched
        returned_count = result.summary.returned_count
        truncation_notice = (
            f"\n\nNote: Showing {returned_count} of {total_matched} total matching elements. "
            f"Adjust query limit to see more results."
        )

        return response + truncation_notice

    def _format_statistics(
        self,
        statistics: Dict[str, PropertyStatistics],
        requested_props: List[str]
    ) -> str:
        """Format statistics for display."""
        lines = []

        for prop in requested_props:
            if prop in statistics:
                stats = statistics[prop]

                if stats.property_type == "numerical" and stats.total is not None:
                    formatted_total = self.property_formatter.format_property(prop, stats.total)
                    lines.append(f"Total {prop}: {formatted_total}")

                    if self.config.include_averages and stats.average is not None:
                        formatted_avg = self.property_formatter.format_property(prop, stats.average)
                        lines.append(f"Average {prop}: {formatted_avg}")

        if lines:
            return "Summary:\n" + "\n".join(f"  {line}" for line in lines)

        return ""

    def _calculate_statistics(self, result: RetrieverResult) -> Dict[str, PropertyStatistics]:
        """Calculate property statistics across all elements."""
        statistics = {}
        requested_props = result.summary.requested_properties

        for prop in requested_props:
            stats = self._calculate_property_statistics(result.elements, prop)
            if stats:
                statistics[prop] = stats

        return statistics

    def _calculate_property_statistics(
        self,
        elements: List[BIMElement],
        property_name: str
    ) -> Optional[PropertyStatistics]:
        """Calculate statistics for a single property."""
        values = []
        null_count = 0

        for elem in elements:
            value = elem.get_property(property_name)
            if value is not None:
                values.append(value)
            else:
                null_count += 1

        if not values:
            return None

        # Determine property type
        first_value = values[0]
        is_numerical = isinstance(first_value, (int, float))

        if is_numerical:
            # Numerical statistics
            return PropertyStatistics(
                property_name=property_name,
                property_type="numerical",
                total=sum(values),
                average=sum(values) / len(values),
                min_value=min(values),
                max_value=max(values),
                count=len(values),
                null_count=null_count
            )
        else:
            # Categorical statistics
            unique_vals = list(set(str(v) for v in values))
            value_counts = {val: values.count(val) for val in unique_vals}

            return PropertyStatistics(
                property_name=property_name,
                property_type="categorical",
                unique_values=unique_vals,
                value_counts=value_counts,
                count=len(values),
                null_count=null_count
            )

    def _extract_key_insights(
        self,
        result: RetrieverResult,
        response_type: ResponseType
    ) -> List[str]:
        """Extract key insights from results."""
        insights = []

        if response_type == ResponseType.EMPTY:
            insights.append("No matching elements found")
        else:
            # Count insight
            category = self._extract_category_from_result(result)
            plural_category = TextFormatter.pluralize(category, result.element_count)
            insights.append(f"{result.element_count} {plural_category} found")

            # Statistics insights
            statistics = self._calculate_statistics(result)
            for prop, stats in statistics.items():
                if stats.property_type == "numerical" and stats.total is not None:
                    formatted = self.property_formatter.format_property(prop, stats.total)
                    insights.append(f"Total {prop}: {formatted}")

            # Truncation insight
            if response_type == ResponseType.TRUNCATED:
                insights.append(
                    f"Showing {result.summary.returned_count} of "
                    f"{result.summary.total_matched} total"
                )

        return insights

    def _generate_metadata(
        self,
        result: RetrieverResult,
        original_query: Optional[str],
        intent_category: Optional[str]
    ) -> Dict[str, Any]:
        """Generate metadata about the response."""
        metadata = {
            "element_count": result.element_count,
            "total_matched": result.summary.total_matched,
            "is_truncated": result.summary.is_truncated,
            "filter_conditions": result.summary.filter_conditions,
            "requested_properties": result.summary.requested_properties
        }

        if original_query:
            metadata["original_query"] = original_query

        if intent_category:
            metadata["intent_category"] = intent_category

        # Extract category if available
        category = self._extract_category_from_result(result)
        if category:
            metadata["category"] = category

        return metadata

    def _extract_category_from_result(self, result: RetrieverResult) -> str:
        """Extract category from result elements or filter conditions."""
        # Try to get from first element
        if result.elements:
            category = result.elements[0].get_property("Category")
            if category:
                return category

        # Try to extract from filter conditions
        filter_cond = result.summary.filter_conditions
        if "Category" in filter_cond:
            # Simple extraction: "Category = 'Walls'" -> "Walls"
            try:
                start = filter_cond.index("'", filter_cond.index("Category")) + 1
                end = filter_cond.index("'", start)
                return filter_cond[start:end]
            except (ValueError, IndexError):
                pass

        return "element"
