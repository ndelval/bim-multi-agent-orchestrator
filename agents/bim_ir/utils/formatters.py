"""
Formatting utilities for BIM property values.

Provides formatters for numerical values, property names, and text.
"""

from typing import Any, Optional
from enum import Enum


class UnitSystem(str, Enum):
    """Supported unit systems."""
    METRIC = "metric"
    IMPERIAL = "imperial"


class PropertyFormatter:
    """
    Format BIM property values with appropriate units and precision.

    Handles numerical properties (Volume, Area, Length) and text properties
    with proper formatting, units, and null value handling.

    Example:
        >>> formatter = PropertyFormatter()
        >>> formatter.format_property("Volume", 15.5)
        '15.50 m³'
        >>> formatter.format_property("Material", "Concrete")
        'Concrete'
        >>> formatter.format_property("Volume", None)
        'Not specified'
    """

    # Property types requiring special formatting
    VOLUME_PROPERTIES = {"Volume", "Total_Volume"}
    AREA_PROPERTIES = {"Area", "Total_Floor_Area", "Net_Area", "Gross_Area"}
    LENGTH_PROPERTIES = {"Height", "Width", "Length", "Thickness", "Depth"}
    COUNT_PROPERTIES = {"Count", "Quantity"}

    def __init__(
        self,
        unit_system: UnitSystem = UnitSystem.METRIC,
        decimal_places: int = 2,
        null_text: str = "Not specified"
    ):
        """
        Initialize property formatter.

        Args:
            unit_system: Unit system to use (metric or imperial)
            decimal_places: Decimal places for numerical values
            null_text: Text to display for null/missing values
        """
        self.unit_system = unit_system
        self.decimal_places = decimal_places
        self.null_text = null_text

    def format_property(self, property_name: str, value: Any) -> str:
        """
        Format property value with appropriate units and precision.

        Args:
            property_name: Name of the property
            value: Property value to format

        Returns:
            Formatted string representation
        """
        # Handle null values
        if value is None:
            return self.null_text

        # Handle different property types
        if property_name in self.VOLUME_PROPERTIES:
            return self._format_volume(value)
        elif property_name in self.AREA_PROPERTIES:
            return self._format_area(value)
        elif property_name in self.LENGTH_PROPERTIES:
            return self._format_length(value)
        elif property_name in self.COUNT_PROPERTIES:
            return self._format_count(value)
        elif isinstance(value, (int, float)):
            return self._format_number(value)
        else:
            return str(value)

    def _format_volume(self, value: float) -> str:
        """Format volume with appropriate units."""
        if self.unit_system == UnitSystem.METRIC:
            return f"{value:.{self.decimal_places}f} m³"
        else:
            # Convert m³ to ft³ (1 m³ = 35.3147 ft³)
            ft3_value = value * 35.3147
            return f"{ft3_value:.{self.decimal_places}f} ft³"

    def _format_area(self, value: float) -> str:
        """Format area with appropriate units."""
        if self.unit_system == UnitSystem.METRIC:
            return f"{value:.{self.decimal_places}f} m²"
        else:
            # Convert m² to ft² (1 m² = 10.7639 ft²)
            ft2_value = value * 10.7639
            return f"{ft2_value:.{self.decimal_places}f} ft²"

    def _format_length(self, value: float) -> str:
        """Format length/distance with appropriate units."""
        if self.unit_system == UnitSystem.METRIC:
            return f"{value:.{self.decimal_places}f} m"
        else:
            # Convert m to ft (1 m = 3.28084 ft)
            ft_value = value * 3.28084
            return f"{ft_value:.{self.decimal_places}f} ft"

    def _format_count(self, value: Any) -> str:
        """Format count (integer, no units)."""
        try:
            return str(int(value))
        except (ValueError, TypeError):
            return str(value)

    def _format_number(self, value: float) -> str:
        """Format generic number with decimal places."""
        return f"{value:.{self.decimal_places}f}"

    def format_property_with_name(self, property_name: str, value: Any) -> str:
        """
        Format property with name label.

        Args:
            property_name: Name of property
            value: Property value

        Returns:
            Formatted string like "Volume: 15.50 m³"
        """
        formatted_value = self.format_property(property_name, value)
        return f"{property_name}: {formatted_value}"


class NumberFormatter:
    """
    Format numerical values with locale-aware formatting.

    Handles large numbers, decimals, and scientific notation.
    """

    @staticmethod
    def format_number(
        value: float,
        decimal_places: int = 2,
        use_thousands_separator: bool = True
    ) -> str:
        """
        Format number with thousands separator and decimal places.

        Args:
            value: Number to format
            decimal_places: Decimal places to show
            use_thousands_separator: Whether to use comma separator

        Returns:
            Formatted number string

        Example:
            >>> NumberFormatter.format_number(1234.567, 2)
            '1,234.57'
        """
        if use_thousands_separator:
            return f"{value:,.{decimal_places}f}"
        else:
            return f"{value:.{decimal_places}f}"

    @staticmethod
    def format_large_number(value: float) -> str:
        """
        Format very large numbers with K/M/B suffixes.

        Args:
            value: Number to format

        Returns:
            Formatted string like "1.5K", "2.3M"
        """
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"
        else:
            return f"{value:.1f}"


class TextFormatter:
    """
    Format text for natural language responses.

    Handles pluralization, capitalization, and list formatting.
    """

    @staticmethod
    def pluralize(word: str, count: int) -> str:
        """
        Pluralize word based on count.

        Simple English pluralization rules.

        Args:
            word: Singular word
            count: Count to determine plurality

        Returns:
            Singular or plural form

        Example:
            >>> TextFormatter.pluralize("wall", 1)
            'wall'
            >>> TextFormatter.pluralize("wall", 2)
            'walls'
        """
        if count == 1:
            return word

        # Simple pluralization rules
        if word.endswith('s'):
            return word + 'es'
        elif word.endswith('y'):
            return word[:-1] + 'ies'
        else:
            return word + 's'

    @staticmethod
    def format_list(items: list, conjunction: str = "and") -> str:
        """
        Format list of items with commas and conjunction.

        Args:
            items: List of items to format
            conjunction: Conjunction word ("and", "or")

        Returns:
            Formatted list string

        Example:
            >>> TextFormatter.format_list(["A", "B", "C"])
            'A, B, and C'
            >>> TextFormatter.format_list(["A", "B"])
            'A and B'
        """
        if not items:
            return ""
        elif len(items) == 1:
            return str(items[0])
        elif len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        else:
            return f"{', '.join(str(i) for i in items[:-1])}, {conjunction} {items[-1]}"

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length with suffix.

        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to append if truncated

        Returns:
            Truncated text

        Example:
            >>> TextFormatter.truncate_text("Very long text", 10)
            'Very lo...'
        """
        if len(text) <= max_length:
            return text

        truncate_at = max_length - len(suffix)
        return text[:truncate_at] + suffix

    @staticmethod
    def capitalize_first(text: str) -> str:
        """
        Capitalize first letter of text.

        Args:
            text: Text to capitalize

        Returns:
            Text with first letter capitalized
        """
        if not text:
            return text
        return text[0].upper() + text[1:]


class BulletFormatter:
    """Format lists with bullets and indentation."""

    @staticmethod
    def format_bulleted_list(
        items: list,
        indent: int = 2,
        bullet: str = "•"
    ) -> str:
        """
        Format items as bulleted list.

        Args:
            items: List of items to format
            indent: Spaces before bullet
            bullet: Bullet character

        Returns:
            Formatted bulleted list

        Example:
            >>> BulletFormatter.format_bulleted_list(["Item 1", "Item 2"])
            '  • Item 1\\n  • Item 2'
        """
        indent_str = " " * indent
        return "\n".join(f"{indent_str}{bullet} {item}" for item in items)

    @staticmethod
    def format_nested_list(
        items: list,
        properties: dict,
        indent: int = 2
    ) -> str:
        """
        Format nested list with items and their properties.

        Args:
            items: List of item names
            properties: Dict mapping item names to property dicts
            indent: Base indentation

        Returns:
            Formatted nested list

        Example:
            >>> items = ["Wall 1"]
            >>> props = {"Wall 1": {"Volume": "15.50 m³", "Area": "45.20 m²"}}
            >>> BulletFormatter.format_nested_list(items, props)
            '  • Wall 1\\n      - Volume: 15.50 m³\\n      - Area: 45.20 m²'
        """
        lines = []
        indent_str = " " * indent
        prop_indent = " " * (indent + 4)

        for item in items:
            lines.append(f"{indent_str}• {item}")

            if item in properties:
                for prop_name, prop_value in properties[item].items():
                    lines.append(f"{prop_indent}- {prop_name}: {prop_value}")

        return "\n".join(lines)
