"""
Data models for viewer highlighting (visual feedback).

This module defines the data structures for generating Autodesk Viewer
highlighting instructions from BIM query results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class HighlightMode(str, Enum):
    """
    Highlighting modes for Autodesk Viewer.

    Modes:
        SELECT: Blue outline highlight (viewer.select)
        COLOR: Custom color theming (viewer.setThemingColor)
        ISOLATE: Show only selected elements, hide others (viewer.isolate)
    """
    SELECT = "select"
    COLOR = "color"
    ISOLATE = "isolate"


@dataclass
class HighlightConfig:
    """
    Configuration for viewer highlighting.

    Attributes:
        element_ids: List of BIM element database IDs (dbIds) to highlight
        model_urn: Autodesk Model URN (e.g., "urn:dXJuOmFkc2sud2lwcHJvZDpmcy...")
        color: Hex color code for COLOR mode (e.g., "#FF6600")
        mode: Highlighting mode (SELECT, COLOR, or ISOLATE)
        fit_to_view: Whether to zoom camera to highlighted elements
        clear_previous: Whether to clear previous highlights before applying new ones
        max_elements: Maximum number of elements to highlight (performance limit)
    """
    element_ids: List[int]
    model_urn: str
    color: str = "#FF6600"  # Default: Orange
    mode: HighlightMode = HighlightMode.SELECT
    fit_to_view: bool = True
    clear_previous: bool = True
    max_elements: int = 500

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.element_ids:
            raise ValueError("element_ids cannot be empty")

        if len(self.element_ids) > self.max_elements:
            raise ValueError(
                f"Too many elements ({len(self.element_ids)}). "
                f"Maximum is {self.max_elements} for performance."
            )

        if not self.model_urn or not self.model_urn.startswith("urn:"):
            raise ValueError(f"Invalid model_urn: {self.model_urn}")

        # Validate hex color format
        if self.mode == HighlightMode.COLOR:
            if not self.color.startswith("#") or len(self.color) not in [4, 7]:
                raise ValueError(
                    f"Invalid color format: {self.color}. "
                    "Expected hex format like #FF6600 or #F60"
                )

        # Validate mode
        if not isinstance(self.mode, HighlightMode):
            raise ValueError(f"Invalid mode: {self.mode}")


@dataclass
class ViewerCommand:
    """
    Individual Autodesk Viewer command.

    Represents a single viewer API call (e.g., viewer.select, viewer.setThemingColor).

    Attributes:
        command_type: Type of viewer command ("select", "isolate", "setThemingColor", etc.)
        parameters: Command parameters as dictionary
        javascript: Generated JavaScript code for this command
    """
    command_type: str
    parameters: Dict[str, Any]
    javascript: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command_type": self.command_type,
            "parameters": self.parameters,
            "javascript": self.javascript
        }


@dataclass
class HighlightResult:
    """
    Result from viewer highlighting instruction generation.

    Contains structured commands and executable JavaScript for frontend.

    Attributes:
        commands: List of viewer commands to execute
        javascript: Complete JavaScript code ready for execution
        element_count: Number of elements being highlighted
        mode: Highlighting mode used
        metadata: Additional information (color, model_urn, truncated, etc.)
    """
    commands: List[ViewerCommand]
    javascript: str
    element_count: int
    mode: HighlightMode
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for API responses
        """
        return {
            "commands": [cmd.to_dict() for cmd in self.commands],
            "javascript": self.javascript,
            "element_count": self.element_count,
            "mode": self.mode.value,
            "metadata": self.metadata
        }


# Custom exceptions for viewer highlighting

class HighlightError(Exception):
    """Base exception for viewer highlighting errors."""
    pass


class InvalidColorError(HighlightError):
    """Raised when color format is invalid."""
    pass


class InvalidElementIDError(HighlightError):
    """Raised when element IDs are invalid."""
    pass


class ViewerCommandError(HighlightError):
    """Raised when viewer command generation fails."""
    pass
