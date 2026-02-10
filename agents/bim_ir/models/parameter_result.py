"""
Parameter extraction result models for BIM-IR Agent Block 2.

Represents extracted filter and projection parameters from natural language queries.
Target accuracy: 97.8% filter_para, 98.2% proj_para (from BIM-GPT paper).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class FilterParameter:
    """
    Single filter parameter (WHERE clause component).

    Attributes:
        name: Property name (e.g., "Category", "Level", "Type")
        value: Target value (e.g., "Walls", "Level 1", "Exterior")
    """
    name: str
    value: str

    def __post_init__(self):
        """Validate filter parameter."""
        if not self.name or not self.name.strip():
            raise ValueError("Filter parameter name cannot be empty")
        if not self.value or not self.value.strip():
            raise ValueError("Filter parameter value cannot be empty")

        # Normalize to avoid case sensitivity issues
        self.name = self.name.strip()
        self.value = self.value.strip()

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "FilterParameter":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            value=data["value"]
        )


@dataclass
class ParameterResult:
    """
    Complete parameter extraction result.

    Attributes:
        filter_para: List of filter parameters (WHERE clause)
        proj_para: List of projection property names (SELECT clause)
        confidence: Optional LLM confidence score (0.0-1.0)
        raw_response: Optional raw LLM response for debugging
    """
    filter_para: List[FilterParameter]
    proj_para: List[str]
    confidence: Optional[float] = None
    raw_response: Optional[str] = None

    # Valid property names from BIM schema
    VALID_FILTER_PROPERTIES = {
        "Category", "Level", "Type", "Material", "Family", "Name"
    }

    VALID_PROJECTION_PROPERTIES = {
        "Volume", "Area", "Height", "Width", "Length", "Thickness",
        "Count", "Location", "Name", "Material", "Category", "Level",
        "Type", "Family", "Total_Floor_Area"
    }

    def __post_init__(self):
        """Validate parameter result."""
        # Validate filter_para
        if not isinstance(self.filter_para, list):
            raise ValueError("filter_para must be a list")

        for param in self.filter_para:
            if not isinstance(param, FilterParameter):
                raise ValueError(f"All filter_para items must be FilterParameter instances, got {type(param)}")

            # Validate against schema
            if param.name not in self.VALID_FILTER_PROPERTIES:
                raise ValueError(
                    f"Invalid filter property: {param.name}. "
                    f"Valid properties: {', '.join(sorted(self.VALID_FILTER_PROPERTIES))}"
                )

        # Validate proj_para
        if not isinstance(self.proj_para, list):
            raise ValueError("proj_para must be a list")

        # BIM queries must have at least one projection
        if len(self.proj_para) == 0:
            raise ValueError("proj_para cannot be empty for BIM queries")

        for prop_name in self.proj_para:
            if not isinstance(prop_name, str):
                raise ValueError(f"All proj_para items must be strings, got {type(prop_name)}")

            # Validate against schema
            if prop_name not in self.VALID_PROJECTION_PROPERTIES:
                raise ValueError(
                    f"Invalid projection property: {prop_name}. "
                    f"Valid properties: {', '.join(sorted(self.VALID_PROJECTION_PROPERTIES))}"
                )

        # Check for duplicate projections
        if len(self.proj_para) != len(set(self.proj_para)):
            raise ValueError("proj_para contains duplicate properties")

        # Validate confidence if provided
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise ValueError("confidence must be a number")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")

    def validate_with_intent(self, intent_category: str) -> None:
        """
        Validate parameters against intent category requirements.

        Args:
            intent_category: Intent category from Block 1 (location, quantity, material, detail, area)

        Raises:
            ValueError: If parameters don't align with intent requirements
        """
        # Category-specific projection requirements
        required_projections = {
            "location": {"Location", "Name"},  # At least one of these
            "quantity": {"Count"},              # Must have Count
            "material": {"Material"},           # Must have Material
            "area": {"Area", "Total_Floor_Area"}  # At least one of these
            # detail: flexible, depends on query
        }

        if intent_category in required_projections:
            required = required_projections[intent_category]
            has_required = any(proj in required for proj in self.proj_para)

            if not has_required:
                raise ValueError(
                    f"Intent category '{intent_category}' requires at least one of: "
                    f"{', '.join(required)}, but proj_para only has: {', '.join(self.proj_para)}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filter_para": [param.to_dict() for param in self.filter_para],
            "proj_para": self.proj_para,
            "confidence": self.confidence,
            "raw_response": self.raw_response
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterResult":
        """Create from dictionary representation."""
        filter_para = [
            FilterParameter.from_dict(param)
            for param in data.get("filter_para", [])
        ]

        return cls(
            filter_para=filter_para,
            proj_para=data.get("proj_para", []),
            confidence=data.get("confidence"),
            raw_response=data.get("raw_response")
        )

    def __repr__(self) -> str:
        """String representation."""
        filter_str = ", ".join(f"{p.name}={p.value}" for p in self.filter_para)
        proj_str = ", ".join(self.proj_para)
        return f"ParameterResult(filter=[{filter_str}], proj=[{proj_str}])"
