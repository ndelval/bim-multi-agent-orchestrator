"""
Resolved values models for BIM-IR Agent Block 3.

Represents normalized parameter values after value resolution pipeline.
Target accuracy: 88.9% normalization (from BIM-GPT paper).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class NormalizedValue:
    """
    Single normalized parameter-value pair.

    Attributes:
        property_name: BIM property name (e.g., "Level", "Category")
        original_value: Value before normalization (e.g., "ground floor")
        normalized_value: Value after normalization (e.g., "Level 1")
        normalization_method: Method used ("exact"|"case_insensitive"|"synonym"|"partial"|"llm"|"passthrough"|"failed")
        confidence: Confidence score (0.0-1.0)
        error: Optional error message if normalization failed
    """
    property_name: str
    original_value: str
    normalized_value: str
    normalization_method: str
    confidence: float
    error: Optional[str] = None

    def __post_init__(self):
        """Validate normalized value."""
        # Validate property name
        if not self.property_name or not self.property_name.strip():
            raise ValueError("property_name cannot be empty")

        # Validate normalization method
        valid_methods = {
            "exact", "case_insensitive", "synonym", "partial",
            "llm", "passthrough", "failed"
        }
        if self.normalization_method not in valid_methods:
            raise ValueError(
                f"Invalid normalization_method: {self.normalization_method}. "
                f"Must be one of: {', '.join(valid_methods)}"
            )

        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        # Normalize whitespace
        self.property_name = self.property_name.strip()
        self.original_value = self.original_value.strip()
        self.normalized_value = self.normalized_value.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "property_name": self.property_name,
            "original_value": self.original_value,
            "normalized_value": self.normalized_value,
            "normalization_method": self.normalization_method,
            "confidence": self.confidence,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedValue":
        """Create from dictionary representation."""
        return cls(
            property_name=data["property_name"],
            original_value=data["original_value"],
            normalized_value=data["normalized_value"],
            normalization_method=data["normalization_method"],
            confidence=data["confidence"],
            error=data.get("error")
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.normalized_value != self.original_value:
            return f"{self.property_name}: '{self.original_value}' → '{self.normalized_value}' ({self.normalization_method}, {self.confidence:.2f})"
        else:
            return f"{self.property_name}: '{self.normalized_value}' ({self.normalization_method}, {self.confidence:.2f})"


@dataclass
class ResolvedValues:
    """
    Complete value resolution result.

    Attributes:
        filter_values: List of normalized filter parameter values
        all_normalized: True if all values successfully normalized (confidence > 0.0)
        normalization_stats: Count of normalizations by method
        low_confidence_values: Values with confidence < 0.5 (may need review)
    """
    filter_values: List[NormalizedValue]
    all_normalized: bool = True
    normalization_stats: Dict[str, int] = field(default_factory=dict)
    low_confidence_values: List[NormalizedValue] = field(default_factory=list)

    def __post_init__(self):
        """Calculate statistics and flags."""
        if not isinstance(self.filter_values, list):
            raise ValueError("filter_values must be a list")

        # Check if all normalized
        self.all_normalized = all(
            v.confidence > 0.0 for v in self.filter_values
        )

        # Calculate normalization method statistics
        self.normalization_stats = {}
        for value in self.filter_values:
            method = value.normalization_method
            self.normalization_stats[method] = self.normalization_stats.get(method, 0) + 1

        # Identify low confidence values
        self.low_confidence_values = [
            v for v in self.filter_values if v.confidence < 0.5
        ]

    def get_normalized_dict(self) -> Dict[str, str]:
        """
        Get dictionary mapping property names to normalized values.

        Returns:
            Dict with property_name → normalized_value
        """
        return {
            v.property_name: v.normalized_value
            for v in self.filter_values
        }

    def get_high_confidence_values(self, threshold: float = 0.8) -> List[NormalizedValue]:
        """
        Get values with confidence >= threshold.

        Args:
            threshold: Minimum confidence (default 0.8)

        Returns:
            List of high-confidence NormalizedValue objects
        """
        return [v for v in self.filter_values if v.confidence >= threshold]

    def get_method_percentage(self, method: str) -> float:
        """
        Get percentage of values normalized by specific method.

        Args:
            method: Normalization method name

        Returns:
            Percentage (0.0-100.0)
        """
        if not self.filter_values:
            return 0.0

        count = self.normalization_stats.get(method, 0)
        return (count / len(self.filter_values)) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filter_values": [v.to_dict() for v in self.filter_values],
            "all_normalized": self.all_normalized,
            "normalization_stats": self.normalization_stats,
            "low_confidence_count": len(self.low_confidence_values)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedValues":
        """Create from dictionary representation."""
        filter_values = [
            NormalizedValue.from_dict(v)
            for v in data.get("filter_values", [])
        ]

        return cls(filter_values=filter_values)

    def __repr__(self) -> str:
        """String representation."""
        stats_str = ", ".join(f"{k}={v}" for k, v in self.normalization_stats.items())
        return (
            f"ResolvedValues("
            f"values={len(self.filter_values)}, "
            f"all_normalized={self.all_normalized}, "
            f"stats=[{stats_str}])"
        )
