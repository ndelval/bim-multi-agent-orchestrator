"""
Intent classification result data model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IntentResult:
    """
    Result of intent classification (Block 1).

    Attributes:
        is_bim: Whether the query is BIM-related
        intent: Intent category (same as category, redundant but matches paper)
        category: One of: location, quantity, material, detail, area (or None if not BIM)
        confidence: Optional confidence score (0.0-1.0)
        raw_response: Optional raw LLM response for debugging
    """

    is_bim: bool
    intent: Optional[str]
    category: Optional[str]
    confidence: Optional[float] = None
    raw_response: Optional[str] = None

    def __post_init__(self):
        """Validate the result after initialization."""
        # Validate is_bim and category consistency
        if not self.is_bim:
            if self.category is not None or self.intent is not None:
                raise ValueError(
                    f"Non-BIM query must have category=None and intent=None, "
                    f"got category={self.category}, intent={self.intent}"
                )
        else:
            # BIM query must have valid category
            valid_categories = {"location", "quantity", "material", "detail", "area"}
            if self.category not in valid_categories:
                raise ValueError(
                    f"BIM query must have valid category, got: {self.category}. "
                    f"Valid categories: {valid_categories}"
                )

            # Intent and category should match
            if self.intent != self.category:
                raise ValueError(
                    f"intent and category should match, got intent={self.intent}, category={self.category}"
                )

        # Validate confidence if provided
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {self.confidence}")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "is_bim": self.is_bim,
            "intent": self.intent,
            "category": self.category,
            "confidence": self.confidence
        }

    def __str__(self) -> str:
        """String representation."""
        if self.is_bim:
            conf_str = f", confidence={self.confidence:.2f}" if self.confidence else ""
            return f"IntentResult(is_bim=True, category={self.category}{conf_str})"
        else:
            return "IntentResult(is_bim=False)"
