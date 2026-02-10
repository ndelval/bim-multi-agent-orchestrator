"""Data models for BIM-IR agent."""

from .intent_result import IntentResult
from .parameter_result import ParameterResult, FilterParameter
from .resolved_values import ResolvedValues, NormalizedValue
from .retriever_result import (
    RetrieverResult,
    BIMElement,
    QueryMetadata,
    QuerySummary,
    RetrieverError,
    MCPConnectionError,
    MCPToolError,
    BIMQueryError,
    ResponseParseError
)
from .summarizer_result import (
    SummarizerResult,
    ResponseType,
    PropertyStatistics,
    SummarizerError,
    InvalidResultError,
    FormattingError
)
from .highlight_result import (
    HighlightResult,
    HighlightConfig,
    HighlightMode,
    ViewerCommand,
    HighlightError,
    InvalidColorError,
    InvalidElementIDError,
    ViewerCommandError
)

__all__ = [
    "IntentResult",
    "ParameterResult",
    "FilterParameter",
    "ResolvedValues",
    "NormalizedValue",
    "RetrieverResult",
    "BIMElement",
    "QueryMetadata",
    "QuerySummary",
    "RetrieverError",
    "MCPConnectionError",
    "MCPToolError",
    "BIMQueryError",
    "ResponseParseError",
    "SummarizerResult",
    "ResponseType",
    "PropertyStatistics",
    "SummarizerError",
    "InvalidResultError",
    "FormattingError",
    "HighlightResult",
    "HighlightConfig",
    "HighlightMode",
    "ViewerCommand",
    "HighlightError",
    "InvalidColorError",
    "InvalidElementIDError",
    "ViewerCommandError"
]
