"""
Route classification system for orchestrator graph routing.

This module provides centralized routing logic with keyword-based classification
and JSON extraction capabilities for LLM-based routing decisions.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RoutingKeywords:
    """Configuration for route-specific keywords."""

    quick: List[str]
    research: List[str]
    analysis: List[str]
    planning: List[str]
    standards: List[str]

    @classmethod
    def default(cls) -> "RoutingKeywords":
        """Create default routing keywords configuration."""
        return cls(
            quick=["quick", "simple", "hello", "hola", "hi"],
            research=[
                "research",
                "search",
                "find",
                "investiga",
                "busca",
                "encuentra",
            ],
            analysis=[
                "analyze",
                "analysis",
                "deep dive",
                "in detail",
                "detailed",
                "analiza",
                "analisis",
                "análisis",
                "detallado",
            ],
            planning=[
                "plan",
                "planning",
                "strategy",
                "planear",
                "estrategia",
                "complex task",
            ],
            standards=[
                "standard",
                "compliance",
                "norm",
                "norma",
                "cumplimiento",
            ],
        )


@dataclass
class RouteDecision:
    """Result of route classification."""

    route: str
    confidence: Optional[float] = None
    reason: str = ""
    source: str = ""  # "keyword", "llm_json", "llm_fallback", "default"


class RouteClassifier:
    """
    Centralized route classification system.

    Provides both keyword-based and LLM-response-based routing classification,
    with fallback mechanisms to ensure robust routing decisions.
    """

    def __init__(
        self,
        keywords: Optional[RoutingKeywords] = None,
        default_route: str = "analysis",
    ):
        """
        Initialize route classifier.

        Args:
            keywords: Custom routing keywords configuration
            default_route: Default route when no match is found
        """
        self.keywords = keywords or RoutingKeywords.default()
        self.default_route = default_route
        self.allowed_routes = {
            "quick",
            "research",
            "analysis",
            "planning",
            "standards",
        }

    def classify_by_keywords(self, prompt: str) -> RouteDecision:
        """
        Classify route based on keyword matching.

        Uses word boundary matching for single-word keywords to avoid
        false positives from substring matches.

        Args:
            prompt: User input prompt to classify

        Returns:
            RouteDecision with matched route and metadata
        """
        prompt_lower = prompt.lower()

        # Check each route type in priority order
        # Priority: planning > analysis > research > standards > quick
        # This prevents "quickly plan my project" from matching quick instead of planning
        if self._match_keywords(prompt_lower, self.keywords.planning):
            logger.info("[RouteClassifier] Matched 'planning' keywords")
            return RouteDecision(
                route="planning",
                confidence=0.8,
                reason="Matched planning keywords",
                source="keyword",
            )

        if self._match_keywords(prompt_lower, self.keywords.analysis):
            logger.info("[RouteClassifier] Matched 'analysis' keywords")
            return RouteDecision(
                route="analysis",
                confidence=0.8,
                reason="Matched analysis keywords",
                source="keyword",
            )

        if self._match_keywords(prompt_lower, self.keywords.research):
            logger.info("[RouteClassifier] Matched 'research' keywords")
            return RouteDecision(
                route="research",
                confidence=0.8,
                reason="Matched research keywords",
                source="keyword",
            )

        if self._match_keywords(prompt_lower, self.keywords.standards):
            logger.info("[RouteClassifier] Matched 'standards' keywords")
            return RouteDecision(
                route="standards",
                confidence=0.8,
                reason="Matched standards keywords",
                source="keyword",
            )

        if self._match_keywords(prompt_lower, self.keywords.quick):
            logger.info("[RouteClassifier] Matched 'quick' keywords")
            return RouteDecision(
                route="quick",
                confidence=0.8,
                reason="Matched quick response keywords",
                source="keyword",
            )

        # No match found, use default
        logger.info(
            "[RouteClassifier] No keyword matched, defaulting to %s",
            self.default_route,
        )
        return RouteDecision(
            route=self.default_route,
            confidence=0.5,
            reason="No keyword match, using default route",
            source="default",
        )

    def _match_keywords(self, text: str, keywords: List[str]) -> bool:
        """
        Match keywords with word boundary support.

        For single-word keywords (no spaces), uses word boundaries to avoid
        substring matches. Multi-word keywords use simple substring matching.

        Args:
            text: Text to search in (should be lowercase)
            keywords: List of keywords to match (should be lowercase)

        Returns:
            True if any keyword matches, False otherwise
        """
        for kw in keywords:
            if " " in kw:
                # Multi-word keyword, use substring match
                if kw in text:
                    return True
            else:
                # Single-word keyword, use word boundary
                # Check if keyword appears as a complete word
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return True
        return False

    def classify_from_llm_response(self, llm_response: str) -> RouteDecision:
        """
        Classify route from LLM response with multiple fallback strategies.

        Attempts to:
        1. Parse clean JSON response
        2. Extract JSON from markdown code blocks or text
        3. Fallback to keyword extraction from response text
        4. Use default route if all fail

        Args:
            llm_response: Raw response from LLM router agent

        Returns:
            RouteDecision with extracted route and metadata
        """
        logger.debug("[RouteClassifier] Parsing LLM response: %s", llm_response)

        # Try parsing as clean JSON first
        parsed_route = self._parse_json(llm_response)
        if parsed_route:
            return parsed_route

        # Try extracting JSON from code blocks or embedded text
        extracted_route = self._extract_json_from_text(llm_response)
        if extracted_route:
            return extracted_route

        # Fallback to keyword extraction from LLM response
        fallback_route = self._fallback_keyword_extraction(llm_response)
        if fallback_route:
            return fallback_route

        # Ultimate fallback
        logger.warning(
            "[RouteClassifier] All parsing strategies failed, using default route"
        )
        return RouteDecision(
            route=self.default_route,
            confidence=0.3,
            reason="All parsing strategies failed",
            source="default",
        )

    def _parse_json(self, text: str) -> Optional[RouteDecision]:
        """
        Attempt to parse text as clean JSON.

        Args:
            text: Text that might contain JSON

        Returns:
            RouteDecision if valid JSON found, None otherwise
        """
        try:
            parsed = json.loads(text)
            return self._validate_route_dict(parsed, "llm_json")
        except json.JSONDecodeError:
            return None

    def _extract_json_from_text(self, text: str) -> Optional[RouteDecision]:
        """
        Extract JSON from markdown code blocks or embedded text.

        Args:
            text: Text potentially containing JSON fragments

        Returns:
            RouteDecision if valid JSON found, None otherwise
        """
        # Try to find JSON object pattern
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return None

        try:
            parsed = json.loads(json_match.group(0))
            return self._validate_route_dict(parsed, "llm_json")
        except json.JSONDecodeError:
            logger.debug("[RouteClassifier] Failed to parse extracted JSON")
            return None

    def _validate_route_dict(
        self, parsed: Dict, source: str
    ) -> Optional[RouteDecision]:
        """
        Validate parsed JSON dictionary and create RouteDecision.

        Uses Pydantic RouterOutput for type coercion and constraint
        enforcement (BP-STRUCT-04).  Falls back gracefully if validation
        fails.

        Args:
            parsed: Parsed JSON dictionary
            source: Source identifier for logging

        Returns:
            RouteDecision if valid route found, None otherwise
        """
        if not isinstance(parsed, dict):
            return None

        route = parsed.get("route")
        if not route:
            return None

        # Pydantic validation for type coercion & constraints (BP-STRUCT-04)
        try:
            from orchestrator.schemas.outputs import RouterOutput

            validated = RouterOutput.model_validate(parsed)
        except Exception:
            logger.debug("[RouteClassifier] Pydantic validation failed for %s", parsed)
            return None

        if validated.route not in self.allowed_routes:
            logger.warning(
                "[RouteClassifier] Invalid route '%s' not in allowed routes",
                validated.route,
            )
            return None

        logger.info("[RouteClassifier] Successfully parsed route: %s", validated.route)
        return RouteDecision(
            route=validated.route,
            confidence=validated.confidence,
            reason=validated.reasoning or "LLM JSON decision",
            source=source,
        )

    def _fallback_keyword_extraction(self, text: str) -> Optional[RouteDecision]:
        """
        Fallback keyword extraction from LLM response text.

        Args:
            text: LLM response text

        Returns:
            RouteDecision if route keyword found, None otherwise
        """
        lower_text = text.lower()

        # Check for route keywords in priority order
        route_checks = [
            ("planning", self.keywords.planning),
            ("quick", self.keywords.quick),
            ("research", self.keywords.research),
            ("standards", self.keywords.standards),
            ("analysis", self.keywords.analysis),
        ]

        for route_name, keywords in route_checks:
            # Check both explicit route name and keywords
            if route_name in lower_text or any(kw in lower_text for kw in keywords):
                logger.info(
                    "[RouteClassifier] Fallback extraction found route: %s", route_name
                )
                return RouteDecision(
                    route=route_name,
                    confidence=0.6,
                    reason=f"Keyword extraction from LLM response",
                    source="llm_fallback",
                )

        return None

    def validate_route(self, route: str) -> str:
        """
        Validate that route is in allowed routes, return default if not.

        Args:
            route: Route to validate

        Returns:
            Validated route (or default if invalid)
        """
        route = route.strip().lower()
        if route in self.allowed_routes:
            return route

        logger.warning(
            "[RouteClassifier] Invalid route '%s', using default '%s'",
            route,
            self.default_route,
        )
        return self.default_route
