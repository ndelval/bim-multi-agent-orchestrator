"""
Value Resolver (Block 3) for BIM-IR Agent.

Normalizes parameter values using 5-step pipeline: exact → case-insensitive → synonym → partial → LLM.
Target accuracy: 88.9% normalization (from BIM-GPT paper).
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..llm.base import LLMClient, LLMError, LLMInvalidResponseError
from ..models.parameter_result import ParameterResult, FilterParameter
from ..models.resolved_values import ResolvedValues, NormalizedValue
from ..utils.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class ValueResolver:
    """
    Block 3: Value Resolution

    Normalizes parameter values from Block 2 using deterministic methods and LLM fallback.
    Implements 5-step normalization pipeline for high accuracy.
    """

    def __init__(self, llm_client: LLMClient, dataset_path: str):
        """
        Initialize Value Resolver.

        Args:
            llm_client: LLM client for fallback normalization
            dataset_path: Path to datasets directory with example_values.json

        Raises:
            FileNotFoundError: If dataset files don't exist
        """
        self.llm_client = llm_client
        self.dataset_loader = DatasetLoader(dataset_path)

        # Load example values once at initialization
        logger.info("Loading example values for ValueResolver...")
        self.example_values = self.dataset_loader.load_example_values()
        self.property_values = self.example_values["property_values"]
        self.normalization_rules = self.example_values["value_normalization_rules"]

        # Build fast lookup structures
        logger.info("Building normalization lookup tables...")
        self._synonym_lookup = self._build_synonym_lookup()
        self._value_sets = self._build_value_sets()

        # Normalization cache for repeated values
        self._normalization_cache: Dict[Tuple[str, str], NormalizedValue] = {}

        logger.info(f"Initialized ValueResolver with {len(self._synonym_lookup)} synonyms")
        logger.info(f"Using LLM: {self.llm_client.model_name}")

    def resolve(
        self,
        params: ParameterResult,
        query: str,
        max_retries: int = 3
    ) -> ResolvedValues:
        """
        Normalize all filter parameter values.

        Args:
            params: Parameter result from Block 2
            query: Original query (for LLM context)
            max_retries: Maximum retry attempts for LLM normalization

        Returns:
            ResolvedValues with normalized filter values

        Raises:
            ValueError: If params is invalid
        """
        if not params or not params.filter_para:
            logger.warning("No filter parameters to normalize")
            return ResolvedValues(filter_values=[])

        logger.info(f"Normalizing {len(params.filter_para)} filter parameter values")

        normalized_values = []

        for filter_param in params.filter_para:
            # Normalize value using 5-step pipeline
            normalized = self._normalize_value(
                property_name=filter_param.name,
                original_value=filter_param.value,
                query=query,
                max_retries=max_retries
            )

            normalized_values.append(normalized)

            logger.debug(f"Normalized: {normalized}")

        result = ResolvedValues(filter_values=normalized_values)

        logger.info(f"Normalization complete: {result}")
        logger.info(f"Statistics: {result.normalization_stats}")

        return result

    def _normalize_value(
        self,
        property_name: str,
        original_value: str,
        query: str,
        max_retries: int
    ) -> NormalizedValue:
        """
        Normalize single value using 5-step pipeline.

        Args:
            property_name: Property name (e.g., "Level")
            original_value: Value to normalize (e.g., "ground floor")
            query: Original query for context
            max_retries: Max LLM retry attempts

        Returns:
            NormalizedValue with normalization result
        """
        # Check cache first
        cache_key = (property_name, original_value.lower())
        if cache_key in self._normalization_cache:
            logger.debug(f"Cache hit for {property_name}={original_value}")
            return self._normalization_cache[cache_key]

        # Step 1: Exact match
        result = self._exact_match(property_name, original_value)
        if result:
            self._normalization_cache[cache_key] = result
            return result

        # Step 2: Case-insensitive match
        result = self._case_insensitive_match(property_name, original_value)
        if result:
            self._normalization_cache[cache_key] = result
            return result

        # Step 3: Synonym lookup
        result = self._synonym_lookup_match(property_name, original_value)
        if result:
            self._normalization_cache[cache_key] = result
            return result

        # Step 4: Partial match
        result = self._partial_match(property_name, original_value)
        if result:
            self._normalization_cache[cache_key] = result
            return result

        # Step 5: LLM prediction (fallback)
        result = self._llm_prediction(
            property_name,
            original_value,
            query,
            max_retries
        )

        self._normalization_cache[cache_key] = result
        return result

    def _exact_match(
        self,
        property_name: str,
        value: str
    ) -> Optional[NormalizedValue]:
        """Step 1: Check if value exactly matches a top value."""
        if property_name not in self._value_sets:
            return None

        if value in self._value_sets[property_name]:
            return NormalizedValue(
                property_name=property_name,
                original_value=value,
                normalized_value=value,
                normalization_method="exact",
                confidence=1.0
            )

        return None

    def _case_insensitive_match(
        self,
        property_name: str,
        value: str
    ) -> Optional[NormalizedValue]:
        """Step 2: Case-insensitive matching against top values."""
        if property_name not in self._value_sets:
            return None

        value_lower = value.lower()

        for top_value in self._value_sets[property_name]:
            if value_lower == top_value.lower():
                return NormalizedValue(
                    property_name=property_name,
                    original_value=value,
                    normalized_value=top_value,
                    normalization_method="case_insensitive",
                    confidence=0.99
                )

        return None

    def _synonym_lookup_match(
        self,
        property_name: str,
        value: str
    ) -> Optional[NormalizedValue]:
        """Step 3: Synonym lookup using normalization rules."""
        value_lower = value.lower()

        # Check if value is in synonym lookup
        if value_lower in self._synonym_lookup:
            expected_prop, normalized = self._synonym_lookup[value_lower]

            # Verify property matches (or accept if property-agnostic)
            if expected_prop == property_name or expected_prop is None:
                return NormalizedValue(
                    property_name=property_name,
                    original_value=value,
                    normalized_value=normalized,
                    normalization_method="synonym",
                    confidence=0.95
                )

        return None

    def _partial_match(
        self,
        property_name: str,
        value: str
    ) -> Optional[NormalizedValue]:
        """Step 4: Partial/substring matching for abbreviations."""
        value_lower = value.lower()

        # Check Type abbreviations (ext, int, etc.)
        if property_name == "Type":
            type_synonyms = self.normalization_rules.get("type_synonyms", {})
            for abbrev, full in type_synonyms.items():
                if abbrev in value_lower:
                    return NormalizedValue(
                        property_name=property_name,
                        original_value=value,
                        normalized_value=full,
                        normalization_method="partial",
                        confidence=0.85
                    )

        # Check Material abbreviations
        if property_name == "Material":
            material_synonyms = self.normalization_rules.get("material_synonyms", {})
            for abbrev, full in material_synonyms.items():
                if abbrev in value_lower:
                    return NormalizedValue(
                        property_name=property_name,
                        original_value=value,
                        normalized_value=full,
                        normalization_method="partial",
                        confidence=0.85
                    )

        # Check for "Level N" pattern
        if property_name == "Level":
            if "level" in value_lower and any(c.isdigit() for c in value):
                # Extract number and format as "Level N"
                number = ''.join(c for c in value if c.isdigit())
                if number:
                    normalized = f"Level {number}"
                    return NormalizedValue(
                        property_name=property_name,
                        original_value=value,
                        normalized_value=normalized,
                        normalization_method="partial",
                        confidence=0.90
                    )

        return None

    def _llm_prediction(
        self,
        property_name: str,
        value: str,
        query: str,
        max_retries: int
    ) -> NormalizedValue:
        """Step 5: LLM-based normalization (fallback)."""
        logger.info(f"Using LLM for normalization: {property_name}={value}")

        # Get top values for context
        top_values = self._get_top_values(property_name)

        # Build LLM prompt
        prompt = self._build_llm_normalization_prompt(
            property_name, value, top_values, query
        )

        # Call LLM with retries
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=150,
                    response_format="json"
                )

                # Parse response
                data = json.loads(response)
                normalized_value = data.get("normalized_value", value)
                llm_confidence = data.get("confidence", 0.75)

                return NormalizedValue(
                    property_name=property_name,
                    original_value=value,
                    normalized_value=normalized_value,
                    normalization_method="llm",
                    confidence=min(llm_confidence, 0.90)  # Cap at 0.90
                )

            except (LLMError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"LLM normalization error on attempt {attempt}: {e}")

                if attempt == max_retries:
                    # Fallback to passthrough
                    logger.warning(f"LLM normalization failed, using passthrough for {property_name}={value}")
                    return NormalizedValue(
                        property_name=property_name,
                        original_value=value,
                        normalized_value=value,  # Keep original
                        normalization_method="passthrough",
                        confidence=0.50
                    )

        # Should never reach here
        return NormalizedValue(
            property_name=property_name,
            original_value=value,
            normalized_value=value,
            normalization_method="failed",
            confidence=0.0,
            error="Max retries exceeded"
        )

    def _build_synonym_lookup(self) -> Dict[str, Tuple[str, str]]:
        """Build combined synonym lookup table from normalization rules."""
        lookup = {}

        # Level synonyms
        level_synonyms = self.normalization_rules.get("level_synonyms", {})
        for syn, canonical in level_synonyms.items():
            lookup[syn.lower()] = ("Level", canonical)

        # Type synonyms
        type_synonyms = self.normalization_rules.get("type_synonyms", {})
        for syn, canonical in type_synonyms.items():
            lookup[syn.lower()] = ("Type", canonical)

        # Material synonyms
        material_synonyms = self.normalization_rules.get("material_synonyms", {})
        for syn, canonical in material_synonyms.items():
            lookup[syn.lower()] = ("Material", canonical)

        # Category variants
        category_variants = self.normalization_rules.get("category_variants", {})
        for variant, canonical in category_variants.items():
            lookup[variant.lower()] = ("Category", canonical)

        return lookup

    def _build_value_sets(self) -> Dict[str, set]:
        """Build fast lookup sets for top values."""
        value_sets = {}

        for prop_name, prop_data in self.property_values.items():
            # Get top_values or sample_values
            top_values = prop_data.get("top_values", [])
            if not top_values:
                top_values = prop_data.get("sample_values", [])

            value_sets[prop_name] = set(top_values)

        return value_sets

    def _get_top_values(self, property_name: str) -> List[str]:
        """Get top values for property (for LLM context)."""
        if property_name in self.property_values:
            prop_data = self.property_values[property_name]
            top_values = prop_data.get("top_values", [])
            if not top_values:
                top_values = prop_data.get("sample_values", [])
            return top_values[:15]  # Limit to 15 for token efficiency

        return []

    def _build_llm_normalization_prompt(
        self,
        property_name: str,
        original_value: str,
        top_values: List[str],
        query_context: str
    ) -> str:
        """Build prompt for LLM-based value normalization."""
        prompt = f"""You are a BIM value normalization specialist.

Property: {property_name}
Original Value: "{original_value}"
Query Context: "{query_context}"

Known valid values for {property_name}:
{', '.join(top_values) if top_values else 'No predefined values available'}

Task: Normalize the original value to the closest valid value, or return the original if already appropriate.

Respond with JSON in this exact format:
{{
  "normalized_value": "exact match from valid values or appropriately formatted value",
  "confidence": 0.0-1.0
}}

Rules:
- Prefer exact matches from known valid values
- Preserve semantic meaning
- Handle synonyms and abbreviations
- If original is already valid, return it unchanged
- Confidence 1.0 for exact matches, lower for interpretations
- Only output the JSON, no additional text"""

        return prompt


class ValueResolutionError(Exception):
    """Error during value resolution."""
    pass
