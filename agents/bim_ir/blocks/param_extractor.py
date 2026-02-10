"""
Parameter Extractor (Block 2) for BIM-IR Agent.

Extracts structured filter and projection parameters from natural language queries.
Target accuracy: 97.8% filter_para, 98.2% proj_para (from BIM-GPT paper).
"""

import json
import logging
from typing import List, Dict, Any

from ..llm.base import LLMClient, LLMError, LLMInvalidResponseError
from ..models.intent_result import IntentResult
from ..models.parameter_result import ParameterResult, FilterParameter
from ..utils.dataset_loader import DatasetLoader
from ..utils.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ParamExtractor:
    """
    Block 2: Parameter Extraction

    Extracts filter_para (WHERE clause) and proj_para (SELECT clause) from queries.
    Requires intent classification result from Block 1 for context-aware extraction.
    """

    def __init__(self, llm_client: LLMClient, dataset_path: str):
        """
        Initialize Parameter Extractor.

        Args:
            llm_client: LLM client for parameter extraction
            dataset_path: Path to datasets directory with few-shot examples

        Raises:
            FileNotFoundError: If dataset files don't exist
        """
        self.llm_client = llm_client
        self.dataset_loader = DatasetLoader(dataset_path)

        # Load datasets once at initialization
        logger.info("Loading datasets for ParamExtractor...")
        self.examples = self.dataset_loader.load_examples()
        self.schema = self.dataset_loader.load_schema()

        logger.info(f"Initialized ParamExtractor with {len(self.examples)} examples")
        logger.info(f"Using LLM: {self.llm_client.model_name}")

    def extract(
        self,
        query: str,
        intent: IntentResult,
        max_retries: int = 3
    ) -> ParameterResult:
        """
        Extract filter and projection parameters from query.

        Args:
            query: Natural language query to extract parameters from
            intent: Intent classification result from Block 1
            max_retries: Maximum retry attempts on LLM failures

        Returns:
            ParameterResult with filter_para and proj_para

        Raises:
            ValueError: If query is empty or intent is invalid
            ParameterExtractionError: If extraction fails after retries
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not intent.is_bim:
            raise ValueError("Cannot extract parameters from non-BIM query")

        if not intent.category:
            raise ValueError("Intent must have valid category for parameter extraction")

        query = query.strip()
        logger.info(f"Extracting parameters from query: {query[:100]}...")
        logger.info(f"Intent context: category={intent.category}")

        # Build prompt with intent context
        prompt = PromptBuilder.build_parameter_extraction_prompt(
            examples=self.examples,
            schema=self.schema,
            query=query,
            intent_category=intent.category
        )

        logger.debug(f"Built prompt with {len(prompt)} characters")

        # Call LLM with retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"LLM call attempt {attempt}/{max_retries}")

                # Generate response (temperature=0.0 for deterministic output)
                response = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=300,  # Slightly larger than Block 1 for parameters
                    response_format="json"
                )

                logger.debug(f"LLM response: {response}")

                # Parse and validate response
                result = self._parse_response(response, intent.category)

                logger.info(f"Parameter extraction result: {result}")

                return result

            except LLMError as e:
                logger.warning(f"LLM error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise ParameterExtractionError(
                        f"Failed to extract parameters after {max_retries} attempts: {e}"
                    ) from e

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Parse error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise ParameterExtractionError(
                        f"Failed to parse LLM response after {max_retries} attempts: {e}"
                    ) from e

        # Should never reach here, but just in case
        raise ParameterExtractionError(f"Unexpected error after {max_retries} attempts")

    def _parse_response(self, response: str, intent_category: str) -> ParameterResult:
        """
        Parse LLM JSON response into ParameterResult.

        Args:
            response: JSON string from LLM
            intent_category: Intent category for validation

        Returns:
            ParameterResult object

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            KeyError: If required fields are missing
            ValueError: If values are invalid
        """
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response}")
            raise LLMInvalidResponseError(f"Invalid JSON: {e}") from e

        # Extract filter_para
        filter_para_data = data.get("filter_para", [])
        if not isinstance(filter_para_data, list):
            raise ValueError("filter_para must be a list")

        filter_para = []
        for item in filter_para_data:
            if not isinstance(item, dict):
                raise ValueError(f"filter_para items must be objects, got {type(item)}")

            if "name" not in item or "value" not in item:
                raise ValueError("filter_para items must have 'name' and 'value' fields")

            filter_para.append(FilterParameter(
                name=item["name"],
                value=item["value"]
            ))

        # Extract proj_para
        proj_para = data.get("proj_para", [])
        if not isinstance(proj_para, list):
            raise ValueError("proj_para must be a list")

        if len(proj_para) == 0:
            raise ValueError("proj_para cannot be empty for BIM queries")

        for item in proj_para:
            if not isinstance(item, str):
                raise ValueError(f"proj_para items must be strings, got {type(item)}")

        # Extract confidence (optional)
        confidence = data.get("confidence")

        # Create result (validation happens in ParameterResult.__post_init__)
        result = ParameterResult(
            filter_para=filter_para,
            proj_para=proj_para,
            confidence=confidence,
            raw_response=response
        )

        # Validate against intent category requirements
        try:
            result.validate_with_intent(intent_category)
        except ValueError as e:
            logger.warning(f"Intent validation warning: {e}")
            # Auto-correct if possible
            result = self._auto_correct_parameters(result, intent_category)

        return result

    def _auto_correct_parameters(
        self,
        result: ParameterResult,
        intent_category: str
    ) -> ParameterResult:
        """
        Auto-correct parameters to meet intent category requirements.

        Args:
            result: Parameter result that failed validation
            intent_category: Intent category

        Returns:
            Corrected ParameterResult
        """
        # Category-specific auto-corrections
        corrections = {
            "location": ["Location", "Name"],
            "quantity": ["Count"],
            "material": ["Material"],
            "area": ["Area"]
        }

        if intent_category in corrections:
            required_props = corrections[intent_category]
            has_required = any(prop in result.proj_para for prop in required_props)

            if not has_required:
                # Add first required property
                logger.info(f"Auto-adding {required_props[0]} to proj_para for {intent_category} query")
                corrected_proj_para = result.proj_para + [required_props[0]]

                # Create corrected result
                result = ParameterResult(
                    filter_para=result.filter_para,
                    proj_para=corrected_proj_para,
                    confidence=result.confidence,
                    raw_response=result.raw_response
                )

        return result


class ParameterExtractionError(Exception):
    """Error during parameter extraction."""
    pass
