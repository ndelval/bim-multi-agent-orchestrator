"""
Intent Classifier (Block 1) for BIM-IR Agent.

Classifies natural language queries into BIM intent categories using few-shot learning.
Target accuracy: 99.5% (from BIM-GPT paper with 2% sampling).
"""

import json
import logging
from typing import List, Dict, Any

from ..llm.base import LLMClient, LLMError, LLMInvalidResponseError
from ..models.intent_result import IntentResult
from ..utils.dataset_loader import DatasetLoader
from ..utils.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Block 1: Intent Classification

    Classifies queries into intent categories (location, quantity, material, detail, area)
    and determines if query is BIM-related.
    """

    def __init__(self, llm_client: LLMClient, dataset_path: str):
        """
        Initialize Intent Classifier.

        Args:
            llm_client: LLM client for classification
            dataset_path: Path to datasets directory with few-shot examples

        Raises:
            FileNotFoundError: If dataset files don't exist
        """
        self.llm_client = llm_client
        self.dataset_loader = DatasetLoader(dataset_path)

        # Load datasets once at initialization
        logger.info("Loading datasets for Intent Classifier...")
        self.examples = self.dataset_loader.load_examples()
        self.schema = self.dataset_loader.load_schema()

        logger.info(f"Initialized IntentClassifier with {len(self.examples)} examples")
        logger.info(f"Using LLM: {self.llm_client.model_name}")

    def classify(self, query: str, max_retries: int = 3) -> IntentResult:
        """
        Classify a user query into intent category.

        Args:
            query: Natural language query to classify
            max_retries: Maximum retry attempts on LLM failures

        Returns:
            IntentResult with is_bim, intent, and category

        Raises:
            ValueError: If query is empty
            IntentClassificationError: If classification fails after retries
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()
        logger.info(f"Classifying query: {query[:100]}...")

        # Build prompt
        prompt = PromptBuilder.build_intent_classification_prompt(
            examples=self.examples,
            schema=self.schema,
            query=query
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
                    max_tokens=200,
                    response_format="json"
                )

                logger.debug(f"LLM response: {response}")

                # Parse and validate response
                result = self._parse_response(response)

                logger.info(f"Classification result: {result}")

                return result

            except LLMError as e:
                logger.warning(f"LLM error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise IntentClassificationError(
                        f"Failed to classify query after {max_retries} attempts: {e}"
                    ) from e

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Parse error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise IntentClassificationError(
                        f"Failed to parse LLM response after {max_retries} attempts: {e}"
                    ) from e

        # Should never reach here, but just in case
        raise IntentClassificationError(f"Unexpected error after {max_retries} attempts")

    def _parse_response(self, response: str) -> IntentResult:
        """
        Parse LLM JSON response into IntentResult.

        Args:
            response: JSON string from LLM

        Returns:
            IntentResult object

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

        # Extract fields
        is_bim = data.get("is_bim")
        intent = data.get("intent")
        category = data.get("category")
        confidence = data.get("confidence")

        # Validate required fields
        if is_bim is None:
            raise ValueError("Missing required field: is_bim")

        # Create result (validation happens in IntentResult.__post_init__)
        result = IntentResult(
            is_bim=bool(is_bim),
            intent=intent,
            category=category,
            confidence=confidence,
            raw_response=response
        )

        return result


class IntentClassificationError(Exception):
    """Error during intent classification."""
    pass
