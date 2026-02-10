"""
Dataset loader utility for BIM-IR agent.

Loads few-shot examples, property schema, and value normalization data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Utility class for loading BIM-IR datasets."""

    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to datasets directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        logger.info(f"Initialized DatasetLoader with path: {dataset_path}")

    def load_examples(self) -> List[Dict[str, Any]]:
        """
        Load few-shot examples from JSON file.

        Returns:
            List of example dictionaries

        Raises:
            FileNotFoundError: If examples file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        examples_file = self.dataset_path / "few_shot_examples.json"

        if not examples_file.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_file}")

        logger.debug(f"Loading examples from: {examples_file}")

        with open(examples_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = data.get("examples", [])
        logger.info(f"Loaded {len(examples)} few-shot examples")

        return examples

    def load_schema(self) -> Dict[str, Any]:
        """
        Load BIM property schema from JSON file.

        Returns:
            Schema dictionary with property definitions

        Raises:
            FileNotFoundError: If schema file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        schema_file = self.dataset_path / "bim_property_schema.json"

        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        logger.debug(f"Loading schema from: {schema_file}")

        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)

        logger.info("Loaded BIM property schema")

        return schema

    def load_example_values(self) -> Dict[str, Any]:
        """
        Load example values for value normalization.

        Returns:
            Dictionary with example values and normalization rules

        Raises:
            FileNotFoundError: If values file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        values_file = self.dataset_path / "example_values.json"

        if not values_file.exists():
            raise FileNotFoundError(f"Values file not found: {values_file}")

        logger.debug(f"Loading example values from: {values_file}")

        with open(values_file, "r", encoding="utf-8") as f:
            values = json.load(f)

        logger.info("Loaded example values and normalization rules")

        return values

    def get_examples_by_category(
        self,
        examples: List[Dict[str, Any]],
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Filter examples by intent category.

        Args:
            examples: List of all examples
            category: Category to filter by (or "non_bim" for is_bim=false)

        Returns:
            Filtered list of examples
        """
        if category == "non_bim":
            filtered = [ex for ex in examples if not ex["intent"]["is_bim"]]
        else:
            filtered = [
                ex for ex in examples
                if ex["intent"].get("category") == category
            ]

        logger.debug(f"Filtered {len(filtered)} examples for category: {category}")

        return filtered
