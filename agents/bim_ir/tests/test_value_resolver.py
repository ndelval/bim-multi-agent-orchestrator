"""
Test suite for ValueResolver (Block 3).

Tests normalization accuracy against target: ≥88.9% (from BIM-GPT paper).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.bim_ir.llm.base import LLMClient
from agents.bim_ir.blocks.intent_classifier import IntentClassifier
from agents.bim_ir.blocks.param_extractor import ParamExtractor
from agents.bim_ir.blocks.value_resolver import ValueResolver
from agents.bim_ir.models.parameter_result import ParameterResult, FilterParameter
from agents.bim_ir.models.resolved_values import ResolvedValues, NormalizedValue
from agents.bim_ir.utils.dataset_loader import DatasetLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""

    def __init__(self):
        self._model = "mock-gpt-4"

    def generate(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 300, response_format: str = None) -> str:
        """
        Generate mock response for value normalization.

        Simple heuristic-based normalization for testing.
        """
        # Extract property and value from prompt
        if 'Property: ' in prompt and 'Original Value: ' in prompt:
            lines = prompt.split('\n')
            property_name = None
            original_value = None

            for line in lines:
                if line.startswith('Property: '):
                    property_name = line.replace('Property: ', '').strip()
                elif line.startswith('Original Value: '):
                    original_value = line.replace('Original Value: ', '').strip().strip('"')

            if property_name and original_value:
                # Simple normalization heuristics
                normalized = original_value
                confidence = 0.75

                # Property-specific normalization
                if property_name == "Level":
                    if "ground" in original_value.lower() or "first" in original_value.lower():
                        normalized = "Level 1"
                        confidence = 0.85
                    elif "second" in original_value.lower():
                        normalized = "Level 2"
                        confidence = 0.85

                elif property_name == "Category":
                    # Ensure plural
                    if not original_value.endswith('s'):
                        normalized = original_value + 's'
                    confidence = 0.80

                elif property_name == "Type":
                    if "ext" in original_value.lower():
                        normalized = "Exterior"
                        confidence = 0.85
                    elif "int" in original_value.lower():
                        normalized = "Interior"
                        confidence = 0.85

                return json.dumps({
                    "normalized_value": normalized,
                    "confidence": confidence
                })

        # Fallback
        return json.dumps({
            "normalized_value": "Unknown",
            "confidence": 0.50
        })

    @property
    def model_name(self) -> str:
        return self._model


def test_normalization_methods():
    """Test individual normalization methods."""
    print("\n" + "="*70)
    print("Testing Individual Normalization Methods")
    print("="*70)

    # Get dataset path
    current_dir = Path(__file__).parent
    dataset_path = current_dir.parent / "datasets"

    llm_client = MockLLMClient()
    resolver = ValueResolver(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )

    # Test data
    test_cases = [
        # (property, value, expected_normalized, expected_method, min_confidence)
        ("Level", "Level 1", "Level 1", "exact", 1.0),
        ("Level", "level 1", "Level 1", "case_insensitive", 0.99),
        ("Level", "ground floor", "Level 1", "synonym", 0.95),
        ("Level", "lvl 2", "Level 2", "partial", 0.85),
        ("Category", "wall", "Walls", "synonym", 0.95),
        ("Category", "Walls", "Walls", "exact", 1.0),
        ("Type", "ext", "Exterior", "partial", 0.85),
        ("Type", "Exterior", "Exterior", "exact", 1.0),
        ("Material", "Concrete", "Concrete", "exact", 1.0),
        ("Material", "cement", "Concrete", "synonym", 0.95),
    ]

    passed = 0
    failed = 0

    for prop_name, value, expected_norm, expected_method, min_conf in test_cases:
        # Create test parameter
        param_result = ParameterResult(
            filter_para=[FilterParameter(prop_name, value)],
            proj_para=["Name"]
        )

        # Resolve
        resolved = resolver.resolve(param_result, f"test query with {value}")

        if resolved.filter_values:
            norm_value = resolved.filter_values[0]

            # Check normalization
            if (norm_value.normalized_value == expected_norm and
                norm_value.normalization_method == expected_method and
                norm_value.confidence >= min_conf):
                print(f"✅ {prop_name}='{value}' → '{expected_norm}' ({expected_method})")
                passed += 1
            else:
                print(f"❌ {prop_name}='{value}'")
                print(f"   Expected: '{expected_norm}' ({expected_method}, ≥{min_conf})")
                print(f"   Got: '{norm_value.normalized_value}' ({norm_value.normalization_method}, {norm_value.confidence})")
                failed += 1
        else:
            print(f"❌ {prop_name}='{value}' - No normalization result")
            failed += 1

    print(f"\nMethod Tests: {passed}/{passed+failed} passed")
    return passed, failed


def validate_value_resolver(
    intent_classifier: IntentClassifier,
    param_extractor: ParamExtractor,
    value_resolver: ValueResolver,
    examples: List[Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate ValueResolver against examples (Block 1 + 2 + 3 integration).

    Args:
        intent_classifier: Initialized IntentClassifier (Block 1)
        param_extractor: Initialized ParamExtractor (Block 2)
        value_resolver: Initialized ValueResolver (Block 3)
        examples: List of test examples
        verbose: Whether to print detailed results

    Returns:
        Dictionary with normalization accuracy metrics
    """
    total_normalizations = 0
    correct_normalizations = 0
    errors = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Validating ValueResolver (Block 1 + 2 + 3 Integration)")
        print(f"{'='*70}\n")

    for i, example in enumerate(examples, 1):
        query = example["query"]

        try:
            # Block 1: Classify intent
            intent = intent_classifier.classify(query)

            # Skip non-BIM queries
            if not intent.is_bim:
                continue

            # Block 2: Extract parameters
            params = param_extractor.extract(query, intent)

            # Block 3: Resolve values
            resolved = value_resolver.resolve(params, query)

            # Validate normalizations
            for norm_value in resolved.filter_values:
                total_normalizations += 1

                # Find expected value from example
                expected_filter = next(
                    (f for f in example["parameters"]["filter_para"]
                     if f["property"] == norm_value.property_name),
                    None
                )

                if expected_filter:
                    expected_value = expected_filter["value"]

                    # Check if normalized value matches expected (case-insensitive)
                    if norm_value.normalized_value.lower() == expected_value.lower():
                        correct_normalizations += 1

                        if verbose and i <= 10:  # Show first 10
                            print(f"✅ {i}: {norm_value.property_name}='{norm_value.original_value}' → '{norm_value.normalized_value}'")
                    else:
                        errors.append({
                            "example_id": i,
                            "query": query,
                            "property": norm_value.property_name,
                            "expected": expected_value,
                            "actual": norm_value.normalized_value,
                            "method": norm_value.normalization_method,
                            "confidence": norm_value.confidence
                        })

                        if verbose:
                            print(f"❌ {i}: {norm_value.property_name}")
                            print(f"   Expected: '{expected_value}'")
                            print(f"   Got: '{norm_value.normalized_value}' ({norm_value.normalization_method})")

        except Exception as e:
            logger.error(f"Exception on example {i}: {e}")

    accuracy = correct_normalizations / total_normalizations if total_normalizations > 0 else 0

    results = {
        "total_normalizations": total_normalizations,
        "correct": correct_normalizations,
        "errors": len(errors),
        "accuracy": accuracy,
        "target_met": accuracy >= 0.889,  # 88.9% target
        "error_details": errors
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Total normalizations:  {total_normalizations}")
        print(f"Correct:               {correct_normalizations} ({accuracy*100:.1f}%)")
        print(f"Errors:                {len(errors)}")
        print(f"Target (88.9%):        {'✅ MET' if results['target_met'] else '❌ NOT MET'}")
        print(f"{'='*70}\n")

    return results


def main():
    """Run validation tests."""
    print("\n" + "="*70)
    print("ValueResolver Validation Test (Block 1 + 2 + 3 Integration)")
    print("="*70)

    # Get dataset path
    current_dir = Path(__file__).parent
    dataset_path = current_dir.parent / "datasets"

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return

    print(f"✅ Dataset path: {dataset_path}")

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print("✅ OpenAI API key found - using real LLM")
        from agents.bim_ir.llm.openai_client import OpenAIClient
        llm_client = OpenAIClient(api_key=api_key, model="gpt-4-turbo")
    else:
        print("ℹ️  No OpenAI API key - using MockLLMClient")
        print("   Set OPENAI_API_KEY environment variable for real validation")
        llm_client = MockLLMClient()

    # Initialize all blocks
    print("\nInitializing BIM-IR pipeline...")
    intent_classifier = IntentClassifier(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )
    param_extractor = ParamExtractor(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )
    value_resolver = ValueResolver(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )

    # Load examples
    loader = DatasetLoader(str(dataset_path))
    examples = loader.load_examples()
    print(f"✅ Loaded {len(examples)} examples for validation")

    # Test individual methods
    method_passed, method_failed = test_normalization_methods()

    # Run integration validation
    results = validate_value_resolver(
        intent_classifier=intent_classifier,
        param_extractor=param_extractor,
        value_resolver=value_resolver,
        examples=examples,
        verbose=True
    )

    # Summary
    if results["target_met"]:
        print("\n🎉 SUCCESS: Target accuracy (88.9%) achieved!")
    else:
        print(f"\n⚠️  WARNING: Target accuracy not met ({results['accuracy']*100:.1f}% < 88.9%)")

        if results["error_details"]:
            print(f"\n   Normalization errors ({len(results['error_details'][:5])}):")
            for error in results["error_details"][:5]:
                print(f"   - Query: {error['query'][:50]}...")
                print(f"     Property: {error['property']}")
                print(f"     Expected: '{error['expected']}'")
                print(f"     Got: '{error['actual']}' ({error['method']})\n")

    return results


if __name__ == "__main__":
    main()
