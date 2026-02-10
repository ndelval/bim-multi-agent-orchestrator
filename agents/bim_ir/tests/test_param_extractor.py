"""
Test suite for ParamExtractor (Block 2).

Tests accuracy against 44 few-shot examples with dual targets:
- filter_para: ≥97.8% (≤1 error allowed)
- proj_para: ≥98.2% (≤1 error allowed)
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
from agents.bim_ir.models.intent_result import IntentResult
from agents.bim_ir.models.parameter_result import ParameterResult, FilterParameter
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
        Generate mock response for parameter extraction.

        This uses heuristics for testing. For real validation, use actual LLM API.
        """
        # Extract query and intent category from prompt
        if 'Query: "' in prompt:
            query = prompt.split('Query: "')[1].split('"')[0].lower()

        # Extract intent category
        intent_category = "detail"  # default
        if "location query" in prompt:
            intent_category = "location"
        elif "quantity query" in prompt:
            intent_category = "quantity"
        elif "material query" in prompt:
            intent_category = "material"
        elif "area query" in prompt:
            intent_category = "area"

        # Simple heuristics for parameter extraction
        filter_para = []
        proj_para = []

        # Filter parameter extraction
        if "wall" in query:
            filter_para.append({"name": "Category", "value": "Walls"})
        elif "door" in query:
            filter_para.append({"name": "Category", "value": "Doors"})
        elif "window" in query:
            filter_para.append({"name": "Category", "value": "Windows"})

        if "level 1" in query or "ground floor" in query:
            filter_para.append({"name": "Level", "value": "Level 1"})
        elif "level 2" in query or "first floor" in query:
            filter_para.append({"name": "Level", "value": "Level 2"})

        if "exterior" in query or "external" in query:
            filter_para.append({"name": "Type", "value": "Exterior"})
        elif "interior" in query or "partition" in query:
            filter_para.append({"name": "Type", "value": "Interior"})

        # Projection parameter extraction based on intent
        if intent_category == "location":
            proj_para = ["Location", "Name"]
        elif intent_category == "quantity":
            proj_para = ["Count"]
        elif intent_category == "material":
            proj_para = ["Material"]
        elif intent_category == "area":
            proj_para = ["Area"]
        else:  # detail
            if "volume" in query:
                proj_para.append("Volume")
            if "area" in query:
                proj_para.append("Area")
            if "height" in query:
                proj_para.append("Height")
            if "width" in query:
                proj_para.append("Width")

            # Default if no specific properties
            if not proj_para:
                proj_para = ["Name"]

        return json.dumps({
            "filter_para": filter_para,
            "proj_para": proj_para,
            "confidence": 0.85
        })

    @property
    def model_name(self) -> str:
        return self._model


def parameters_match(
    actual: List[FilterParameter],
    expected: List[Dict[str, str]]
) -> bool:
    """
    Check if actual filter parameters match expected (order-independent).

    Args:
        actual: List of FilterParameter objects
        expected: List of expected parameter dicts

    Returns:
        True if parameters match
    """
    if len(actual) != len(expected):
        return False

    # Convert to sets of tuples for comparison
    actual_set = {(p.name, p.value.lower()) for p in actual}
    expected_set = {(e["property"], e["value"].lower()) for e in expected}

    return actual_set == expected_set


def projections_match(
    actual: List[str],
    expected: List[Dict[str, str]]
) -> bool:
    """
    Check if actual projections match expected (order-independent).

    Args:
        actual: List of projection property names
        expected: List of expected projection dicts

    Returns:
        True if projections match
    """
    if len(actual) != len(expected):
        return False

    # Convert to sets for comparison
    actual_set = {p.lower() for p in actual}
    expected_set = {e["property"].lower() for e in expected}

    return actual_set == expected_set


def validate_param_extractor(
    intent_classifier: IntentClassifier,
    param_extractor: ParamExtractor,
    examples: List[Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate ParamExtractor against examples with dual accuracy metrics.

    Args:
        intent_classifier: Initialized IntentClassifier (Block 1)
        param_extractor: Initialized ParamExtractor (Block 2)
        examples: List of test examples with expected parameters
        verbose: Whether to print detailed results

    Returns:
        Dictionary with accuracy metrics for filter_para and proj_para
    """
    total = len(examples)
    filter_correct = 0
    proj_correct = 0
    filter_errors = []
    proj_errors = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Validating ParamExtractor on {total} examples (Block 1 + Block 2)")
        print(f"{'='*70}\n")

    for i, example in enumerate(examples, 1):
        query = example["query"]
        expected_params = example["parameters"]

        try:
            # Step 1: Classify intent (Block 1)
            intent = intent_classifier.classify(query)

            # Skip non-BIM queries for parameter extraction
            if not intent.is_bim:
                logger.info(f"Skipping non-BIM query: {query}")
                continue

            # Step 2: Extract parameters (Block 2)
            result = param_extractor.extract(query, intent)

            # Check filter_para accuracy
            filter_match = parameters_match(
                result.filter_para,
                expected_params["filter_para"]
            )

            if filter_match:
                filter_correct += 1
            else:
                filter_errors.append({
                    "example_id": i,
                    "query": query,
                    "expected": expected_params["filter_para"],
                    "actual": [{"name": p.name, "value": p.value} for p in result.filter_para]
                })

            # Check proj_para accuracy
            proj_match = projections_match(
                result.proj_para,
                expected_params["proj_para"]
            )

            if proj_match:
                proj_correct += 1
            else:
                proj_errors.append({
                    "example_id": i,
                    "query": query,
                    "expected": [p["property"] for p in expected_params["proj_para"]],
                    "actual": result.proj_para
                })

            if verbose:
                filter_status = "✅" if filter_match else "❌"
                proj_status = "✅" if proj_match else "❌"
                print(f"{i}/{total}: filter={filter_status} proj={proj_status}")
                print(f"   Query: {query[:60]}...")
                print(f"   Filter: {result.filter_para}")
                print(f"   Proj: {result.proj_para}\n")

        except Exception as e:
            logger.error(f"Exception on example {i}: {e}")
            filter_errors.append({
                "example_id": i,
                "query": query,
                "error": str(e)
            })
            proj_errors.append({
                "example_id": i,
                "query": query,
                "error": str(e)
            })

    # Calculate accuracies (excluding non-BIM queries)
    bim_count = total - 3  # Assuming 3 non-BIM queries in dataset
    filter_accuracy = filter_correct / bim_count if bim_count > 0 else 0
    proj_accuracy = proj_correct / bim_count if bim_count > 0 else 0

    results = {
        "total_examples": total,
        "bim_examples": bim_count,
        "filter_correct": filter_correct,
        "proj_correct": proj_correct,
        "filter_accuracy": filter_accuracy,
        "proj_accuracy": proj_accuracy,
        "filter_errors": filter_errors,
        "proj_errors": proj_errors,
        "filter_target_met": filter_accuracy >= 0.978,  # 97.8% target
        "proj_target_met": proj_accuracy >= 0.982       # 98.2% target
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Total examples:        {total}")
        print(f"BIM examples:          {bim_count}")
        print(f"\nFilter Parameters:")
        print(f"  Correct:             {filter_correct} ({filter_accuracy*100:.1f}%)")
        print(f"  Errors:              {len(filter_errors)}")
        print(f"  Target (97.8%):      {'✅ MET' if results['filter_target_met'] else '❌ NOT MET'}")
        print(f"\nProjection Parameters:")
        print(f"  Correct:             {proj_correct} ({proj_accuracy*100:.1f}%)")
        print(f"  Errors:              {len(proj_errors)}")
        print(f"  Target (98.2%):      {'✅ MET' if results['proj_target_met'] else '❌ NOT MET'}")
        print(f"{'='*70}\n")

    return results


def main():
    """Run validation tests."""
    print("\n" + "="*70)
    print("ParamExtractor Validation Test (Block 1 + Block 2 Integration)")
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

    # Initialize Block 1 (IntentClassifier)
    print("\nInitializing IntentClassifier (Block 1)...")
    intent_classifier = IntentClassifier(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )

    # Initialize Block 2 (ParamExtractor)
    print("Initializing ParamExtractor (Block 2)...")
    param_extractor = ParamExtractor(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )

    # Load examples for validation
    loader = DatasetLoader(str(dataset_path))
    examples = loader.load_examples()

    print(f"✅ Loaded {len(examples)} examples for validation")

    # Run validation
    results = validate_param_extractor(
        intent_classifier=intent_classifier,
        param_extractor=param_extractor,
        examples=examples,
        verbose=True
    )

    # Summary
    if results["filter_target_met"] and results["proj_target_met"]:
        print("\n🎉 SUCCESS: Both accuracy targets achieved!")
    else:
        print("\n⚠️  WARNING: Accuracy targets not met")

        if not results["filter_target_met"]:
            print(f"\n   Filter parameter errors ({len(results['filter_errors'])})")
            for error in results["filter_errors"][:5]:  # Show first 5
                print(f"   - Query: {error['query'][:50]}...")
                print(f"     Expected: {error.get('expected', 'N/A')}")
                print(f"     Got: {error.get('actual', error.get('error'))}\n")

        if not results["proj_target_met"]:
            print(f"\n   Projection parameter errors ({len(results['proj_errors'])})")
            for error in results["proj_errors"][:5]:  # Show first 5
                print(f"   - Query: {error['query'][:50]}...")
                print(f"     Expected: {error.get('expected', 'N/A')}")
                print(f"     Got: {error.get('actual', error.get('error'))}\n")

    return results


if __name__ == "__main__":
    main()
