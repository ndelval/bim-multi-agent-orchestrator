"""
Test suite for IntentClassifier (Block 1).

Tests accuracy against 44 few-shot examples with target: ≥99.5% (≤1 error allowed).
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
from agents.bim_ir.models.intent_result import IntentResult
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
                 max_tokens: int = 200, response_format: str = None) -> str:
        """
        Generate mock response based on query pattern matching.

        This is a simple heuristic-based mock for testing the classifier structure.
        For real validation, use actual LLM API.
        """
        # Extract query from prompt
        if 'Query: "' in prompt:
            query = prompt.split('Query: "')[1].split('"')[0].lower()

            # Simple heuristics
            if any(word in query for word in ["weather", "architect", "deadline", "date", "time"]):
                return json.dumps({"is_bim": False, "intent": None, "category": None})

            elif any(word in query for word in ["where", "show", "find", "locate"]):
                return json.dumps({"is_bim": True, "intent": "location", "category": "location"})

            elif any(word in query for word in ["how many", "count", "number"]):
                return json.dumps({"is_bim": True, "intent": "quantity", "category": "quantity"})

            elif any(word in query for word in ["material", "made of"]):
                return json.dumps({"is_bim": True, "intent": "material", "category": "material"})

            elif any(word in query for word in ["area", "floor area", "total area"]):
                return json.dumps({"is_bim": True, "intent": "area", "category": "area"})

            else:
                # Default to detail for other BIM queries
                return json.dumps({"is_bim": True, "intent": "detail", "category": "detail"})

        # Fallback
        return json.dumps({"is_bim": False, "intent": None, "category": None})

    @property
    def model_name(self) -> str:
        return self._model


def validate_intent_classifier(
    classifier: IntentClassifier,
    examples: List[Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate IntentClassifier against examples.

    Args:
        classifier: Initialized IntentClassifier
        examples: List of test examples with expected outputs
        verbose: Whether to print detailed results

    Returns:
        Dictionary with accuracy metrics and errors
    """
    total = len(examples)
    correct = 0
    errors = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating IntentClassifier on {total} examples")
        print(f"{'='*60}\n")

    for i, example in enumerate(examples, 1):
        query = example["query"]
        expected = example["intent"]

        try:
            # Classify
            result = classifier.classify(query)

            # Check correctness
            is_correct = (
                result.is_bim == expected["is_bim"] and
                result.category == expected.get("category")
            )

            if is_correct:
                correct += 1
                if verbose:
                    print(f"✅ {i}/{total}: CORRECT")
                    print(f"   Query: {query[:60]}...")
                    print(f"   Result: is_bim={result.is_bim}, category={result.category}\n")
            else:
                errors.append({
                    "example_id": i,
                    "query": query,
                    "expected": expected,
                    "actual": {
                        "is_bim": result.is_bim,
                        "category": result.category
                    }
                })

                if verbose:
                    print(f"❌ {i}/{total}: ERROR")
                    print(f"   Query: {query}")
                    print(f"   Expected: is_bim={expected['is_bim']}, category={expected.get('category')}")
                    print(f"   Got:      is_bim={result.is_bim}, category={result.category}\n")

        except Exception as e:
            logger.error(f"Exception on example {i}: {e}")
            errors.append({
                "example_id": i,
                "query": query,
                "expected": expected,
                "error": str(e)
            })

    accuracy = correct / total
    error_rate = (total - correct) / total

    results = {
        "total_examples": total,
        "correct": correct,
        "errors_count": len(errors),
        "accuracy": accuracy,
        "error_rate": error_rate,
        "errors": errors,
        "target_met": accuracy >= 0.995  # 99.5% target
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Total examples:  {total}")
        print(f"Correct:         {correct} ({accuracy*100:.1f}%)")
        print(f"Errors:          {len(errors)} ({error_rate*100:.1f}%)")
        print(f"Target (99.5%):  {'✅ MET' if results['target_met'] else '❌ NOT MET'}")
        print(f"{'='*60}\n")

    return results


def main():
    """Run validation tests."""
    print("\n" + "="*60)
    print("IntentClassifier Validation Test")
    print("="*60)

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

    # Initialize classifier
    print("\nInitializing IntentClassifier...")
    classifier = IntentClassifier(
        llm_client=llm_client,
        dataset_path=str(dataset_path)
    )

    # Load examples for validation
    loader = DatasetLoader(str(dataset_path))
    examples = loader.load_examples()

    print(f"✅ Loaded {len(examples)} examples for validation")

    # Run validation
    results = validate_intent_classifier(
        classifier=classifier,
        examples=examples,
        verbose=True
    )

    # Summary
    if results["target_met"]:
        print("\n🎉 SUCCESS: Target accuracy (99.5%) achieved!")
    else:
        print(f"\n⚠️  WARNING: Target accuracy not met ({results['accuracy']*100:.1f}% < 99.5%)")
        print(f"   Errors: {results['errors_count']}")
        print("\n   Error details:")
        for error in results["errors"]:
            print(f"   - Query: {error['query'][:60]}...")
            print(f"     Expected: {error['expected']}")
            print(f"     Got: {error.get('actual', error.get('error'))}\n")

    return results


if __name__ == "__main__":
    main()
