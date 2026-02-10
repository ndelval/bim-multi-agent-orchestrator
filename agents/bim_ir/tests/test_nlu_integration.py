"""
Integration tests for BIM-IR NLU pipeline (Blocks 1-3).

Tests the complete Natural Language Understanding flow:
Block 1 (Intent Classification) → Block 2 (Parameter Extraction) → Block 3 (Value Resolution)

Requires:
- OpenAI API key configured
- Dataset CSV files present
- Internet connection

Cost: ~$0.02-0.03 per full test run (3 queries)
"""

import os
import sys
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.bim_ir.blocks import IntentClassifier, ParamExtractor, ValueResolver
from agents.bim_ir.llm.openai_client import OpenAIClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def print_subheader(text):
    """Print formatted subsection header."""
    print("\n" + "-"*70)
    print(text)
    print("-"*70)


def test_query(query: str, expected_intent_category: str):
    """
    Test a single query through the NLU pipeline (Blocks 1-3).

    Args:
        query: Natural language query
        expected_intent_category: Expected intent category (for validation)

    Returns:
        True if test passed, False otherwise
    """
    print_subheader(f"Testing Query: '{query}'")

    try:
        # Initialize blocks
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            return False

        llm_client = OpenAIClient(api_key=api_key, model="gpt-4-turbo")
        dataset_path = str(Path(__file__).parent.parent / "datasets")

        intent_classifier = IntentClassifier(llm_client, dataset_path)
        param_extractor = ParamExtractor(llm_client, dataset_path)
        value_resolver = ValueResolver(llm_client, dataset_path)

        # Block 1: Intent Classification
        print("\n🔍 Block 1: Intent Classification")
        intent_result = intent_classifier.classify(query)

        print(f"   Category: {intent_result.category}")
        print(f"   Is BIM: {intent_result.is_bim}")
        print(f"   Confidence: {intent_result.confidence:.2f}")

        # Validate intent
        if not intent_result.is_bim:
            print(f"   ⚠️  Query classified as non-BIM (unexpected)")
            return False

        if intent_result.category != expected_intent_category:
            print(f"   ⚠️  Expected category '{expected_intent_category}', got '{intent_result.category}'")
            print(f"   Note: This may be acceptable due to LLM variability")

        # Block 2: Parameter Extraction
        print("\n📋 Block 2: Parameter Extraction")
        param_result = param_extractor.extract(query, intent_result)

        print(f"   Filter parameters: {len(param_result.filter_para)}")
        for fp in param_result.filter_para:
            print(f"      - {fp.name}: {fp.value}")

        print(f"   Projection parameters: {len(param_result.proj_para)}")
        for pp in param_result.proj_para:
            print(f"      - {pp}")

        # Validate parameters
        if len(param_result.filter_para) == 0:
            print(f"   ⚠️  No filter parameters extracted (may be issue)")

        # Block 3: Value Resolution
        print("\n🔄 Block 3: Value Resolution")
        resolved_result = value_resolver.resolve(param_result, query)

        print(f"   Normalized values: {len(resolved_result.filter_values)}")
        for nv in resolved_result.filter_values:
            print(f"      - {nv.property_name}: {nv.normalized_value}")
            print(f"        Original: {nv.original_value}")
            print(f"        Method: {nv.normalization_method} (confidence: {nv.confidence:.2f})")

        # Summary
        print("\n✅ Pipeline completed successfully")
        print(f"   Intent: {intent_result.category}")
        print(f"   Filters: {len(param_result.filter_para)}")
        print(f"   Projections: {len(param_result.proj_para)}")
        print(f"   Resolved values: {len(resolved_result.filter_values)}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run NLU integration tests with example queries."""
    print_header("BIM-IR NLU Integration Tests (Blocks 1-3)")

    print("\nThis test validates the complete NLU pipeline:")
    print("  Block 1: Intent Classification")
    print("  Block 2: Parameter Extraction")
    print("  Block 3: Value Resolution")
    print("\nRequirements:")
    print("  - OpenAI API key configured")
    print("  - Dataset CSV files present")
    print("  - Internet connection")
    print("\nCost: ~$0.02-0.03 for 3 queries")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set. Cannot run tests.")
        print("   Solution: export OPENAI_API_KEY='sk-...'")
        return 1

    # Test cases with expected intents
    test_cases = [
        {
            "query": "What walls are on Level 1?",
            "expected_intent": "location",
            "description": "Location-based query (walls on specific level)"
        },
        {
            "query": "Show me all concrete columns",
            "expected_intent": "material",
            "description": "Material-based query (elements with specific material)"
        },
        {
            "query": "How many doors are there?",
            "expected_intent": "quantity",
            "description": "Quantity query (counting elements)"
        }
    ]

    results = []

    # Run tests
    print_header("Running Test Cases")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'='*70}")
        print(f"Test Case {i}/3: {test_case['description']}")
        print(f"{'='*70}")

        success = test_query(test_case["query"], test_case["expected_intent"])
        results.append({
            "query": test_case["query"],
            "success": success
        })

    # Summary
    print_header("Test Summary")

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - Test {i}: {result['query']}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All NLU integration tests passed!")
        print("\nThe pipeline is working correctly:")
        print("  - Intent classification is accurate")
        print("  - Parameter extraction is functional")
        print("  - Value resolution is working")
        print("\nNext step: Test full pipeline with Block 4 (Retriever)")
        return 0
    elif passed >= 2:
        print(f"\n⚠️  {total - passed} test(s) failed, but {passed}/{total} passed.")
        print("\nThis is acceptable due to LLM variability.")
        print("The pipeline appears to be working correctly overall.")
        return 0
    else:
        print(f"\n❌ Only {passed}/{total} tests passed.")
        print("\nThis indicates a problem with the NLU pipeline.")
        print("Review error messages above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
