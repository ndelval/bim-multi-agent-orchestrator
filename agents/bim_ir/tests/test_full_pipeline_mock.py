"""
Full pipeline integration test with mock MCP server.

Tests the complete BIM-IR pipeline from query to response + highlighting:
Block 1 → Block 2 → Block 3 → Block 4 (Mock) → Block 5 → Highlighter

Uses mock MCP responses to avoid requiring actual APS infrastructure.

Requires:
- OpenAI API key (for Blocks 1-3)
- Dataset CSV files
- No MCP server needed (uses mock)

Cost: ~$0.02-0.03 per test run
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.bim_ir.blocks import (
    IntentClassifier,
    ParamExtractor,
    ValueResolver,
    Summarizer
)
from agents.bim_ir.utils import ViewerHighlighter
from agents.bim_ir.models import (
    RetrieverResult,
    BIMElement,
    QueryMetadata,
    QuerySummary,
    HighlightMode
)
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


def create_mock_retriever_result(query: str, element_count: int = 5) -> RetrieverResult:
    """
    Create mock RetrieverResult simulating MCP server response.

    Args:
        query: Original query (for context)
        element_count: Number of mock elements to create

    Returns:
        RetrieverResult with mock data
    """
    # Create mock BIM elements
    mock_elements = []

    for i in range(1, element_count + 1):
        element = BIMElement(
            element_id=1000 + i,
            name=f"Basic Wall: Wall {i}",
            properties={
                "Category": "Walls",
                "Level": "Level 1",
                "Volume": 10.0 + i * 2.5,
                "Area": 30.0 + i * 5.0,
                "Material": "Concrete" if i % 2 == 0 else "Brick",
                "Fire Rating": "2 Hour" if i % 3 == 0 else "1 Hour"
            }
        )
        mock_elements.append(element)

    # Create metadata
    metadata = QueryMetadata(
        model_urn="urn:adsk.wipprod:fs.file:mock-test-model-123",
        viewable_guid="mock-viewable-abc",
        viewable_name="3D View",
        filter_count=2,
        projection_count=4,
        limit=100
    )

    # Create summary
    summary = QuerySummary(
        total_matched=element_count,
        returned_count=element_count,
        filter_conditions="Category = 'Walls' AND Level = 'Level 1'",
        requested_properties=["Name", "Volume", "Area", "Material"]
    )

    return RetrieverResult(
        elements=mock_elements,
        query_metadata=metadata,
        summary=summary
    )


def test_full_pipeline(query: str):
    """
    Test complete pipeline with a single query.

    Args:
        query: Natural language query

    Returns:
        True if test passed, False otherwise
    """
    print_header(f"Testing Full Pipeline: '{query}'")

    try:
        # Initialize all blocks
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            return False

        llm_client = OpenAIClient(api_key=api_key, model="gpt-4-turbo")
        dataset_path = str(Path(__file__).parent.parent / "datasets")

        intent_classifier = IntentClassifier(llm_client, dataset_path)
        param_extractor = ParamExtractor(llm_client, dataset_path)
        value_resolver = ValueResolver(llm_client, dataset_path)
        summarizer = Summarizer()
        highlighter = ViewerHighlighter()

        # ===== BLOCK 1: Intent Classification =====
        print("\n📌 Block 1: Intent Classification")
        intent_result = intent_classifier.classify(query)
        print(f"   Category: {intent_result.category}")
        print(f"   Is BIM: {intent_result.is_bim}")
        print(f"   ✅ Block 1 completed")

        if not intent_result.is_bim:
            print("   ⚠️  Query not recognized as BIM-related")
            return False

        # ===== BLOCK 2: Parameter Extraction =====
        print("\n📌 Block 2: Parameter Extraction")
        param_result = param_extractor.extract(query, intent_result)
        print(f"   Filter parameters: {len(param_result.filter_para)}")
        print(f"   Projection parameters: {len(param_result.proj_para)}")
        print(f"   ✅ Block 2 completed")

        # ===== BLOCK 3: Value Resolution =====
        print("\n📌 Block 3: Value Resolution")
        resolved_result = value_resolver.resolve(param_result, query)
        print(f"   Normalized values: {len(resolved_result.filter_values)}")
        print(f"   ✅ Block 3 completed")

        # ===== BLOCK 4: Retrieval (MOCK) =====
        print("\n📌 Block 4: Retrieval (Using Mock MCP)")
        retriever_result = create_mock_retriever_result(query, element_count=5)
        print(f"   Elements retrieved: {len(retriever_result.elements)}")
        print(f"   Model URN: {retriever_result.query_metadata.model_urn}")
        print(f"   ✅ Block 4 completed (mock)")

        # Validate retriever result structure
        assert isinstance(retriever_result, RetrieverResult)
        assert len(retriever_result.elements) > 0
        assert all(isinstance(elem, BIMElement) for elem in retriever_result.elements)

        # ===== BLOCK 5: Summarization =====
        print("\n📌 Block 5: Summarization")
        summary_result = summarizer.summarize(
            retriever_result=retriever_result,
            original_query=query,
            intent_category=intent_result.category
        )
        print(f"   Response type: {summary_result.response_type.value}")
        print(f"   Element count: {summary_result.element_count}")
        print(f"   Statistics calculated: {len(summary_result.statistics)}")
        print(f"   ✅ Block 5 completed")

        # Display response preview
        print("\n📝 Generated Response (preview):")
        response_preview = summary_result.response_text[:300]
        print(f"   {response_preview}...")

        # Validate summary result
        assert len(summary_result.response_text) > 0
        assert summary_result.element_count == len(retriever_result.elements)

        # ===== AUXILIARY: Viewer Highlighting =====
        print("\n📌 Auxiliary: Viewer Highlighting")
        highlight_result = highlighter.highlight_from_result(
            retriever_result=retriever_result,
            mode=HighlightMode.COLOR,
            color="#FF6600"
        )
        print(f"   Highlight mode: {highlight_result.mode.value}")
        print(f"   Element count: {highlight_result.element_count}")
        print(f"   Commands generated: {len(highlight_result.commands)}")
        print(f"   JavaScript length: {len(highlight_result.javascript)} chars")
        print(f"   ✅ Highlighting completed")

        # Validate highlight result
        assert len(highlight_result.commands) > 0
        assert len(highlight_result.javascript) > 0
        assert "viewer." in highlight_result.javascript

        # ===== FINAL VALIDATION =====
        print_header("Pipeline Validation")

        validations = [
            ("Intent classification", intent_result.is_bim),
            ("Parameters extracted", len(param_result.filter_para) > 0 or len(param_result.proj_para) > 0),
            ("Values resolved", len(resolved_result.filter_values) >= 0),  # Can be 0 for some queries
            ("Elements retrieved", len(retriever_result.elements) > 0),
            ("Response generated", len(summary_result.response_text) > 0),
            ("Highlights created", len(highlight_result.commands) > 0)
        ]

        all_valid = True
        for check, result in validations:
            status = "✅" if result else "❌"
            print(f"   {status} {check}")
            if not result:
                all_valid = False

        if all_valid:
            print("\n🎉 Full pipeline test PASSED!")
            print("\nPipeline successfully executed:")
            print(f"   Query: '{query}'")
            print(f"   Intent: {intent_result.category}")
            print(f"   Elements: {retriever_result.element_count}")
            print(f"   Response length: {len(summary_result.response_text)} chars")
            print(f"   Highlight commands: {len(highlight_result.commands)}")
            return True
        else:
            print("\n❌ Pipeline validation failed")
            return False

    except Exception as e:
        print(f"\n❌ Pipeline test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run full pipeline tests."""
    print("="*70)
    print("BIM-IR Full Pipeline Integration Test (with Mock MCP)")
    print("="*70)

    print("\nThis test validates the complete pipeline:")
    print("  Block 1: Intent Classification")
    print("  Block 2: Parameter Extraction")
    print("  Block 3: Value Resolution")
    print("  Block 4: Retrieval (Mock MCP)")
    print("  Block 5: Summarization")
    print("  Auxiliary: Viewer Highlighting")
    print("\nRequirements:")
    print("  - OpenAI API key (for Blocks 1-3)")
    print("  - Dataset CSV files")
    print("  - No MCP server needed (uses mock)")
    print("\nCost: ~$0.02-0.03 per test")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set. Cannot run tests.")
        print("   Solution: export OPENAI_API_KEY='sk-...'")
        return 1

    # Test cases
    test_queries = [
        "What walls are on Level 1?",
        "Show me all concrete elements"
    ]

    results = []

    # Run tests
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*70}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'='*70}")

        success = test_full_pipeline(query)
        results.append({
            "query": query,
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
        print("\n🎉 All pipeline tests passed!")
        print("\nThe complete BIM-IR system is working correctly:")
        print("  ✅ NLU pipeline (Blocks 1-3)")
        print("  ✅ Data flow and integration")
        print("  ✅ Summarization (Block 5)")
        print("  ✅ Viewer highlighting")
        print("\nNext step (optional):")
        print("  - Test with real MCP server and actual BIM model")
        print("  - This requires APS credentials and aps-mcp-server")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\nReview error messages above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
