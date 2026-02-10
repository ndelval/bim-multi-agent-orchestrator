"""
Test suite for Summarizer (Block 5).

Tests natural language response generation from BIM query results.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.bim_ir.blocks.summarizer import Summarizer, FormattingConfig
from agents.bim_ir.models.retriever_result import (
    RetrieverResult,
    BIMElement,
    QueryMetadata,
    QuerySummary
)
from agents.bim_ir.models.summarizer_result import ResponseType
from agents.bim_ir.utils.formatters import PropertyFormatter, TextFormatter, UnitSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_property_formatting():
    """Test property value formatting with units."""
    print("\n" + "="*70)
    print("Testing Property Formatting")
    print("="*70)

    formatter = PropertyFormatter()

    # Volume formatting
    assert formatter.format_property("Volume", 15.5) == "15.50 m³"
    print("✅ Volume formatted correctly: 15.50 m³")

    # Area formatting
    assert formatter.format_property("Area", 45.123) == "45.12 m²"
    print("✅ Area formatted correctly: 45.12 m²")

    # Length formatting
    assert formatter.format_property("Height", 3.456) == "3.46 m"
    print("✅ Length formatted correctly: 3.46 m")

    # Count formatting
    assert formatter.format_property("Count", 5.0) == "5"
    print("✅ Count formatted correctly: 5")

    # Text formatting
    assert formatter.format_property("Material", "Concrete") == "Concrete"
    print("✅ Text formatted correctly: Concrete")

    # Null formatting
    assert formatter.format_property("Volume", None) == "Not specified"
    print("✅ Null formatted correctly: Not specified")


def test_text_utilities():
    """Test text formatting utilities."""
    print("\n" + "="*70)
    print("Testing Text Utilities")
    print("="*70)

    # Pluralization
    assert TextFormatter.pluralize("wall", 1) == "wall"
    assert TextFormatter.pluralize("wall", 2) == "walls"
    print("✅ Pluralization works correctly")

    # List formatting
    assert TextFormatter.format_list(["A"]) == "A"
    assert TextFormatter.format_list(["A", "B"]) == "A and B"
    assert TextFormatter.format_list(["A", "B", "C"]) == "A, B, and C"
    print("✅ List formatting works correctly")


def test_empty_response():
    """Test response generation for empty results."""
    print("\n" + "="*70)
    print("Testing Empty Response Generation")
    print("="*70)

    # Create empty result
    result = RetrieverResult(
        elements=[],
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=1,
            projection_count=2,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=0,
            returned_count=0,
            filter_conditions="Category = 'NonExistent'",
            requested_properties=["Name", "Volume"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(result)

    # Validate
    assert summary.response_type == ResponseType.EMPTY
    assert summary.element_count == 0
    assert "No elements were found" in summary.response_text
    assert "Category = 'NonExistent'" in summary.response_text
    assert len(summary.key_insights) > 0

    print("✅ Empty response generated correctly")
    print(f"   Response: {summary.response_text[:60]}...")


def test_single_element_response():
    """Test response generation for single element."""
    print("\n" + "="*70)
    print("Testing Single Element Response")
    print("="*70)

    # Create single element result
    result = RetrieverResult(
        elements=[
            BIMElement(
                element_id=12345,
                name="Basic Wall: Exterior - Brick on CMU",
                properties={
                    "Category": "Walls",
                    "Level": "Level 1",
                    "Volume": 15.5,
                    "Area": 45.2,
                    "Material": "Concrete"
                }
            )
        ],
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=2,
            projection_count=3,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=1,
            returned_count=1,
            filter_conditions="Category = 'Walls' AND Level = 'Level 1'",
            requested_properties=["Name", "Volume", "Area"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(result)

    # Validate
    assert summary.response_type == ResponseType.SINGLE
    assert summary.element_count == 1
    assert "Found 1" in summary.response_text
    assert "Basic Wall: Exterior - Brick on CMU" in summary.response_text
    assert "15.50 m³" in summary.response_text  # Volume formatted
    assert "45.20 m²" in summary.response_text  # Area formatted

    print("✅ Single element response generated correctly")
    print(f"   Response preview:\n{summary.response_text[:150]}...")


def test_multiple_elements_response():
    """Test response generation for multiple elements."""
    print("\n" + "="*70)
    print("Testing Multiple Elements Response")
    print("="*70)

    # Create multiple elements result
    elements = [
        BIMElement(
            element_id=1,
            name=f"Wall {i}",
            properties={
                "Category": "Walls",
                "Level": "Level 1",
                "Volume": 10.0 + i,
                "Area": 30.0 + i
            }
        )
        for i in range(1, 6)  # 5 walls
    ]

    result = RetrieverResult(
        elements=elements,
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=1,
            projection_count=2,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=5,
            returned_count=5,
            filter_conditions="Category = 'Walls'",
            requested_properties=["Name", "Volume", "Area"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(result)

    # Validate
    assert summary.response_type == ResponseType.MULTIPLE
    assert summary.element_count == 5
    assert "Found 5 walls" in summary.response_text
    assert "Wall 1" in summary.response_text
    assert len(summary.statistics) > 0  # Should have statistics

    print("✅ Multiple elements response generated correctly")
    print(f"   Response preview:\n{summary.response_text[:200]}...")
    print(f"   Statistics: {list(summary.statistics.keys())}")


def test_truncated_response():
    """Test response generation for truncated results."""
    print("\n" + "="*70)
    print("Testing Truncated Response")
    print("="*70)

    # Create truncated result (showing 10 of 100)
    elements = [
        BIMElement(
            element_id=i,
            name=f"Wall {i}",
            properties={"Category": "Walls", "Volume": 10.0 + i}
        )
        for i in range(1, 11)  # 10 walls shown
    ]

    result = RetrieverResult(
        elements=elements,
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=1,
            projection_count=1,
            limit=10
        ),
        summary=QuerySummary(
            total_matched=100,  # 100 total
            returned_count=10,  # Only 10 returned
            filter_conditions="Category = 'Walls'",
            requested_properties=["Volume"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(result)

    # Validate
    assert summary.response_type == ResponseType.TRUNCATED
    assert summary.element_count == 10
    assert "10 of 100" in summary.response_text
    assert "Showing" in summary.response_text or "Note:" in summary.response_text

    print("✅ Truncated response generated correctly")
    print(f"   Truncation notice included")
    print(f"   Key insights: {summary.key_insights}")


def test_statistics_calculation():
    """Test property statistics calculation."""
    print("\n" + "="*70)
    print("Testing Statistics Calculation")
    print("="*70)

    # Create result with numerical properties
    elements = [
        BIMElement(
            element_id=i,
            name=f"Wall {i}",
            properties={
                "Volume": 10.0 * i,
                "Area": 30.0 * i,
                "Material": "Concrete" if i % 2 == 0 else "Brick"
            }
        )
        for i in range(1, 6)
    ]

    result = RetrieverResult(
        elements=elements,
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=0,
            projection_count=3,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=5,
            returned_count=5,
            filter_conditions="None",
            requested_properties=["Volume", "Area", "Material"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(result)

    # Validate statistics
    assert "Volume" in summary.statistics
    assert "Area" in summary.statistics
    assert "Material" in summary.statistics

    volume_stats = summary.statistics["Volume"]
    assert volume_stats.property_type == "numerical"
    assert volume_stats.total == 10 + 20 + 30 + 40 + 50  # 150
    assert volume_stats.average == 30.0
    assert volume_stats.min_value == 10.0
    assert volume_stats.max_value == 50.0
    assert volume_stats.count == 5

    material_stats = summary.statistics["Material"]
    assert material_stats.property_type == "categorical"
    assert len(material_stats.unique_values) == 2
    assert set(material_stats.unique_values) == {"Concrete", "Brick"}

    print("✅ Statistics calculated correctly")
    print(f"   Volume total: {volume_stats.total}")
    print(f"   Volume average: {volume_stats.average}")
    print(f"   Materials: {material_stats.unique_values}")


def test_formatting_configurations():
    """Test different formatting configurations."""
    print("\n" + "="*70)
    print("Testing Formatting Configurations")
    print("="*70)

    # Create test result
    elements = [
        BIMElement(
            element_id=i,
            name=f"Wall {i}",
            properties={"Volume": 10.0}
        )
        for i in range(1, 6)
    ]

    result = RetrieverResult(
        elements=elements,
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=0,
            projection_count=1,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=5,
            returned_count=5,
            filter_conditions="None",
            requested_properties=["Volume"]
        )
    )

    # Test default config
    summarizer_default = Summarizer()
    summary_default = summarizer_default.summarize(result)
    assert summary_default.element_count == 5
    print("✅ Default configuration works")

    # Test compact config
    summarizer_compact = Summarizer(config=FormattingConfig.compact())
    summary_compact = summarizer_compact.summarize(result)
    assert summary_compact.element_count == 5
    print("✅ Compact configuration works")

    # Test detailed config
    summarizer_detailed = Summarizer(config=FormattingConfig.detailed())
    summary_detailed = summarizer_detailed.summarize(result)
    assert summary_detailed.element_count == 5
    print("✅ Detailed configuration works")


def test_response_metadata():
    """Test metadata generation."""
    print("\n" + "="*70)
    print("Testing Response Metadata")
    print("="*70)

    result = RetrieverResult(
        elements=[
            BIMElement(
                element_id=1,
                name="Wall 1",
                properties={"Category": "Walls", "Volume": 10.0}
            )
        ],
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=1,
            projection_count=1,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=1,
            returned_count=1,
            filter_conditions="Category = 'Walls'",
            requested_properties=["Volume"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(
        result,
        original_query="What walls are in the model?",
        intent_category="location"
    )

    # Validate metadata
    assert "original_query" in summary.metadata
    assert summary.metadata["original_query"] == "What walls are in the model?"
    assert "intent_category" in summary.metadata
    assert summary.metadata["intent_category"] == "location"
    assert "element_count" in summary.metadata
    assert summary.metadata["element_count"] == 1

    print("✅ Metadata generated correctly")
    print(f"   Metadata keys: {list(summary.metadata.keys())}")


def test_to_dict_serialization():
    """Test result serialization to dictionary."""
    print("\n" + "="*70)
    print("Testing Result Serialization")
    print("="*70)

    result = RetrieverResult(
        elements=[
            BIMElement(
                element_id=1,
                name="Wall 1",
                properties={"Volume": 10.0}
            )
        ],
        query_metadata=QueryMetadata(
            model_urn="urn:test",
            viewable_guid="abc",
            viewable_name="3D",
            filter_count=1,
            projection_count=1,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=1,
            returned_count=1,
            filter_conditions="None",
            requested_properties=["Volume"]
        )
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(result)

    # Convert to dict
    result_dict = summary.to_dict()

    # Validate
    assert "response_text" in result_dict
    assert "response_type" in result_dict
    assert "element_count" in result_dict
    assert "metadata" in result_dict
    assert "key_insights" in result_dict
    assert "statistics" in result_dict

    assert result_dict["response_type"] == "single"
    assert result_dict["element_count"] == 1

    print("✅ Serialization works correctly")
    print(f"   Dict keys: {list(result_dict.keys())}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Summarizer Test Suite (Block 5)")
    print("="*70)

    # Run tests
    test_property_formatting()
    test_text_utilities()
    test_empty_response()
    test_single_element_response()
    test_multiple_elements_response()
    test_truncated_response()
    test_statistics_calculation()
    test_formatting_configurations()
    test_response_metadata()
    test_to_dict_serialization()

    print("\n" + "="*70)
    print("All Tests Passed! ✅")
    print("="*70)


if __name__ == "__main__":
    main()
