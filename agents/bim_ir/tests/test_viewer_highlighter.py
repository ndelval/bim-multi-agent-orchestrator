"""
Test suite for ViewerHighlighter (visual feedback tool).

Tests viewer highlighting instruction generation from BIM query results.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.bim_ir.utils.viewer_highlighter import ViewerHighlighter
from agents.bim_ir.models.highlight_result import (
    HighlightConfig,
    HighlightMode,
    InvalidColorError,
    InvalidElementIDError
)
from agents.bim_ir.models.retriever_result import (
    RetrieverResult,
    BIMElement,
    QueryMetadata,
    QuerySummary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_color_conversion():
    """Test hex color to RGB normalization."""
    print("\n" + "="*70)
    print("Testing Color Conversion")
    print("="*70)

    highlighter = ViewerHighlighter()

    # Test 6-digit hex
    r, g, b = highlighter._hex_to_rgb_normalized("#FF6600")
    assert r == 1.0 and g == 0.4 and b == 0.0
    print("✅ 6-digit hex converted correctly: #FF6600 → (1.0, 0.4, 0.0)")

    # Test 3-digit hex
    r, g, b = highlighter._hex_to_rgb_normalized("#F60")
    assert r == 1.0 and g == 0.4 and b == 0.0
    print("✅ 3-digit hex converted correctly: #F60 → (1.0, 0.4, 0.0)")

    # Test without # prefix
    r, g, b = highlighter._hex_to_rgb_normalized("00FF00")
    assert r == 0.0 and g == 1.0 and b == 0.0
    print("✅ No prefix hex converted correctly: 00FF00 → (0.0, 1.0, 0.0)")

    # Test black and white
    r, g, b = highlighter._hex_to_rgb_normalized("#000000")
    assert r == 0.0 and g == 0.0 and b == 0.0
    print("✅ Black converted correctly: #000000 → (0.0, 0.0, 0.0)")

    r, g, b = highlighter._hex_to_rgb_normalized("#FFFFFF")
    assert r == 1.0 and g == 1.0 and b == 1.0
    print("✅ White converted correctly: #FFFFFF → (1.0, 1.0, 1.0)")


def test_select_mode_commands():
    """Test SELECT mode command generation."""
    print("\n" + "="*70)
    print("Testing SELECT Mode Commands")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[1234, 5678, 9012],
        model_urn="urn:adsk.wipprod:fs.file:test123",
        mode=HighlightMode.SELECT,
        fit_to_view=True,
        clear_previous=True
    )

    result = highlighter.highlight(config)

    # Validate result
    assert result.mode == HighlightMode.SELECT
    assert result.element_count == 3
    assert len(result.commands) == 3  # clearSelection, select, fitToView

    # Check command types
    command_types = [cmd.command_type for cmd in result.commands]
    assert "clearSelection" in command_types
    assert "select" in command_types
    assert "fitToView" in command_types

    # Check JavaScript contains element IDs
    assert "1234" in result.javascript
    assert "5678" in result.javascript
    assert "viewer.select" in result.javascript

    print("✅ SELECT mode commands generated correctly")
    print(f"   Commands: {command_types}")
    print(f"   JavaScript length: {len(result.javascript)} chars")


def test_color_mode_commands():
    """Test COLOR mode command generation."""
    print("\n" + "="*70)
    print("Testing COLOR Mode Commands")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[1111, 2222],
        model_urn="urn:adsk.wipprod:fs.file:test456",
        mode=HighlightMode.COLOR,
        color="#FF6600",
        fit_to_view=True,
        clear_previous=True
    )

    result = highlighter.highlight(config)

    # Validate result
    assert result.mode == HighlightMode.COLOR
    assert result.element_count == 2
    assert result.metadata["color"] == "#FF6600"

    # Check JavaScript contains color and theming
    assert "THREE.Vector4" in result.javascript
    assert "setThemingColor" in result.javascript
    assert "1.0, 0.4, 0.0" in result.javascript  # Normalized RGB for #FF6600

    # Check clearThemingColors is called
    assert "clearThemingColors" in result.javascript

    print("✅ COLOR mode commands generated correctly")
    print(f"   Color: {config.color} → RGB in JS")
    print(f"   Includes clearThemingColors: Yes")


def test_isolate_mode_commands():
    """Test ISOLATE mode command generation."""
    print("\n" + "="*70)
    print("Testing ISOLATE Mode Commands")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[3333, 4444, 5555],
        model_urn="urn:adsk.wipprod:fs.file:test789",
        mode=HighlightMode.ISOLATE,
        fit_to_view=False  # Test without fit
    )

    result = highlighter.highlight(config)

    # Validate result
    assert result.mode == HighlightMode.ISOLATE
    assert result.element_count == 3

    # Check command types
    command_types = [cmd.command_type for cmd in result.commands]
    assert "isolate" in command_types
    assert "fitToView" not in command_types  # We set fit_to_view=False

    # Check JavaScript
    assert "viewer.isolate" in result.javascript
    assert "3333" in result.javascript

    print("✅ ISOLATE mode commands generated correctly")
    print(f"   Fit to view: {config.fit_to_view}")
    print(f"   Commands: {command_types}")


def test_error_handling_invalid_color():
    """Test error handling for invalid color formats."""
    print("\n" + "="*70)
    print("Testing Error Handling - Invalid Color")
    print("="*70)

    highlighter = ViewerHighlighter()

    # Test invalid hex format
    try:
        highlighter._hex_to_rgb_normalized("ZZZZZZ")
        assert False, "Should have raised InvalidColorError"
    except InvalidColorError as e:
        print(f"✅ Invalid color rejected: {e}")

    # Test wrong length
    try:
        highlighter._hex_to_rgb_normalized("#FF")
        assert False, "Should have raised InvalidColorError"
    except InvalidColorError as e:
        print(f"✅ Wrong length rejected: {e}")


def test_error_handling_empty_elements():
    """Test error handling for empty element lists."""
    print("\n" + "="*70)
    print("Testing Error Handling - Empty Elements")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[],
        model_urn="urn:adsk.wipprod:fs.file:test999",
        mode=HighlightMode.SELECT
    )

    try:
        result = highlighter.highlight(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Empty element list rejected: {e}")


def test_error_handling_invalid_model_urn():
    """Test error handling for invalid model URN."""
    print("\n" + "="*70)
    print("Testing Error Handling - Invalid Model URN")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[1234],
        model_urn="not-a-valid-urn",
        mode=HighlightMode.SELECT
    )

    try:
        result = highlighter.highlight(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Invalid model URN rejected: {e}")


def test_integration_with_retriever_result():
    """Test integration with RetrieverResult (Block 4 output)."""
    print("\n" + "="*70)
    print("Testing Integration with RetrieverResult")
    print("="*70)

    highlighter = ViewerHighlighter()

    # Create mock retriever result
    retriever_result = RetrieverResult(
        elements=[
            BIMElement(element_id=1001, name="Wall 1", properties={"Category": "Walls"}),
            BIMElement(element_id=1002, name="Wall 2", properties={"Category": "Walls"}),
            BIMElement(element_id=1003, name="Wall 3", properties={"Category": "Walls"})
        ],
        query_metadata=QueryMetadata(
            model_urn="urn:adsk.wipprod:fs.file:integration-test",
            viewable_guid="abc123",
            viewable_name="3D View",
            filter_count=1,
            projection_count=1,
            limit=100
        ),
        summary=QuerySummary(
            total_matched=3,
            returned_count=3,
            filter_conditions="Category = 'Walls'",
            requested_properties=["Name"]
        )
    )

    # Highlight from retriever result
    result = highlighter.highlight_from_result(
        retriever_result=retriever_result,
        mode=HighlightMode.COLOR,
        color="#00FF00"
    )

    # Validate
    assert result.element_count == 3
    assert result.mode == HighlightMode.COLOR
    assert "1001" in result.javascript
    assert "1002" in result.javascript
    assert "1003" in result.javascript
    assert result.metadata["model_urn"] == "urn:adsk.wipprod:fs.file:integration-test"

    print("✅ Integration with RetrieverResult works correctly")
    print(f"   Extracted {result.element_count} element IDs")
    print(f"   Model URN: {result.metadata['model_urn']}")


def test_performance_limits():
    """Test performance limits for large element counts."""
    print("\n" + "="*70)
    print("Testing Performance Limits")
    print("="*70)

    highlighter = ViewerHighlighter()

    # Test within limit (500)
    config_ok = HighlightConfig(
        element_ids=list(range(1, 501)),  # 500 elements
        model_urn="urn:adsk.wipprod:fs.file:perf-test",
        mode=HighlightMode.SELECT,
        max_elements=500
    )

    result = highlighter.highlight(config_ok)
    assert result.element_count == 500
    print(f"✅ 500 elements accepted (within limit)")

    # Test exceeding limit
    config_exceed = HighlightConfig(
        element_ids=list(range(1, 502)),  # 501 elements
        model_urn="urn:adsk.wipprod:fs.file:perf-test",
        mode=HighlightMode.SELECT,
        max_elements=500
    )

    try:
        result = highlighter.highlight(config_exceed)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ 501 elements rejected (exceeds limit): {e}")


def test_javascript_validity():
    """Test that generated JavaScript is syntactically valid."""
    print("\n" + "="*70)
    print("Testing JavaScript Validity")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[9999],
        model_urn="urn:adsk.wipprod:fs.file:js-test",
        mode=HighlightMode.COLOR,
        color="#123456"
    )

    result = highlighter.highlight(config)

    # Check for common JavaScript syntax elements
    assert "viewer." in result.javascript
    assert ";" in result.javascript  # Statement terminator
    assert "//" in result.javascript  # Comments
    assert "function" in result.javascript or "=>" in result.javascript  # Function syntax

    # Check for balanced parentheses
    assert result.javascript.count("(") == result.javascript.count(")")
    assert result.javascript.count("[") == result.javascript.count("]")
    assert result.javascript.count("{") == result.javascript.count("}")

    print("✅ Generated JavaScript appears syntactically valid")
    print(f"   Contains viewer API calls: Yes")
    print(f"   Balanced parentheses: Yes")


def test_to_dict_serialization():
    """Test result serialization to dictionary."""
    print("\n" + "="*70)
    print("Testing Result Serialization")
    print("="*70)

    highlighter = ViewerHighlighter()

    config = HighlightConfig(
        element_ids=[7777, 8888],
        model_urn="urn:adsk.wipprod:fs.file:serialize-test",
        mode=HighlightMode.SELECT
    )

    result = highlighter.highlight(config)
    result_dict = result.to_dict()

    # Validate dictionary structure
    assert "commands" in result_dict
    assert "javascript" in result_dict
    assert "element_count" in result_dict
    assert "mode" in result_dict
    assert "metadata" in result_dict

    assert result_dict["element_count"] == 2
    assert result_dict["mode"] == "select"
    assert isinstance(result_dict["commands"], list)
    assert isinstance(result_dict["metadata"], dict)

    print("✅ Serialization works correctly")
    print(f"   Dict keys: {list(result_dict.keys())}")
    print(f"   Command count: {len(result_dict['commands'])}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ViewerHighlighter Test Suite")
    print("="*70)

    # Run tests
    test_color_conversion()
    test_select_mode_commands()
    test_color_mode_commands()
    test_isolate_mode_commands()
    test_error_handling_invalid_color()
    test_error_handling_empty_elements()
    test_error_handling_invalid_model_urn()
    test_integration_with_retriever_result()
    test_performance_limits()
    test_javascript_validity()
    test_to_dict_serialization()

    print("\n" + "="*70)
    print("All Tests Passed! ✅")
    print("="*70)


if __name__ == "__main__":
    main()
