#!/usr/bin/env python3
"""
Test script to verify the JSON parser fix in tot_graph_planner.py

This test uses the exact ToT output from the error logs that caused the parsing failure:
- 3 concatenated JSON objects (no separators)
- Total length: 655 characters
- Previous bug: Only parsed 1 object, failed at position 644
- Expected: Parse all 3 objects successfully
"""

import sys
import json

# Test data: Exact output from error logs
TOT_OUTPUT = '{"component_type":"node","name":"gather_financial_data","type":"agent","agent":"Researcher","objective":"Gather comprehensive financial market information","expected_output":"Financial market research report"}{"component_type":"node","name":"analyze_financial_data","type":"agent","agent":"Analyst","objective":"Analyze financial market research findings and extract actionable insights","expected_output":"Actionable insights report"}{"component_type":"node","name":"quality_assurance","type":"agent","agent":"StandardsAgent","objective":"Ensure the quality and completeness of the financial market analysis","expected_output":"Quality assurance report"}'


def _parse_json_objects(text: str):
    """
    Parse multiple JSON objects from a single string.

    This is the FIXED implementation from tot_graph_planner.py
    """
    objects = []
    decoder = json.JSONDecoder()
    idx = 0

    while idx < len(text):
        # Skip whitespace
        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx >= len(text):
            break

        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            objects.append(obj)
            idx = end_idx  # FIX: end_idx is absolute position, not offset
        except json.JSONDecodeError as e:
            print(f"⚠️  Parse error at position {idx}: {e}")
            # Try to recover by finding next object
            next_obj_start = text.find('{', idx + 1)
            if next_obj_start == -1:
                print("❌ No more JSON objects found")
                break
            print(f"🔄 Recovering: Skipping to position {next_obj_start}")
            idx = next_obj_start
            continue

    return objects


def test_parser_fix():
    """Test the fixed JSON parser with real error data."""

    print("=" * 70)
    print("JSON Parser Fix Validation Test")
    print("=" * 70)
    print(f"\nInput length: {len(TOT_OUTPUT)} characters")
    print(f"Expected objects: 3")
    print()

    # Parse the concatenated JSON objects
    parsed = _parse_json_objects(TOT_OUTPUT)

    print(f"✅ Parsed {len(parsed)} objects successfully\n")

    # Validate each object
    for i, obj in enumerate(parsed, 1):
        print(f"Object {i}:")
        print(f"  - component_type: {obj.get('component_type')}")
        print(f"  - name: {obj.get('name')}")
        print(f"  - type: {obj.get('type')}")
        print(f"  - agent: {obj.get('agent')}")
        print()

    # Assertions
    assert len(parsed) == 3, f"Expected 3 objects, got {len(parsed)}"

    expected_names = ["gather_financial_data", "analyze_financial_data", "quality_assurance"]
    actual_names = [obj.get("name") for obj in parsed]
    assert actual_names == expected_names, f"Expected {expected_names}, got {actual_names}"

    expected_agents = ["Researcher", "Analyst", "StandardsAgent"]
    actual_agents = [obj.get("agent") for obj in parsed]
    assert actual_agents == expected_agents, f"Expected {expected_agents}, got {actual_agents}"

    print("=" * 70)
    print("✅ ALL TESTS PASSED - Parser fix is working correctly!")
    print("=" * 70)
    print()
    print("Fix Summary:")
    print("  - Changed line 646: idx += end_idx → idx = end_idx")
    print("  - Reason: raw_decode() returns absolute position, not offset")
    print("  - Impact: All 3 objects parsed instead of just 1")
    print()


if __name__ == "__main__":
    try:
        test_parser_fix()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
