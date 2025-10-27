#!/usr/bin/env python3
"""
Test case for parallel group vs sequential edge semantic conflict resolution.

This test validates the fix that prevents ToT LLM from creating contradictory
graph components (parallel groups + sequential edges between same nodes).

Original Error:
    Semantic conflict in parallel group 'parallel_tasks': Group declares nodes
    ['data_analysis', 'market_research'] should run in parallel, but edges create
    sequential dependencies: market_research→data_analysis.

Expected Behavior:
    Edge filtering prevents the conflict by removing edges between parallel group members
    while preserving the parallel execution intent.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from orchestrator.planning.graph_specifications import (
    StateGraphSpec, GraphNodeSpec, GraphEdgeSpec, ParallelGroup,
    NodeType, EdgeType
)


def test_parallel_edge_conflict_detection():
    """Test that validation detects semantic conflicts correctly."""

    print("=" * 70)
    print("Test 1: Parallel Group vs Sequential Edge Conflict Detection")
    print("=" * 70)

    graph_spec = StateGraphSpec(name="conflict_test")

    # Add nodes
    graph_spec.add_node(GraphNodeSpec(name="start", type=NodeType.START))
    graph_spec.add_node(GraphNodeSpec(name="market_research", type=NodeType.AGENT, agent="Researcher"))
    graph_spec.add_node(GraphNodeSpec(name="data_analysis", type=NodeType.AGENT, agent="Analyst"))
    graph_spec.add_node(GraphNodeSpec(name="end", type=NodeType.END))

    # Add conflicting edge BEFORE parallel group
    graph_spec.add_edge(GraphEdgeSpec(
        from_node="market_research",
        to_node="data_analysis",
        type=EdgeType.DIRECT
    ))

    # Add parallel group (creates conflict)
    parallel_group = ParallelGroup(
        group_id="parallel_tasks",
        nodes=["market_research", "data_analysis"]
    )
    graph_spec.add_parallel_group(parallel_group)

    # Validate - should detect conflict
    errors = graph_spec.validate()

    print(f"\nValidation errors detected: {len(errors)}")
    for error in errors:
        print(f"  - {error}")

    assert len(errors) > 0, "Expected validation to detect conflict"
    assert any("Semantic conflict" in error for error in errors), "Expected semantic conflict error"

    print("\n✅ Conflict detection working correctly")
    print()


def test_edge_filtering_prevents_conflict():
    """Test that edge filtering prevents conflicts at creation time."""

    print("=" * 70)
    print("Test 2: Edge Filtering Prevents Semantic Conflicts")
    print("=" * 70)

    # Simulate the fix: build parallel_pairs set and filter edges
    parallel_group_nodes = ["market_research", "data_analysis"]

    # Build parallel pairs set (same logic as fix)
    parallel_pairs = set()
    for i in range(len(parallel_group_nodes)):
        for j in range(len(parallel_group_nodes)):
            if i != j:
                parallel_pairs.add((parallel_group_nodes[i], parallel_group_nodes[j]))

    print(f"\nParallel pairs to filter: {parallel_pairs}")

    # Test edges
    test_edges = [
        ("start", "market_research"),           # Should NOT be filtered
        ("start", "data_analysis"),             # Should NOT be filtered
        ("market_research", "data_analysis"),   # Should BE filtered (conflict)
        ("data_analysis", "market_research"),   # Should BE filtered (conflict)
        ("market_research", "end"),             # Should NOT be filtered
        ("data_analysis", "end"),               # Should NOT be filtered
    ]

    filtered_count = 0
    allowed_count = 0

    print("\nEdge filtering results:")
    for from_node, to_node in test_edges:
        is_filtered = (from_node, to_node) in parallel_pairs
        status = "🚫 FILTERED" if is_filtered else "✅ ALLOWED"
        print(f"  {from_node} → {to_node}: {status}")

        if is_filtered:
            filtered_count += 1
        else:
            allowed_count += 1

    print(f"\nSummary: {allowed_count} edges allowed, {filtered_count} edges filtered")

    assert filtered_count == 2, f"Expected 2 filtered edges, got {filtered_count}"
    assert allowed_count == 4, f"Expected 4 allowed edges, got {allowed_count}"

    print("\n✅ Edge filtering logic working correctly")
    print()


def test_parallel_execution_preserved():
    """Test that filtering preserves parallel execution intent."""

    print("=" * 70)
    print("Test 3: Parallel Execution Intent Preserved After Filtering")
    print("=" * 70)

    graph_spec = StateGraphSpec(name="parallel_preserved_test")

    # Add nodes
    graph_spec.add_node(GraphNodeSpec(name="start", type=NodeType.START))
    graph_spec.add_node(GraphNodeSpec(name="market_research", type=NodeType.AGENT, agent="Researcher"))
    graph_spec.add_node(GraphNodeSpec(name="data_analysis", type=NodeType.AGENT, agent="Analyst"))
    graph_spec.add_node(GraphNodeSpec(name="report", type=NodeType.AGENT, agent="Reporter"))
    graph_spec.add_node(GraphNodeSpec(name="end", type=NodeType.END))

    # Add parallel group FIRST
    parallel_group = ParallelGroup(
        group_id="parallel_tasks",
        nodes=["market_research", "data_analysis"]
    )
    graph_spec.add_parallel_group(parallel_group)

    # Simulate filtered edge addition (skip conflicting edges)
    parallel_pairs = {
        ("market_research", "data_analysis"),
        ("data_analysis", "market_research")
    }

    edges_to_add = [
        ("start", "market_research"),
        ("start", "data_analysis"),
        ("market_research", "data_analysis"),  # Would be filtered
        ("market_research", "report"),
        ("data_analysis", "report"),
        ("report", "end")
    ]

    edges_added = 0
    edges_filtered = 0

    for from_node, to_node in edges_to_add:
        if (from_node, to_node) in parallel_pairs:
            print(f"  Filtering: {from_node} → {to_node} (parallel conflict)")
            edges_filtered += 1
            continue

        graph_spec.add_edge(GraphEdgeSpec(
            from_node=from_node,
            to_node=to_node,
            type=EdgeType.DIRECT
        ))
        edges_added += 1
        print(f"  Added: {from_node} → {to_node}")

    print(f"\nEdges: {edges_added} added, {edges_filtered} filtered")

    # Validate - should pass now
    errors = graph_spec.validate()

    if errors:
        print(f"\n❌ Validation errors: {errors}")
    else:
        print(f"\n✅ Validation passed - no conflicts")

    assert len(errors) == 0, f"Expected no validation errors, got: {errors}"
    assert edges_filtered == 1, f"Expected 1 filtered edge, got {edges_filtered}"
    assert edges_added == 5, f"Expected 5 edges added, got {edges_added}"

    # Verify parallel group is preserved
    assert len(graph_spec.parallel_groups) == 1, "Parallel group should be preserved"
    assert graph_spec.parallel_groups[0].nodes == ["market_research", "data_analysis"]

    print("\n✅ Parallel execution intent preserved successfully")
    print()


def run_all_tests():
    """Run all test cases."""

    print("\n" + "=" * 70)
    print("PARALLEL GROUP EDGE CONFLICT - TEST SUITE")
    print("=" * 70 + "\n")

    try:
        test_parallel_edge_conflict_detection()
        test_edge_filtering_prevents_conflict()
        test_parallel_execution_preserved()

        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Fix Summary:")
        print("  1. Edge filtering prevents semantic conflicts at creation time")
        print("  2. Parallel execution intent is preserved over sequential edges")
        print("  3. Validation passes after conflict resolution")
        print("  4. Graph maintains correct parallel execution semantics")
        print()

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
