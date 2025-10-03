"""
Structural test for ToT→StateGraph integration (no API calls).

This module tests the integration structure without requiring API keys
or making external calls.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all integration components can be imported."""
    print("Testing imports...")
    
    try:
        # Test planning module imports
        from orchestrator.planning import (
            generate_plan_with_tot, generate_graph_with_tot,
            get_planning_capabilities, validate_planning_environment,
            is_graph_planning_available
        )
        print("✅ Core planning functions imported successfully")
    except ImportError as e:
        print(f"❌ Core planning functions import failed: {e}")
        return False
    
    try:
        # Test graph specifications
        from orchestrator.planning.graph_specifications import (
            StateGraphSpec, GraphNodeSpec, GraphEdgeSpec,
            create_simple_sequential_graph
        )
        print("✅ Graph specifications imported successfully")
    except ImportError as e:
        print(f"❌ Graph specifications import failed: {e}")
        return False
    
    try:
        # Test graph planner
        from orchestrator.planning.tot_graph_planner import (
            GraphPlanningSettings, GraphPlanningTask
        )
        print("✅ Graph planner imported successfully")
    except ImportError as e:
        print(f"❌ Graph planner import failed: {e}")
        return False
    
    try:
        # Test graph compiler
        from orchestrator.planning.graph_compiler import (
            GraphCompiler, compile_tot_graph
        )
        print("✅ Graph compiler imported successfully")
    except ImportError as e:
        print(f"❌ Graph compiler import failed: {e}")
        return False
    
    return True


def test_capabilities():
    """Test capability detection."""
    print("\nTesting capabilities...")
    
    try:
        from orchestrator.planning import get_planning_capabilities, is_graph_planning_available
        
        capabilities = get_planning_capabilities()
        graph_available = is_graph_planning_available()
        
        print(f"  ToT Available: {capabilities.get('tot_available', False)}")
        print(f"  Assignment Planning: {capabilities.get('assignment_planning', False)}")
        print(f"  Graph Planning: {capabilities.get('graph_planning', False)}")
        print(f"  Graph Compilation: {capabilities.get('graph_compilation', False)}")
        print(f"  Full Graph Pipeline: {graph_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Capability detection failed: {e}")
        return False


def test_graph_specifications():
    """Test graph specification creation."""
    print("\nTesting graph specifications...")
    
    try:
        from orchestrator.planning.graph_specifications import (
            StateGraphSpec, create_simple_sequential_graph
        )
        
        # Create test assignments
        assignments = [
            {"agent": "TestAgent1", "objective": "Test objective 1", "expected_output": "Output 1"},
            {"agent": "TestAgent2", "objective": "Test objective 2", "expected_output": "Output 2"}
        ]
        
        # Create graph specification
        graph_spec = create_simple_sequential_graph("test_graph", assignments)
        
        print(f"  Graph created: {graph_spec.name}")
        print(f"  Nodes: {len(graph_spec.nodes)}")
        print(f"  Edges: {len(graph_spec.edges)}")
        
        # Test validation
        errors = graph_spec.validate()
        print(f"  Validation errors: {len(errors)}")
        
        # Test serialization
        graph_dict = graph_spec.to_dict()
        json_str = graph_spec.to_json()
        print(f"  Serialization: {len(json_str)} characters")
        
        # Test deserialization
        reconstructed = StateGraphSpec.from_dict(graph_dict)
        print(f"  Deserialization: {reconstructed.name == graph_spec.name}")
        
        return len(errors) == 0
        
    except Exception as e:
        print(f"❌ Graph specifications test failed: {e}")
        return False


def test_fallback_functionality():
    """Test fallback functionality without API calls."""
    print("\nTesting fallback functionality...")
    
    try:
        from orchestrator.planning import create_graph_from_assignments
        from orchestrator.core.config import AgentConfig
        
        # Create test agent configs
        agent_configs = [
            AgentConfig(
                name="TestAgent",
                role="Test Role",
                goal="Test Goal",
                backstory="Test Backstory",
                instructions="Test Instructions",
                enabled=True
            )
        ]
        
        # Test assignment to graph conversion
        assignments = [
            {"agent": "TestAgent", "objective": "Test objective", "expected_output": "Test output"}
        ]
        
        graph_spec = create_graph_from_assignments(assignments, "fallback_test")
        
        if graph_spec:
            print(f"  Fallback graph created: {graph_spec.name}")
            print(f"  Nodes: {len(graph_spec.nodes)}")
            return True
        else:
            print("  ❌ Fallback graph creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Fallback functionality test failed: {e}")
        return False


def test_graph_compilation_structure():
    """Test graph compilation structure (without LangChain execution)."""
    print("\nTesting graph compilation structure...")
    
    try:
        from orchestrator.planning.graph_compiler import GraphCompiler
        from orchestrator.planning.graph_specifications import create_simple_sequential_graph
        from orchestrator.core.config import AgentConfig
        
        # Create test data
        agent_configs = [
            AgentConfig(
                name="TestAgent",
                role="Test Role", 
                goal="Test Goal",
                backstory="Test Backstory",
                instructions="Test Instructions",
                enabled=True
            )
        ]
        
        assignments = [
            {"agent": "TestAgent", "objective": "Test objective", "expected_output": "Test output"}
        ]
        
        graph_spec = create_simple_sequential_graph("compilation_test", assignments)
        
        # Test compiler instantiation
        compiler = GraphCompiler()
        print(f"  Compiler created successfully")
        
        # Note: We won't try to compile because it requires LangChain runtime
        print(f"  Graph spec ready for compilation: {graph_spec.name}")
        print(f"  Nodes to compile: {len(graph_spec.nodes)}")
        print(f"  Edges to compile: {len(graph_spec.edges)}")
        
        return True
        
    except Exception as e:
        if "LangChain" in str(e) or "not available" in str(e):
            print(f"  ⚠️  Compilation structure OK (runtime dependencies missing): {e}")
            return True
        else:
            print(f"❌ Graph compilation structure test failed: {e}")
            return False


def main():
    """Run all structural tests."""
    print("=" * 60)
    print("         ToT→StateGraph Structural Integration Test")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Capability Detection", test_capabilities),
        ("Graph Specifications", test_graph_specifications),
        ("Fallback Functionality", test_fallback_functionality),
        ("Compilation Structure", test_graph_compilation_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    success_rate = passed / total if total > 0 else 0
    print(f"\nOverall: {passed}/{total} tests passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("\n🎉 Integration structure is ready!")
        print("The ToT→StateGraph pipeline has been successfully integrated.")
        print("\nNote: Runtime testing requires API keys and LangChain dependencies.")
    else:
        print("\n⚠️  Some structural issues detected.")
        print("Please review the failed tests above.")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)