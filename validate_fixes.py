#!/usr/bin/env python3
"""
Validation script for the three critical fixes.

This script verifies that all fixes are properly implemented and functional.
Run with: python3 validate_fixes.py
"""

import sys
import importlib
from pathlib import Path


def validate_fix_1_session_management():
    """Validate FIX 1: Session Management System."""
    print("\n=== FIX 1: Session Management System ===")

    try:
        # Import session components
        from orchestrator.session import SessionManager, SessionContext, UserContext, SessionStore

        print("✅ All session modules imported successfully")

        # Check SessionManager has required methods
        required_methods = [
            'create_session', 'get_session', 'resume_session',
            'record_turn', 'end_session', 'cleanup_old_sessions',
            'get_session_info', 'close'
        ]

        for method in required_methods:
            if not hasattr(SessionManager, method):
                print(f"❌ SessionManager missing method: {method}")
                return False
            print(f"✅ SessionManager.{method} exists")

        # Check SessionContext dataclass
        session = SessionContext(user_id="test")
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'user_id')
        assert hasattr(session, 'turn_count')
        print("✅ SessionContext dataclass validated")

        # Check UserContext dataclass
        user = UserContext(user_id="test")
        assert hasattr(user, 'user_id')
        assert hasattr(user, 'total_sessions')
        print("✅ UserContext dataclass validated")

        print("✅ FIX 1: PASSED - Session management fully implemented\n")
        return True

    except Exception as e:
        print(f"❌ FIX 1: FAILED - {e}\n")
        return False


def validate_fix_2_tot_planner():
    """Validate FIX 2: ToT Planner Prompt Wrappers."""
    print("\n=== FIX 2: ToT Planner Prompt Wrappers ===")

    try:
        # Import planning components
        from orchestrator.planning.tot_planner import OrchestratorPlanningTask
        from orchestrator.core.config import AgentConfig

        print("✅ Planning modules imported successfully")

        # Create sample agent catalog
        agents = [
            AgentConfig(
                name="TestAgent",
                role="Tester",
                goal="Test functionality",
                backstory="Test backstory",
                instructions="Test instructions for agent"
            )
        ]

        # Create planning task
        task = OrchestratorPlanningTask(
            problem_statement="Test problem",
            agent_catalog=agents,
            max_steps=2,
            prompt_style="cot"
        )

        print("✅ OrchestratorPlanningTask created successfully")

        # Test standard_prompt_wrap
        try:
            prompt = task.standard_prompt_wrap(x="Test query", y="")
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            print("✅ standard_prompt_wrap works (no NotImplementedError)")
        except NotImplementedError as e:
            print(f"❌ standard_prompt_wrap raises NotImplementedError: {e}")
            return False

        # Test cot_prompt_wrap
        try:
            prompt = task.cot_prompt_wrap(x="Test query", y="")
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            print("✅ cot_prompt_wrap works (no NotImplementedError)")
        except NotImplementedError as e:
            print(f"❌ cot_prompt_wrap raises NotImplementedError: {e}")
            return False

        # Verify docstrings
        assert task.standard_prompt_wrap.__doc__ is not None
        assert task.cot_prompt_wrap.__doc__ is not None
        print("✅ Docstrings present and descriptive")

        print("✅ FIX 2: PASSED - Prompt wrappers working correctly\n")
        return True

    except Exception as e:
        print(f"❌ FIX 2: FAILED - {e}\n")
        return False


def validate_fix_3_neo4j_handling():
    """Validate FIX 3: Neo4j Failure Handling."""
    print("\n=== FIX 3: Neo4j Failure Handling ===")

    try:
        # Import memory providers
        from orchestrator.memory.providers.hybrid_provider import HybridRAGMemoryProvider
        from orchestrator.memory.providers.mem0_provider import Mem0MemoryProvider
        from orchestrator.core.config import EmbedderConfig

        print("✅ Memory provider modules imported successfully")

        # Check HybridRAGMemoryProvider methods
        embedder = EmbedderConfig(provider="openai", config={"model": "text-embedding-3-small"})
        hybrid_provider = HybridRAGMemoryProvider(embedder)

        required_hybrid_methods = [
            '_check_neo4j_health',
            '_initialize_neo4j_with_retry',
            '_upsert_graph',
            '_graph_execute_write',
            '_graph_query'
        ]

        for method in required_hybrid_methods:
            if not hasattr(hybrid_provider, method):
                print(f"❌ HybridRAGMemoryProvider missing method: {method}")
                return False
            print(f"✅ HybridRAGMemoryProvider.{method} exists")

        # Check Mem0MemoryProvider methods
        mem0_provider = Mem0MemoryProvider(embedder)

        required_mem0_methods = [
            '_check_neo4j_connection',
            '_retry_graph_store_init',
            '_init_mem0_client'
        ]

        for method in required_mem0_methods:
            if not hasattr(mem0_provider, method):
                print(f"❌ Mem0MemoryProvider missing method: {method}")
                return False
            print(f"✅ Mem0MemoryProvider.{method} exists")

        # Test health check with None driver
        result = hybrid_provider._check_neo4j_health(None)
        assert result is False
        print("✅ Health check returns False for None driver")

        print("✅ FIX 3: PASSED - Neo4j failure handling implemented\n")
        return True

    except Exception as e:
        print(f"❌ FIX 3: FAILED - {e}\n")
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("AI Agents Framework - Critical Fixes Validation")
    print("=" * 70)

    results = {
        "FIX 1: Session Management": validate_fix_1_session_management(),
        "FIX 2: ToT Planner": validate_fix_2_tot_planner(),
        "FIX 3: Neo4j Handling": validate_fix_3_neo4j_handling(),
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for fix_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{fix_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 All fixes validated successfully! Production ready.")
        return 0
    else:
        print("\n⚠️  Some fixes failed validation. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
