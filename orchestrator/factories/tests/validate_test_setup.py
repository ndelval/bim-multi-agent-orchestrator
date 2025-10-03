#!/usr/bin/env python3
"""
Validation script for GraphRAG tool integration test setup.

This script checks:
1. Python environment and dependencies
2. Test file existence and syntax
3. Module imports
4. Test discovery
5. Quick test execution

Usage:
    python orchestrator/factories/tests/validate_test_setup.py
    uv run python orchestrator/factories/tests/validate_test_setup.py
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
import importlib.util


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_check(text: str, status: bool, details: str = "") -> None:
    """Print a check result with colored status."""
    symbol = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    print(f"{symbol} {text}")
    if details:
        print(f"  {Colors.YELLOW}→ {details}{Colors.END}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    required = (3, 11)
    current = sys.version_info[:2]

    if current >= required:
        return True, f"Python {current[0]}.{current[1]} (required: ≥ {required[0]}.{required[1]})"
    else:
        return False, f"Python {current[0]}.{current[1]} (required: ≥ {required[0]}.{required[1]})"


def check_dependency(package: str) -> Tuple[bool, str]:
    """Check if a Python package is installed."""
    try:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            return True, f"{package} installed"
        else:
            return False, f"{package} not found"
    except (ImportError, ModuleNotFoundError):
        return False, f"{package} not found"


def check_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if a file exists."""
    if file_path.exists():
        size = file_path.stat().st_size
        return True, f"{file_path.name} ({size:,} bytes)"
    else:
        return False, f"{file_path} not found"


def check_module_import(module_path: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        module = __import__(module_path, fromlist=[''])
        return True, f"{module_path} imported successfully"
    except ImportError as e:
        return False, f"{module_path} import failed: {str(e)}"


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, f"{description} successful"
        else:
            return False, f"{description} failed:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return False, f"{description} timed out"
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except Exception as e:
        return False, f"{description} error: {str(e)}"


def main() -> int:
    """Main validation routine."""
    all_checks_passed = True

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    test_file = script_dir / "test_graphrag_tool_integration.py"

    os.chdir(project_root)

    print_header("GraphRAG Tool Integration - Test Setup Validation")

    # ========================================================================
    # 1. Python Environment Checks
    # ========================================================================
    print(f"{Colors.BOLD}1. Python Environment{Colors.END}")

    status, details = check_python_version()
    print_check("Python version", status, details)
    all_checks_passed &= status

    # ========================================================================
    # 2. Core Dependencies
    # ========================================================================
    print(f"\n{Colors.BOLD}2. Core Dependencies{Colors.END}")

    core_deps = [
        "pytest",
        "pytest_mock",
        "pytest_asyncio",
        "pydantic",
        "langchain",
        "langgraph"
    ]

    for dep in core_deps:
        status, details = check_dependency(dep)
        print_check(f"Package: {dep}", status, details)
        all_checks_passed &= status

    # ========================================================================
    # 3. Test Files
    # ========================================================================
    print(f"\n{Colors.BOLD}3. Test Files{Colors.END}")

    test_files = [
        test_file,
        script_dir / "conftest.py",
        script_dir / "GRAPHRAG_INTEGRATION_TEST_STRATEGY.md",
        script_dir / "UV_TEST_EXECUTION_GUIDE.md",
        script_dir / "EDGE_CASES_CATALOG.md"
    ]

    for file_path in test_files:
        status, details = check_file_exists(file_path)
        print_check(f"File: {file_path.name}", status, details)
        # Only mark as failure if test file missing
        if file_path == test_file:
            all_checks_passed &= status

    # ========================================================================
    # 4. Module Imports
    # ========================================================================
    print(f"\n{Colors.BOLD}4. Module Imports{Colors.END}")

    modules = [
        "orchestrator.core.orchestrator",
        "orchestrator.core.config",
        "orchestrator.memory.memory_manager",
        "orchestrator.tools.graph_rag_tool",
        "orchestrator.integrations.langchain_integration"
    ]

    for module in modules:
        status, details = check_module_import(module)
        print_check(f"Module: {module}", status, details)
        all_checks_passed &= status

    # ========================================================================
    # 5. Test Discovery
    # ========================================================================
    print(f"\n{Colors.BOLD}5. Test Discovery{Colors.END}")

    # Check if pytest can discover tests
    status, details = run_command(
        ["pytest", str(test_file), "--collect-only", "-q"],
        "Test discovery"
    )
    print_check("pytest --collect-only", status, details)
    all_checks_passed &= status

    # ========================================================================
    # 6. Quick Test Execution
    # ========================================================================
    print(f"\n{Colors.BOLD}6. Quick Test Execution{Colors.END}")

    print(f"{Colors.YELLOW}Running basic tests (this may take 5-10 seconds)...{Colors.END}\n")

    status, details = run_command(
        ["pytest", str(test_file), "-k", "test_basic", "-v", "--tb=short"],
        "Quick test run"
    )
    print_check("pytest -k 'test_basic'", status, details)

    if not status:
        print(f"\n{Colors.YELLOW}Note: Test failures don't necessarily mean setup is broken.{Colors.END}")
        print(f"{Colors.YELLOW}Review the error output above for details.{Colors.END}")

    # ========================================================================
    # 7. Summary
    # ========================================================================
    print_header("Validation Summary")

    if all_checks_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All critical checks passed!{Colors.END}\n")
        print("You can now run the full test suite:")
        print(f"  {Colors.BLUE}uv run pytest {test_file} -v{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed{Colors.END}\n")
        print("Please address the failures above before running tests.\n")
        print("Common fixes:")
        print(f"  {Colors.BLUE}# Install missing dependencies{Colors.END}")
        print(f"  {Colors.BLUE}uv pip install -e .[test]{Colors.END}\n")
        print(f"  {Colors.BLUE}# Or install specific packages{Colors.END}")
        print(f"  {Colors.BLUE}uv pip install pytest pytest-mock pytest-asyncio{Colors.END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())