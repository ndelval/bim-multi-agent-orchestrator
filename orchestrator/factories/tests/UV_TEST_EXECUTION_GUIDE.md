# UV-Based Test Execution Guide
## GraphRAG Tool Integration Tests

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies with uv (fast, recommended)
uv pip install -e .[test]

# Or install with specific feature sets
uv pip install -e .[all-basic]  # Includes LangGraph + test dependencies
```

### 2. Basic Test Execution

```bash
# Run all GraphRAG integration tests
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v

# Expected output:
# =================== test session starts ===================
# collected 22 items
#
# test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_basic_tool_creation PASSED
# test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_tool_execution_basic PASSED
# ... [20 more tests]
# =================== 22 passed in 2.34s ===================
```

---

## Test Execution Modes

### Mode 1: Quick Validation (< 10 seconds)
Run only basic tool creation and attachment tests:

```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v
```

**Tests Executed**: 3-5 tests
**Coverage**: Tool creation, basic attachment, simple execution

---

### Mode 2: Full Test Suite (< 60 seconds)
Run all integration tests with detailed output:

```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v
```

**Tests Executed**: 22+ tests
**Coverage**: All test classes, complete edge case coverage

---

### Mode 3: Specific Test Class
Run tests from a single test class:

```bash
# Tool creation tests only
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestGraphRAGToolCreation -v

# Edge case tests only
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestEdgeCasesAndErrors -v

# Performance tests only
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestPerformanceAndStress -v
```

---

### Mode 4: Single Test Execution
Run a specific test with detailed logging:

```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_basic_tool_creation \
  -v -s --log-cli-level=INFO
```

**Use cases**: Debugging, CI troubleshooting, focused validation

---

## Advanced Test Options

### With Code Coverage

```bash
# Generate coverage report
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator.core.orchestrator \
  --cov=orchestrator.tools.graph_rag_tool \
  --cov=orchestrator.memory.memory_manager \
  --cov-report=term-missing

# Generate HTML coverage report
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator \
  --cov-report=html

# View HTML report: open htmlcov/index.html
```

**Expected Coverage**:
- `orchestrator.core.orchestrator.py`: ≥ 75%
- `orchestrator.tools.graph_rag_tool.py`: ≥ 90%
- `orchestrator.memory.memory_manager.py`: ≥ 70%

---

### With Performance Profiling

```bash
# Run with execution time reporting
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --durations=10 -v

# Expected output:
# =================== slowest 10 durations ===================
# 0.12s call test_concurrent_tool_execution
# 0.08s call test_complete_workflow_with_graphrag_tool
# ... [8 more entries]
```

---

### With Test Output Capture

```bash
# Show print statements and logs
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v -s --log-cli-level=DEBUG

# Save output to file
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v --log-file=test_output.log --log-file-level=DEBUG
```

---

### With Parallel Execution

```bash
# Install pytest-xdist
uv pip install pytest-xdist

# Run tests in parallel (4 workers)
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v -n 4

# Expected speedup: 2-3x faster for full suite
```

---

### Interactive Debugging

```bash
# Drop into debugger on first failure
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --pdb -x

# Drop into debugger on all failures
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --pdb

# Use with breakpoint() in test code for controlled debugging
```

---

## Test Filtering

### By Test Name Pattern

```bash
# Run all tests with "tool" in name
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "tool" -v

# Run all tests with "error" OR "edge" in name
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "error or edge" -v

# Run tests NOT matching pattern
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "not performance" -v
```

---

### By Test Markers

```bash
# Run only integration tests (if marked)
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -m integration -v

# Run everything except performance tests
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -m "not performance" -v
```

---

### By Test Collection

```bash
# Show what tests would run without executing
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --collect-only

# Expected output:
# <Module test_graphrag_tool_integration.py>
#   <Class TestGraphRAGToolCreation>
#     <Function test_basic_tool_creation>
#     <Function test_tool_execution_basic>
#     ... [20 more functions]
```

---

## Continuous Integration Commands

### CI Pipeline - Quick Check (< 10s)
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v --tb=short
```

**Use case**: Pre-commit hook, rapid feedback on basic functionality

---

### CI Pipeline - Full Validation (< 60s)
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v --tb=short --maxfail=3 \
  --junitxml=test_results.xml
```

**Use case**: Pull request validation, merge checks

**Outputs**:
- `test_results.xml`: JUnit-format results for CI dashboards
- Exit code 0 on success, non-zero on failure

---

### CI Pipeline - With Coverage Enforcement
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator \
  --cov-report=xml \
  --cov-report=term \
  --cov-fail-under=85 \
  --junitxml=test_results.xml
```

**Use case**: Quality gate enforcement, coverage regression prevention

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "ModuleNotFoundError: No module named 'pytest'"
**Solution**:
```bash
uv pip install pytest pytest-mock pytest-asyncio
```

---

#### Issue: "LangGraph components not available"
**Solution**:
```bash
uv pip install langgraph langchain langchain-openai
```

**Alternative**: Skip LangGraph tests:
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v -k "not langgraph"
```

---

#### Issue: "ImportError: cannot import name 'OrchestratorState'"
**Solution**: Verify package is installed in editable mode:
```bash
uv pip install -e .
```

---

#### Issue: Tests fail with "Mock object has no attribute 'retrieve_with_graph'"
**Solution**: Check fixture usage - ensure `mock_memory_manager` fixture is passed to test

---

#### Issue: "collected 0 items"
**Solution**: Check test discovery patterns:
```bash
# Verify test file location
ls orchestrator/factories/tests/test_graphrag_tool_integration.py

# Check Python path
uv run python -c "import orchestrator; print(orchestrator.__file__)"
```

---

## Test Result Interpretation

### Success Output
```
=================== test session starts ===================
platform darwin -- Python 3.11.x, pytest-8.x.x
collected 22 items

test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_basic_tool_creation PASSED [  4%]
test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_tool_execution_basic PASSED [  9%]
... [18 more PASSED]
test_graphrag_tool_integration.py::TestPerformanceAndStress::test_memory_manager_cleanup PASSED [100%]

=================== 22 passed in 2.34s ===================
```

**Interpretation**: All integration points validated successfully

---

### Partial Failure Output
```
=================== test session starts ===================
... [successful tests]
test_graphrag_tool_integration.py::TestEdgeCasesAndErrors::test_tool_execution_memory_error FAILED [ 68%]

_________________________ FAILURES _________________________
________________ test_tool_execution_memory_error __________

    def test_tool_execution_memory_error(self, mock_memory_manager):
        mock_memory_manager.retrieve_with_graph = Mock(
            side_effect=Exception("Memory retrieval failed")
        )
>       with pytest.raises(Exception) as exc_info:
E       Failed: DID NOT RAISE <class 'Exception'>

=================== 1 failed, 21 passed in 2.45s ===================
```

**Interpretation**: Edge case handling needs review - tool may be swallowing exception

---

### Skip Output
```
test_graphrag_tool_integration.py::TestLangGraphWorkflowWithTools::test_graph_compilation_with_tools SKIPPED [ 50%]

reason: LangGraph not available
```

**Interpretation**: Test skipped due to missing optional dependency (expected behavior)

---

## Performance Baselines

### Expected Execution Times (Mocked Tests)

| Test Suite | Tests | Time (uv run) | Time (pytest) |
|------------|-------|---------------|---------------|
| Quick Validation | 3-5 | < 5s | < 8s |
| Full Suite | 22+ | < 30s | < 45s |
| Single Test | 1 | < 1s | < 2s |
| Performance Tests | 2 | < 5s | < 8s |

**Note**: Times measured on macOS M1, Python 3.11, mocked dependencies

---

## Integration with Development Workflow

### Pre-Commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v --tb=short

if [ $? -ne 0 ]; then
    echo "GraphRAG integration tests failed - commit aborted"
    exit 1
fi
```

---

### Watch Mode (Development)
```bash
# Install pytest-watch
uv pip install pytest-watch

# Run tests on file changes
uv run ptw orchestrator/factories/tests/test_graphrag_tool_integration.py -- -v
```

---

### VSCode Integration

**settings.json**:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "orchestrator/factories/tests/test_graphrag_tool_integration.py",
    "-v"
  ],
  "python.testing.unittestEnabled": false
}
```

---

## Best Practices

### 1. Always Use Virtual Environment
```bash
# Create and activate with uv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Then install with uv
uv pip install -e .[test]
```

---

### 2. Keep Dependencies Updated
```bash
# Update all test dependencies
uv pip install --upgrade pytest pytest-mock pytest-asyncio

# Verify versions
uv pip list | grep pytest
```

---

### 3. Run Tests Before Commits
```bash
# Quick validation before commit
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v --tb=short
```

---

### 4. Use Verbose Mode During Development
```bash
# See full test names and logging
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v -s --log-cli-level=INFO
```

---

### 5. Clean Test Cache Periodically
```bash
# Remove pytest cache
rm -rf .pytest_cache
rm -rf orchestrator/__pycache__
rm -rf orchestrator/factories/__pycache__

# Re-run tests
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v
```

---

## Additional Resources

- **pytest Documentation**: https://docs.pytest.org/
- **pytest-mock Guide**: https://pytest-mock.readthedocs.io/
- **uv Documentation**: https://github.com/astral-sh/uv
- **Test Strategy Document**: `GRAPHRAG_INTEGRATION_TEST_STRATEGY.md`

---

## Quick Reference Card

```bash
# Essential Commands
uv run pytest <file> -v                           # Run with verbose output
uv run pytest <file> -k "pattern" -v              # Filter by test name
uv run pytest <file>::TestClass::test_func -v     # Run specific test
uv run pytest <file> --cov=orchestrator -v        # With coverage
uv run pytest <file> --pdb -x                     # Debug on first failure
uv run pytest <file> -v --tb=short                # Short traceback
uv run pytest <file> --collect-only               # Show tests without running
uv run pytest <file> --durations=10               # Show slowest tests
```

---

**Last Updated**: 2025-01-21
**Maintained By**: Quality Engineering Team
**Related Docs**: `GRAPHRAG_INTEGRATION_TEST_STRATEGY.md`