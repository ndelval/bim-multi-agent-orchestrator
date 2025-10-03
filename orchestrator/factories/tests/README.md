# GraphRAG Tool Integration - Test Suite Documentation

## Overview

Comprehensive integration test suite for GraphRAG tool attachment to agent configurations in LangGraph/PraisonAI orchestrator workflows. This suite validates the complete integration lifecycle from tool creation through graph compilation to execution.

**Test Coverage**: 22+ integration tests covering tool creation, configuration attachment, graph compilation, error handling, and edge cases.

---

## Quick Start

### 1. Setup Environment

```bash
# Install with uv (recommended)
uv pip install -e .[test]

# Verify setup
python orchestrator/factories/tests/validate_test_setup.py
```

### 2. Run Tests

```bash
# Full test suite
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v

# Quick validation (< 10s)
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -k "test_basic" -v

# With coverage
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator --cov-report=html
```

### 3. Review Results

```bash
# Open HTML coverage report
open htmlcov/index.html

# View test logs
cat test_output.log  # If using --log-file
```

---

## Documentation Structure

### Core Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| **README.md** (this file) | Overview and quick start | All users |
| **test_graphrag_tool_integration.py** | Executable test suite | Developers, QA |
| **GRAPHRAG_INTEGRATION_TEST_STRATEGY.md** | Comprehensive test strategy | QA Engineers, Architects |
| **UV_TEST_EXECUTION_GUIDE.md** | Test execution reference | Developers, CI/CD |
| **EDGE_CASES_CATALOG.md** | Edge case encyclopedia | QA Engineers, Developers |
| **validate_test_setup.py** | Setup validation script | All users |

### Document Relationships

```
README.md (You are here)
    │
    ├─→ validate_test_setup.py
    │   └─→ Quick environment check
    │
    ├─→ UV_TEST_EXECUTION_GUIDE.md
    │   └─→ How to run tests
    │
    ├─→ GRAPHRAG_INTEGRATION_TEST_STRATEGY.md
    │   ├─→ Test scenarios (TC-001 to TC-022)
    │   ├─→ Validation criteria
    │   └─→ Coverage metrics
    │
    ├─→ EDGE_CASES_CATALOG.md
    │   ├─→ 46 documented edge cases
    │   └─→ Mitigation strategies
    │
    └─→ test_graphrag_tool_integration.py
        └─→ Executable tests
```

---

## Test Architecture

### Integration Points Tested

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │         Memory Manager                          │   │
│   │  create_graph_tool() ─→ GraphRAGTool            │   │
│   └─────────────────────────┬───────────────────────┘   │
│                              │                           │
│   ┌──────────────────────────▼──────────────────────┐   │
│   │         AgentConfig                             │   │
│   │  tools: List[Callable] ← GraphRAGTool           │   │
│   └──────────────────────────┬──────────────────────┘   │
│                              │                           │
│   ┌──────────────────────────▼──────────────────────┐   │
│   │         AgentFactory                            │   │
│   │  create_agent(config) → LangChainAgent          │   │
│   └──────────────────────────┬──────────────────────┘   │
│                              │                           │
│   ┌──────────────────────────▼──────────────────────┐   │
│   │         GraphFactory                            │   │
│   │  create_*_graph() → StateGraph                  │   │
│   └──────────────────────────┬──────────────────────┘   │
│                              │                           │
│   ┌──────────────────────────▼──────────────────────┐   │
│   │         StateGraph Execution                    │   │
│   │  invoke(state) → Agent executes with tools      │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Test Classes

| Test Class | Tests | Focus Area |
|------------|-------|------------|
| **TestGraphRAGToolCreation** | 4 | Tool instantiation, basic execution |
| **TestToolAttachmentToAgentConfigs** | 4 | Configuration attachment, validation |
| **TestLangGraphWorkflowWithTools** | 3 | Graph compilation, node execution |
| **TestEdgeCasesAndErrors** | 8 | Error handling, edge cases |
| **TestIntegrationFlowEndToEnd** | 2 | Complete workflow validation |
| **TestPerformanceAndStress** | 2 | Performance, resource management |

---

## Test Execution Patterns

### Development Workflow

```bash
# 1. Make code changes
vim orchestrator/core/orchestrator.py

# 2. Run quick validation
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v

# 3. If passed, run full suite
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v

# 4. Check coverage
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator --cov-report=term-missing
```

### CI/CD Workflow

```bash
# Pre-commit hook
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v --tb=short

# Pull request validation
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v --tb=short --maxfail=3 --junitxml=test_results.xml

# Merge to main
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator --cov-report=xml --cov-fail-under=85
```

### Debugging Workflow

```bash
# Run single test with debug output
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_basic_tool_creation \
  -v -s --log-cli-level=DEBUG

# Drop into debugger on failure
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --pdb -x

# Show detailed traceback
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v --tb=long
```

---

## Test Coverage Metrics

### Current Coverage (Target)

| Component | Target | Status |
|-----------|--------|--------|
| Tool Creation | 95% | TBD |
| Tool Attachment | 90% | TBD |
| Graph Compilation | 85% | TBD |
| Error Handling | 95% | TBD |
| E2E Integration | 80% | TBD |
| Performance | 75% | TBD |

### Coverage Gaps Identified

1. **LangGraph unavailability**: Some tests skipped if LangGraph not installed
2. **Real database integration**: All DB interactions mocked
3. **Multi-provider testing**: Only hybrid provider covered
4. **Large-scale stress testing**: Performance tests use small datasets

**Mitigation Plans**: See `GRAPHRAG_INTEGRATION_TEST_STRATEGY.md` Section 6

---

## Key Test Scenarios

### TC-001: Basic Tool Creation
**What it tests**: GraphRAG tool instantiation from memory manager
**Why it matters**: Core functionality validation
**Run individually**:
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_basic_tool_creation -v
```

### TC-003: Tool Execution with Filters
**What it tests**: Parameter parsing (tags, documents, sections)
**Why it matters**: Advanced query capabilities
**Run individually**:
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_tool_execution_with_filters -v
```

### TC-009: Graph Compilation with Tools
**What it tests**: StateGraph compilation with tool-enabled agents
**Why it matters**: LangGraph integration validation
**Run individually**:
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestLangGraphWorkflowWithTools::test_graph_compilation_with_tools -v
```

### TC-017: Concurrent Tool Execution
**What it tests**: Thread-safety of tool execution
**Why it matters**: Production-ready concurrency
**Run individually**:
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestEdgeCasesAndErrors::test_concurrent_tool_execution -v
```

### TC-019: Complete Workflow Integration
**What it tests**: Full lifecycle from config to execution
**Why it matters**: End-to-end validation
**Run individually**:
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestIntegrationFlowEndToEnd::test_complete_workflow_with_graphrag_tool -v
```

---

## Edge Cases Summary

**Total Edge Cases Documented**: 46
**Test Coverage**: 50% (23 covered, 23 uncovered)

### Critical Edge Cases (Must Cover)

| Case ID | Description | Status |
|---------|-------------|--------|
| MEMORY-102 | Neo4j connection unavailable | ✓ Covered |
| GRAPH-201 | LangGraph not installed | ✓ Covered |
| ERROR-501 | Memory manager timeout | ✓ Covered |
| SEC-701 | SQL injection in filters | ✗ Not covered |

### High Priority Gaps

1. **EXEC-302**: Query exceeds token limit
2. **STATE-403**: Tool fails mid-execution
3. **ERROR-502**: Embedding generation fails

**Action**: See `EDGE_CASES_CATALOG.md` for complete list and prioritization

---

## Troubleshooting

### Common Issues

#### "LangGraph components not available"
**Solution**: Install LangGraph dependencies
```bash
uv pip install langgraph langchain langchain-openai
```

#### "Memory manager not initialized"
**Solution**: Verify fixture usage in test
```python
def test_something(self, mock_memory_manager):  # ← Fixture required
    tool = mock_memory_manager.create_graph_tool(...)
```

#### "pytest: command not found"
**Solution**: Install pytest
```bash
uv pip install pytest pytest-mock pytest-asyncio
```

#### Tests fail with import errors
**Solution**: Install package in editable mode
```bash
uv pip install -e .
```

### Debug Commands

```bash
# Validate environment
python orchestrator/factories/tests/validate_test_setup.py

# Check test discovery
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py --collect-only

# Show available fixtures
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py --fixtures

# Run with maximum verbosity
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -vv -s --log-cli-level=DEBUG
```

---

## Performance Baselines

### Expected Execution Times

| Test Type | Tests | Time (uv) | Time (pytest) |
|-----------|-------|-----------|---------------|
| Quick validation | 3-5 | < 5s | < 8s |
| Full suite | 22+ | < 30s | < 45s |
| Single test | 1 | < 1s | < 2s |
| Performance tests | 2 | < 5s | < 8s |

**Note**: All tests use mocked dependencies for fast execution

---

## Maintenance

### Update Triggers

**Update tests when**:
1. GraphRAG tool API changes
2. Memory manager interface changes
3. Agent configuration schema updates
4. LangGraph StateGraph API changes
5. New edge cases discovered in production

### Review Schedule

- **Weekly**: Review failed test logs, update mocks
- **Monthly**: Add new edge cases, update performance baselines
- **Quarterly**: Comprehensive coverage audit, refactor for maintainability

### Contact

**Maintainer**: Quality Engineering Team
**Last Updated**: 2025-01-21
**Next Review**: 2025-04-21

---

## Related Documentation

### Internal Documentation

- **Orchestrator Implementation**: `/orchestrator/core/orchestrator.py`
- **Tool Implementation**: `/orchestrator/tools/graph_rag_tool.py`
- **Memory Manager**: `/orchestrator/memory/memory_manager.py`
- **LangGraph Integration**: `/orchestrator/integrations/langchain_integration.py`
- **Graph Factory**: `/orchestrator/factories/graph_factory.py`

### External Documentation

- **pytest**: https://docs.pytest.org/
- **pytest-mock**: https://pytest-mock.readthedocs.io/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **uv**: https://github.com/astral-sh/uv

---

## Contributing

### Adding New Tests

1. **Identify integration point or edge case**
2. **Add test to appropriate test class**
3. **Update test strategy document**
4. **Run validation**:
   ```bash
   uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v
   ```
5. **Update edge case catalog if applicable**

### Test Naming Convention

```python
def test_<feature>_<scenario>_<expected_behavior>(self, fixtures):
    """
    Test <feature> <scenario>.

    Expected: <expected_behavior>
    Coverage: <related_edge_cases>
    """
    # Test implementation
```

**Example**:
```python
def test_tool_execution_with_filters_parses_csv_strings(self, mock_memory_manager):
    """
    Test tool execution with CSV filter strings.

    Expected: CSV strings parsed to lists correctly
    Coverage: EXEC-303, CONFIG-002
    """
```

---

## License and Usage

This test suite is part of the orchestrator project. See main project LICENSE for details.

**Usage**: Free for development, testing, and CI/CD purposes within the project.

---

## Quick Reference

### Essential Commands

```bash
# Setup
uv pip install -e .[test]
python orchestrator/factories/tests/validate_test_setup.py

# Run tests
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v

# Quick check
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -k "test_basic" -v

# With coverage
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py --cov=orchestrator --cov-report=html

# Debug mode
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py --pdb -x
```

### Documentation

- **Full test strategy**: `GRAPHRAG_INTEGRATION_TEST_STRATEGY.md`
- **Execution guide**: `UV_TEST_EXECUTION_GUIDE.md`
- **Edge cases**: `EDGE_CASES_CATALOG.md`

---

**Need Help?** Check troubleshooting section above or review detailed documentation in strategy document.