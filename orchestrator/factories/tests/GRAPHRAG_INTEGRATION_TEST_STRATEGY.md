# GraphRAG Tool Integration Testing Strategy

## Executive Summary

Comprehensive integration test strategy for GraphRAG tool attachment to agent configurations in LangGraph/PraisonAI workflows, covering tool creation, configuration attachment, graph compilation, and execution with systematic edge case coverage.

**Test File**: `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/factories/tests/test_graphrag_tool_integration.py`

**Execution Command**: `uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v`

---

## 1. Integration Test Architecture

### 1.1 Test Pyramid Structure

```
                    ┌─────────────────────┐
                    │   E2E Integration   │ ← 10% (Complete workflows)
                    └─────────────────────┘
                  ┌───────────────────────────┐
                  │  Component Integration    │ ← 30% (Graph + Tool)
                  └───────────────────────────┘
              ┌─────────────────────────────────────┐
              │    Unit Integration Tests           │ ← 60% (Tool creation/attachment)
              └─────────────────────────────────────┘
```

### 1.2 Integration Points Tested

1. **Tool Creation Layer**: `MemoryManager.create_graph_tool()` → `GraphRAGTool`
2. **Configuration Layer**: `GraphRAGTool` → `AgentConfig.tools[]`
3. **Factory Layer**: `AgentConfig` → `AgentFactory.create_agent()`
4. **Graph Compilation Layer**: `AgentFactory` → `GraphFactory.create_*_graph()`
5. **Execution Layer**: `StateGraph.invoke()` → Agent execution with tool access

### 1.3 Test Coverage Dimensions

| Dimension | Coverage | Test Classes |
|-----------|----------|--------------|
| **Tool Creation** | 95% | `TestGraphRAGToolCreation` |
| **Tool Attachment** | 90% | `TestToolAttachmentToAgentConfigs` |
| **Graph Compilation** | 85% | `TestLangGraphWorkflowWithTools` |
| **Error Handling** | 95% | `TestEdgeCasesAndErrors` |
| **E2E Integration** | 80% | `TestIntegrationFlowEndToEnd` |
| **Performance** | 75% | `TestPerformanceAndStress` |

---

## 2. Test Scenarios by Category

### 2.1 Tool Creation Tests (`TestGraphRAGToolCreation`)

#### TC-001: Basic Tool Creation
**Objective**: Verify GraphRAG tool creation from memory manager
**Preconditions**: Mock MemoryManager with `create_graph_tool()` support
**Steps**:
1. Initialize mock memory manager
2. Call `create_graph_tool(default_user_id="test_user", default_run_id="test_run")`
3. Validate returned callable has correct signature and documentation

**Expected Results**:
- Tool is callable function
- Has `__name__ == "graph_rag_lookup"`
- Has Spanish documentation mentioning "GraphRAG" or "documentos"

**Validation Criteria**:
```python
assert callable(tool)
assert tool.__name__ == "graph_rag_lookup"
assert "GraphRAG" in tool.__doc__ or "documentos" in tool.__doc__
```

---

#### TC-002: Tool Execution Basic
**Objective**: Verify tool execution with basic query
**Input**: `query="safety standards for industrial equipment", top_k=5`
**Expected Output**: String containing formatted results with document citations

**Validation Criteria**:
- Result is non-empty string
- Contains "documento" or "GraphRAG"
- Memory manager `retrieve_with_graph()` called once with correct query

---

#### TC-003: Tool Execution with Filters
**Objective**: Test parameter parsing for tags, documents, sections
**Input**:
```python
query="compliance requirements"
tags="safety,industrial"  # CSV string
documents="doc_001,doc_002"  # CSV string
sections="Section 1.1"
top_k=3
```

**Expected Behavior**: CSV strings parsed to lists before memory manager call

**Validation Criteria**:
```python
call_kwargs = mock_memory_manager.retrieve_with_graph.call_args[1]
assert call_kwargs['tags'] == ['safety', 'industrial']
assert call_kwargs['document_ids'] == ['doc_001', 'doc_002']
assert call_kwargs['sections'] == ['Section 1.1']
assert call_kwargs['limit'] == 3
```

---

#### TC-004: No Memory Manager Error
**Objective**: Verify error when memory manager not initialized
**Preconditions**: Orchestrator with `memory=False` in execution config
**Expected Error**: `MemoryError` or similar with message containing "Memory" and "not initialized"

---

### 2.2 Tool Attachment Tests (`TestToolAttachmentToAgentConfigs`)

#### TC-005: Single Agent Tool Attachment
**Objective**: Attach GraphRAG tool to single agent configuration
**Steps**:
1. Create `AgentConfig` with empty `tools=[]`
2. Create GraphRAG tool
3. Assign `agent_config.tools = [tool]`

**Validation Criteria**:
```python
assert len(agent_config.tools) == 1
assert callable(agent_config.tools[0])
```

---

#### TC-006: Multiple Agent Tool Sharing
**Objective**: Verify same tool instance can be shared across multiple agents
**Test Pattern**: Create 3 agents, assign same tool to all

**Expected Behavior**:
- All agents have `len(tools) == 1`
- All tool references point to same callable
- No conflicts when agents execute tool concurrently

---

#### TC-007: Multiple Tools per Agent
**Objective**: Test agent with GraphRAG tool + additional tools
**Configuration**:
```python
agent_config.tools = [graph_rag_tool, other_tool_1, other_tool_2]
```

**Validation**: Agent factory creates agent with all 3 tools accessible

---

#### TC-008: Tool Validation on Agent Creation
**Objective**: Verify validation of tool configurations during agent creation
**Negative Test**: Assign non-callable object to `tools` list

**Expected Error**: `TypeError`, `ValueError`, or `AttributeError` during `AgentFactory.create_agent()`

---

### 2.3 LangGraph Integration Tests (`TestLangGraphWorkflowWithTools`)

#### TC-009: Graph Compilation with Tool-Enabled Agents
**Objective**: Verify StateGraph compilation with agents containing tools
**Preconditions**: LangGraph available (`is_available() == True`)
**Steps**:
1. Create `OrchestratorConfig` with agent having GraphRAG tool
2. Initialize `Orchestrator`
3. Call `orchestrator._create_langgraph_system()`

**Expected Behavior**: Graph compiles without errors, tools accessible in nodes

**Note**: Test marked with `@pytest.mark.skipif(not _check_langgraph_available())`

---

#### TC-010: Agent Node with Tool Access
**Objective**: Verify agent execution nodes have access to configured tools
**Validation Method**: Mock `AgentFactory.create_agent()` and inspect call arguments

**Expected**:
- Agent created with `tools` parameter containing GraphRAG tool
- Agent execution context includes tool callable

---

#### TC-011: Graph State with Tool Results
**Objective**: Test `OrchestratorState` management of tool execution results
**Steps**:
1. Create `OrchestratorState` with initial prompt
2. Execute GraphRAG tool
3. Store result in `state.node_outputs["graph_rag_lookup"]`
4. Increment `state.current_iteration`

**Validation**:
```python
assert "graph_rag_lookup" in state.node_outputs
assert len(state.node_outputs["graph_rag_lookup"]) > 0
assert state.current_iteration == 1
```

---

### 2.4 Edge Cases and Error Handling (`TestEdgeCasesAndErrors`)

#### TC-012: Empty Results Handling
**Scenario**: Memory provider returns empty list `[]`
**Expected Output**: Friendly message like "No se encontraron fragmentos relevantes"

**Validation**:
```python
assert "no" in result.lower() or "sin" in result.lower()
```

---

#### TC-013: Memory Retrieval Failure
**Scenario**: `retrieve_with_graph()` raises exception
**Mock Configuration**: `Mock(side_effect=Exception("Memory retrieval failed"))`
**Expected**: Exception propagates or is handled with error message

---

#### TC-014: LangGraph Unavailability Fallback
**Scenario**: LangGraph not installed or unavailable
**Mock**: `patch('orchestrator.core.orchestrator.USING_LANGGRAPH', False)`
**Expected**: `OrchestratorError` with "LangGraph" in message when calling `_create_langgraph_system()`

---

#### TC-015: Invalid Tool Parameters
**Scenario**: Pass `top_k="invalid"` (string instead of int)
**Expected Behavior**: Tool defaults to `top_k=5` without crashing

**Validation**:
```python
call_kwargs = mock_memory_manager.retrieve_with_graph.call_args[1]
assert call_kwargs['limit'] == 5  # Default fallback
```

---

#### TC-016: Very Large top_k Value
**Scenario**: Pass `top_k=10000` (unreasonably large)
**Expected**: Value passed through to provider (provider may have its own limits)

---

#### TC-017: Concurrent Tool Execution
**Objective**: Test thread-safety of tool execution
**Method**: Execute tool from 5 concurrent threads
**Expected**:
- All 5 executions complete successfully
- No race conditions or deadlocks
- Memory manager called 5 times

---

#### TC-018: Missing Metadata Handling
**Scenario**: Results have incomplete `metadata` field
**Mock Data**:
```python
{
    "content": "Content without metadata",
    "score": 0.8
    # No metadata field
}
```

**Expected**: Tool handles gracefully, displays "desconocido" for missing fields

---

### 2.5 End-to-End Integration Tests (`TestIntegrationFlowEndToEnd`)

#### TC-019: Complete Workflow Integration
**Objective**: Test full lifecycle: config → tool creation → attachment → execution
**Steps**:
1. Create `Orchestrator` with memory-enabled config
2. Create GraphRAG tool via `orchestrator.create_graph_tool()`
3. Attach tool to agent config
4. Execute tool and verify results
5. Validate memory manager interactions

**Success Criteria**: All steps complete without errors, tool produces valid output

---

#### TC-020: Multi-Agent Workflow with Shared Tool
**Scenario**: 3 agents sharing single GraphRAG tool instance
**Validation**:
- All agents have tool in `tools` list
- Tool execution works from any agent context
- No conflicts during parallel execution

---

### 2.6 Performance and Stress Tests (`TestPerformanceAndStress`)

#### TC-021: Tool Execution Performance
**Test**: Execute tool 10 times sequentially
**Performance Target**: < 1 second total (with mocked memory manager)
**Validation**:
```python
assert elapsed_time < 1.0
assert mock_memory_manager.retrieve_with_graph.call_count == 10
```

---

#### TC-022: Memory Manager Cleanup
**Objective**: Verify proper resource cleanup
**Steps**:
1. Create orchestrator with memory manager
2. Create and execute tool
3. Call `orchestrator.cleanup()`

**Validation**: `mock_memory_manager.cleanup()` called exactly once

---

## 3. Edge Case Catalog

### 3.1 Configuration Edge Cases

| Case ID | Scenario | Expected Behavior | Test Coverage |
|---------|----------|-------------------|---------------|
| EC-001 | Agent with `tools=None` | Initialize as empty list `[]` | TC-005 |
| EC-002 | Agent with non-list tools | Raise `TypeError` | TC-008 |
| EC-003 | Tool assigned to disabled agent | Tool exists but agent not executed | Implicit |
| EC-004 | Agent config without `llm` field | Use default LLM | TC-005 |
| EC-005 | Circular tool dependencies | No issue (tools are independent) | N/A |

### 3.2 Memory Provider Edge Cases

| Case ID | Scenario | Expected Behavior | Test Coverage |
|---------|----------|-------------------|---------------|
| EC-101 | Memory provider not graph-capable | `create_graph_tool()` raises error | TC-004 |
| EC-102 | Neo4j connection unavailable | Error during tool creation | TC-013 |
| EC-103 | Chroma vector DB empty | Empty results, friendly message | TC-012 |
| EC-104 | Lexical index corrupted | Error propagates or degrades gracefully | TC-013 |
| EC-105 | Reranker model not loaded | Tool works without reranking | Implicit |

### 3.3 Graph Compilation Edge Cases

| Case ID | Scenario | Expected Behavior | Test Coverage |
|---------|----------|-------------------|---------------|
| EC-201 | LangGraph not installed | Fallback to PraisonAI | TC-014 |
| EC-202 | Agent with tools in PraisonAI mode | Tools passed to PraisonAI agent | TC-019 |
| EC-203 | Graph with no agents | Compilation fails with error | Not covered |
| EC-204 | Circular graph edges | LangGraph validation catches | Not covered |
| EC-205 | Agent node without execute method | Error during graph compilation | TC-010 |

### 3.4 Tool Execution Edge Cases

| Case ID | Scenario | Expected Behavior | Test Coverage |
|---------|----------|-------------------|---------------|
| EC-301 | Query with special characters | Escaped properly | Implicit |
| EC-302 | Query exceeds token limit | Truncation or error | Not covered |
| EC-303 | Tags/docs with invalid format | Parsed as empty list | TC-015 |
| EC-304 | `top_k=0` | Default to 5 or raise error | TC-015 |
| EC-305 | `top_k` negative | Default to 5 or raise error | TC-015 |
| EC-306 | Concurrent tool calls | Thread-safe execution | TC-017 |
| EC-307 | Tool called without query | Error or empty results | Not covered |
| EC-308 | Query in non-Latin script | Works if embedder supports | Not covered |

### 3.5 State Management Edge Cases

| Case ID | Scenario | Expected Behavior | Test Coverage |
|---------|----------|-------------------|---------------|
| EC-401 | Tool result exceeds state size limit | Truncation or pagination | Not covered |
| EC-402 | Multiple tool calls in single iteration | All results stored separately | TC-011 |
| EC-403 | Tool fails mid-execution | Error stored in `state.errors[]` | Not covered |
| EC-404 | State serialization with tool results | JSON-serializable output | Not covered |
| EC-405 | State persistence across graph restarts | Tool results retained | Not covered |

### 3.6 Error Propagation Edge Cases

| Case ID | Scenario | Expected Behavior | Test Coverage |
|---------|----------|-------------------|---------------|
| EC-501 | Memory manager timeout | Timeout error propagates | TC-013 |
| EC-502 | Embedding generation fails | Error or degraded search | TC-013 |
| EC-503 | Graph DB query error | Error with actionable message | TC-013 |
| EC-504 | Tool execution during agent error | Tool result available for recovery | Not covered |
| EC-505 | Orchestrator cleanup with pending tool calls | Graceful shutdown | TC-022 |

---

## 4. Test Execution Strategy

### 4.1 Test Execution Modes

#### Mode 1: Quick Validation (< 10 seconds)
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -k "test_basic" -v
```
**Coverage**: Tool creation and basic attachment (TC-001, TC-002, TC-005)

#### Mode 2: Full Integration (< 60 seconds)
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v
```
**Coverage**: All test classes, mocked integrations

#### Mode 3: Skip LangGraph Tests (for environments without LangGraph)
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v -m "not langgraph"
```
**Note**: Tests already have `@pytest.mark.skipif` decorators for LangGraph availability

#### Mode 4: Performance and Stress Only
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestPerformanceAndStress -v
```
**Coverage**: TC-021, TC-022

### 4.2 Continuous Integration Configuration

#### pytest.ini Configuration
```ini
[pytest]
pythonpath = .
testpaths = orchestrator/factories/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    langgraph: tests requiring LangGraph installation
    integration: integration tests
    performance: performance and stress tests
    edge_case: edge case and error handling tests
```

#### CI Pipeline Steps
1. **Dependency Installation**: `uv pip install -e .[test]`
2. **Quick Validation**: Run Mode 1 on every commit
3. **Full Integration**: Run Mode 2 on PR to main
4. **Performance Baseline**: Run Mode 4 weekly, track metrics

### 4.3 Test Isolation and Cleanup

**Fixture Scope**: All fixtures use `function` scope for complete isolation

**Cleanup Strategy**:
- Mocks reset automatically after each test
- No real Neo4j/ChromaDB connections (all mocked)
- No filesystem writes (in-memory only)

**Resource Management**:
```python
@pytest.fixture
def mock_memory_manager():
    manager = Mock(spec=MemoryManager)
    # ... setup mocks
    yield manager
    # Automatic cleanup via Mock
```

---

## 5. Validation Criteria

### 5.1 Test Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Test Pass Rate** | ≥ 98% | TBD | - |
| **Code Coverage** | ≥ 85% | TBD | - |
| **Branch Coverage** | ≥ 80% | TBD | - |
| **Performance** | < 60s full suite | TBD | - |
| **Flakiness Rate** | < 1% | TBD | - |

### 5.2 Integration Quality Gates

**Pre-Merge Requirements**:
1. ✅ All tests pass in CI
2. ✅ Coverage ≥ 85% for modified files
3. ✅ No new edge cases identified without tests
4. ✅ Performance tests within baseline ± 10%

**Post-Merge Monitoring**:
1. Monitor test execution time trends
2. Track flaky test occurrences
3. Review edge case coverage quarterly

### 5.3 Validation Commands

#### Check Test Coverage
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --cov=orchestrator.core.orchestrator \
  --cov=orchestrator.tools.graph_rag_tool \
  --cov=orchestrator.memory.memory_manager \
  --cov-report=term-missing \
  --cov-report=html
```

#### Run with Verbose Output
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  -v --tb=short --log-cli-level=INFO
```

#### Generate Test Report
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --html=report.html --self-contained-html
```

---

## 6. Known Limitations and Future Work

### 6.1 Current Test Limitations

1. **No Real Neo4j Testing**: All graph DB interactions mocked
   - **Mitigation**: Add optional integration tests with test Neo4j container
   - **Priority**: Medium

2. **Limited LangGraph Coverage**: Some tests skipped if LangGraph unavailable
   - **Mitigation**: Ensure CI environment has LangGraph installed
   - **Priority**: High

3. **No Multi-Provider Testing**: Only hybrid provider tested
   - **Mitigation**: Add parametrized tests for mem0/rag providers
   - **Priority**: Low

4. **No Large-Scale Stress Testing**: Performance tests use small datasets
   - **Mitigation**: Add separate stress test suite with larger datasets
   - **Priority**: Medium

### 6.2 Future Enhancements

#### Phase 2: Real Integration Tests (Q2 2025)
- Docker-based Neo4j test container
- Real ChromaDB integration tests
- End-to-end tests with actual LLM calls (optional)

#### Phase 3: Performance Optimization (Q3 2025)
- Benchmark tool execution latency
- Optimize memory manager caching
- Profile graph compilation time

#### Phase 4: Advanced Scenarios (Q4 2025)
- Multi-modal document retrieval
- Dynamic tool registration during execution
- Tool result caching strategies

---

## 7. Troubleshooting Guide

### 7.1 Common Test Failures

#### Failure: "LangGraph components not available"
**Cause**: LangGraph not installed
**Solution**:
```bash
uv pip install langgraph langchain langchain-openai
```

#### Failure: "Memory manager not initialized"
**Cause**: Test fixture not providing memory manager
**Solution**: Verify `mock_memory_manager` fixture is used and yielding properly

#### Failure: "Tool execution timeout"
**Cause**: Mock not configured to return immediately
**Solution**: Ensure `retrieve_with_graph` mock returns list, not generator

### 7.2 Debugging Commands

#### Run Single Test with Debug Output
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py::TestGraphRAGToolCreation::test_basic_tool_creation \
  -v -s --log-cli-level=DEBUG
```

#### Interactive Debugging with pdb
```bash
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py \
  --pdb -x
```

#### Check Mock Call History
```python
# In test:
print(mock_memory_manager.method_calls)
print(mock_memory_manager.retrieve_with_graph.call_args_list)
```

---

## 8. Maintenance and Updates

### 8.1 Test Maintenance Schedule

**Weekly**:
- Review failed test logs
- Update mocks for API changes

**Monthly**:
- Review and add new edge cases
- Update performance baselines

**Quarterly**:
- Comprehensive edge case audit
- Coverage gap analysis
- Refactor for maintainability

### 8.2 Test Update Triggers

**Update tests when**:
1. GraphRAG tool API changes
2. Memory manager interface changes
3. Agent configuration schema updates
4. LangGraph StateGraph API changes
5. New edge cases discovered in production

---

## Appendix A: Test File Structure

```
orchestrator/factories/tests/
├── test_graphrag_tool_integration.py     # Main integration test file
├── conftest.py                           # Shared fixtures
├── __init__.py
└── GRAPHRAG_INTEGRATION_TEST_STRATEGY.md # This document
```

## Appendix B: Related Documentation

- **Tool Implementation**: `/orchestrator/tools/graph_rag_tool.py`
- **Memory Manager**: `/orchestrator/memory/memory_manager.py`
- **Orchestrator**: `/orchestrator/core/orchestrator.py`
- **LangGraph Integration**: `/orchestrator/integrations/langchain_integration.py`
- **Graph Factory**: `/orchestrator/factories/graph_factory.py`

## Appendix C: Contact and Support

**Test Maintainer**: Quality Engineering Team
**Last Updated**: 2025-01-21
**Review Cycle**: Quarterly
**Next Review**: 2025-04-21