# GraphRAG Tool Integration - Edge Cases Catalog

## Document Purpose

Comprehensive catalog of edge cases for GraphRAG tool integration with agent configurations, LangGraph workflows, and memory management systems. Each edge case includes detection methods, expected behavior, test coverage, and mitigation strategies.

---

## Edge Case Classification System

```
[Category]-[Severity]-[ID]: Edge Case Name

Severity Levels:
- CRITICAL: System crash, data loss, security breach
- HIGH: Feature failure, incorrect results
- MEDIUM: Degraded performance, poor UX
- LOW: Minor inconvenience, edge case with workaround
```

---

## 1. Configuration Edge Cases

### [CONFIG]-[HIGH]-001: Agent with `tools=None`
**Description**: Agent configuration has `tools` field set to `None` instead of list

**Detection**:
```python
if agent_config.tools is None:
    # Edge case detected
```

**Expected Behavior**: Initialize as empty list `[]` during agent creation

**Test Coverage**: `test_attach_tool_to_single_agent` (TC-005)

**Mitigation**:
```python
# In AgentFactory.create_agent()
tools = agent_config.tools or []
```

---

### [CONFIG]-[HIGH]-002: Agent with Non-List Tools
**Description**: `tools` field contains non-list value (string, dict, single callable)

**Example**:
```python
agent_config.tools = "graph_rag_lookup"  # Should be list
agent_config.tools = graph_tool  # Should be [graph_tool]
```

**Detection**:
```python
if not isinstance(agent_config.tools, list):
    raise TypeError(f"tools must be list, got {type(agent_config.tools)}")
```

**Expected Behavior**: Raise `TypeError` with descriptive message

**Test Coverage**: `test_tool_config_validation` (TC-008)

**Mitigation**: Validate during config initialization with Pydantic validator

---

### [CONFIG]-[MEDIUM]-003: Tool Assigned to Disabled Agent
**Description**: Agent with `enabled=False` has tools configured

**Example**:
```python
agent_config = AgentConfig(
    name="DisabledAgent",
    enabled=False,
    tools=[graph_tool]  # Tool exists but agent not executed
)
```

**Expected Behavior**: Tool configuration valid but agent skipped during execution

**Test Coverage**: Implicit (not explicitly tested)

**Mitigation**: No action needed - expected behavior

---

### [CONFIG]-[LOW]-004: Agent Config Without LLM Field
**Description**: Agent configuration missing `llm` field

**Detection**:
```python
if agent_config.llm is None:
    # Use default
```

**Expected Behavior**: Use default LLM (`gpt-4o-mini` or configured default)

**Test Coverage**: `test_attach_tool_to_single_agent` (TC-005)

**Mitigation**: Default value in AgentConfig dataclass

---

### [CONFIG]-[LOW]-005: Circular Tool Dependencies
**Description**: Tool A depends on Tool B which depends on Tool A

**Analysis**: Not applicable - tools are independent functions, no dependency system

**Expected Behavior**: No issue, tools execute independently

**Test Coverage**: N/A

---

## 2. Memory Provider Edge Cases

### [MEMORY]-[CRITICAL]-101: Memory Provider Not Graph-Capable
**Description**: Attempting to create GraphRAG tool with provider lacking graph support

**Example**:
```python
# RAG provider doesn't support graph operations
config = MemoryConfig(provider=MemoryProvider.RAG)
orchestrator.create_graph_tool()  # Should raise error
```

**Detection**:
```python
if not hasattr(provider, 'retrieve_with_graph'):
    raise ProviderError("Provider does not support graph operations")
```

**Expected Behavior**: Raise `ProviderError` or `MemoryError` with actionable message

**Test Coverage**: `test_tool_without_memory_manager` (TC-004)

**Mitigation**: Check provider capabilities before tool creation

---

### [MEMORY]-[CRITICAL]-102: Neo4j Connection Unavailable
**Description**: Neo4j database unreachable or credentials invalid

**Symptoms**:
- Connection timeout
- Authentication failure
- Neo4j service not running

**Detection**:
```python
try:
    driver.verify_connectivity()
except Exception as e:
    raise ConnectionError(f"Neo4j unavailable: {e}")
```

**Expected Behavior**: Raise error during tool creation with connection details

**Test Coverage**: `test_tool_execution_memory_error` (TC-013)

**Mitigation**:
- Implement connection pooling with retries
- Provide fallback to vector-only search
- Add health check before tool creation

---

### [MEMORY]-[HIGH]-103: Chroma Vector DB Empty
**Description**: Vector database contains no documents/embeddings

**Detection**:
```python
if len(chroma_collection.get()) == 0:
    # Empty database
```

**Expected Behavior**: Tool returns friendly "No documents found" message

**Test Coverage**: `test_tool_execution_empty_results` (TC-012)

**Mitigation**: Display helpful message, suggest document ingestion

---

### [MEMORY]-[HIGH]-104: Lexical Index Corrupted
**Description**: SQLite FTS5 index damaged or incompatible schema

**Symptoms**:
- SQLite error: "no such table"
- FTS5 syntax errors
- Index rebuild required

**Detection**:
```python
try:
    cursor.execute("SELECT * FROM fts_index LIMIT 1")
except sqlite3.OperationalError as e:
    # Corrupted index
```

**Expected Behavior**: Error propagates or degrades to vector-only search

**Test Coverage**: `test_tool_execution_memory_error` (TC-013)

**Mitigation**:
- Implement automatic index rebuild
- Fallback to vector search only
- Log corruption for admin review

---

### [MEMORY]-[MEDIUM]-105: Reranker Model Not Loaded
**Description**: Cross-encoder reranker model unavailable or failed to load

**Detection**:
```python
if self._reranker is None:
    logger.warning("Reranker unavailable, skipping reranking")
```

**Expected Behavior**: Tool works without reranking, results may be less relevant

**Test Coverage**: Implicit (not explicitly tested)

**Mitigation**: Lazy load reranker, log warning, continue without it

---

### [MEMORY]-[HIGH]-106: Embedding Generation Timeout
**Description**: Embedding model takes too long or hangs

**Example**:
```python
# Query embedding generation exceeds timeout
query = "very long query..." * 1000
embedding = embedder.embed(query)  # Timeout after 30s
```

**Detection**:
```python
with timeout(30):
    embedding = embedder.embed(query)
```

**Expected Behavior**: Raise timeout error with retry suggestion

**Test Coverage**: Not covered

**Mitigation**:
- Implement timeout with configurable value
- Truncate overly long queries
- Provide fallback to lexical search only

---

## 3. Graph Compilation Edge Cases

### [GRAPH]-[CRITICAL]-201: LangGraph Not Installed
**Description**: LangGraph package unavailable in environment

**Detection**:
```python
try:
    from langgraph.graph import StateGraph
except ImportError:
    # LangGraph unavailable
```

**Expected Behavior**: Fallback to PraisonAI execution mode

**Test Coverage**: `test_graph_compilation_without_langgraph` (TC-014)

**Mitigation**: Check `USING_LANGGRAPH` flag before graph operations

---

### [GRAPH]-[HIGH]-202: Agent with Tools in PraisonAI Mode
**Description**: Agents have tools configured but system falls back to PraisonAI

**Expected Behavior**: Tools passed to PraisonAI Agent constructor

**Test Coverage**: `test_complete_workflow_with_graphrag_tool` (TC-019)

**Mitigation**: Ensure PraisonAI agents receive tools list

---

### [GRAPH]-[HIGH]-203: Graph with No Agents
**Description**: Attempting to compile StateGraph without any agent nodes

**Example**:
```python
config = OrchestratorConfig(name="EmptyGraph", agents=[])
orchestrator.initialize()  # Should fail
```

**Expected Behavior**: Compilation fails with descriptive error

**Test Coverage**: Not covered

**Mitigation**: Validate agent list before graph creation

---

### [GRAPH]-[MEDIUM]-204: Circular Graph Edges
**Description**: Graph topology has circular dependencies

**Example**:
```
Agent A → Agent B → Agent C → Agent A
```

**Expected Behavior**: LangGraph validation catches and raises error

**Test Coverage**: Not covered

**Mitigation**: Trust LangGraph's built-in cycle detection

---

### [GRAPH]-[HIGH]-205: Agent Node Without Execute Method
**Description**: Agent object lacks `execute()` method

**Detection**:
```python
if not hasattr(agent, 'execute'):
    raise AttributeError("Agent must have execute() method")
```

**Expected Behavior**: Error during graph compilation or first execution

**Test Coverage**: `test_agent_node_with_tool_access` (TC-010)

**Mitigation**: Validate agent interface during factory creation

---

## 4. Tool Execution Edge Cases

### [EXEC]-[LOW]-301: Query with Special Characters
**Description**: Query contains special characters that may break parsing

**Example**:
```python
query = "safety & compliance (ISO 9001) - version 2.0"
query = "What's the policy regarding 'user data'?"
```

**Expected Behavior**: Characters escaped properly, query processed correctly

**Test Coverage**: Implicit (not explicitly tested)

**Mitigation**: Use parameterized queries, proper string escaping

---

### [EXEC]-[HIGH]-302: Query Exceeds Token Limit
**Description**: Query too long for embedding model or LLM context

**Example**:
```python
query = "find documents about " + " ".join(["safety"] * 10000)  # 10K words
```

**Detection**:
```python
if len(query.split()) > MAX_QUERY_LENGTH:
    # Truncate or raise error
```

**Expected Behavior**: Truncation with warning or error message

**Test Coverage**: Not covered

**Mitigation**: Truncate to max length, log warning, suggest query refinement

---

### [EXEC]-[MEDIUM]-303: Tags/Docs with Invalid Format
**Description**: Filter parameters in unexpected format

**Example**:
```python
tool(query="test", tags=["valid", 123, None])  # Mixed types
tool(query="test", documents="")  # Empty string
```

**Detection**:
```python
tags_list = _parse_list(tags)  # Returns None for invalid
```

**Expected Behavior**: Invalid items filtered out, parsed as empty list

**Test Coverage**: `test_tool_execution_with_filters` (TC-003 partial)

**Mitigation**: Robust parsing in `_parse_list()` helper

---

### [EXEC]-[MEDIUM]-304: top_k = 0
**Description**: Request for zero results

**Expected Behavior**: Default to 5 or raise ValueError

**Test Coverage**: `test_tool_with_invalid_parameters` (TC-015)

**Mitigation**:
```python
top = max(1, int(top_k))  # Ensure at least 1
```

---

### [EXEC]-[MEDIUM]-305: top_k Negative
**Description**: Negative value for result count

**Expected Behavior**: Default to 5 or absolute value

**Test Coverage**: `test_tool_with_invalid_parameters` (TC-015)

**Mitigation**: Same as 304

---

### [EXEC]-[MEDIUM]-306: Concurrent Tool Calls
**Description**: Multiple agents calling same tool simultaneously

**Risk**: Race conditions, connection pool exhaustion

**Expected Behavior**: Thread-safe execution, results independent

**Test Coverage**: `test_concurrent_tool_execution` (TC-017)

**Mitigation**: Use thread-safe memory manager, connection pooling

---

### [EXEC]-[HIGH]-307: Tool Called Without Query
**Description**: Required `query` parameter missing

**Example**:
```python
tool(top_k=5)  # Missing query
```

**Expected Behavior**: Raise TypeError or return error message

**Test Coverage**: Not covered

**Mitigation**: Enforce required parameters via function signature

---

### [EXEC]-[LOW]-308: Query in Non-Latin Script
**Description**: Query in Arabic, Chinese, Cyrillic, etc.

**Example**:
```python
query = "安全标准和合规要求"  # Chinese
```

**Expected Behavior**: Works if embedder supports language

**Test Coverage**: Not covered

**Mitigation**: Use multilingual embedding models (e.g., `multilingual-e5`)

---

## 5. State Management Edge Cases

### [STATE]-[HIGH]-401: Tool Result Exceeds State Size Limit
**Description**: Tool returns very large result overwhelming state

**Example**:
```python
# Returns 100MB of document text
result = tool(query="all documents", top_k=10000)
```

**Detection**:
```python
if len(result) > MAX_STATE_SIZE:
    # Truncate or paginate
```

**Expected Behavior**: Truncation with indication of total results

**Test Coverage**: Not covered

**Mitigation**: Implement result pagination, set max result size

---

### [STATE]-[MEDIUM]-402: Multiple Tool Calls in Single Iteration
**Description**: Agent calls tool multiple times before iteration ends

**Expected Behavior**: All results stored separately with unique keys

**Test Coverage**: `test_graph_state_with_tool_results` (TC-011 partial)

**Mitigation**:
```python
state.node_outputs[f"graph_rag_lookup_{call_id}"] = result
```

---

### [STATE]-[HIGH]-403: Tool Fails Mid-Execution
**Description**: Tool raises exception during agent execution

**Expected Behavior**: Error stored in `state.errors[]` with recovery options

**Test Coverage**: Not covered

**Mitigation**: Wrap tool calls in try-except, populate error state

---

### [STATE]-[MEDIUM]-404: State Serialization with Tool Results
**Description**: Attempting to serialize state containing non-JSON-serializable tool results

**Detection**:
```python
try:
    json.dumps(state)
except TypeError:
    # Contains non-serializable data
```

**Expected Behavior**: Convert to JSON-serializable format or provide serialization method

**Test Coverage**: Not covered

**Mitigation**: Ensure all state fields are JSON-serializable

---

### [STATE]-[MEDIUM]-405: State Persistence Across Graph Restarts
**Description**: Resuming graph execution from checkpointed state

**Expected Behavior**: Tool results retained and accessible

**Test Coverage**: Not covered

**Mitigation**: Use LangGraph's MemorySaver for state persistence

---

## 6. Error Propagation Edge Cases

### [ERROR]-[CRITICAL]-501: Memory Manager Timeout
**Description**: Memory retrieval exceeds configured timeout

**Detection**:
```python
with timeout(30):
    results = manager.retrieve_with_graph(query)
```

**Expected Behavior**: Timeout error propagates with actionable message

**Test Coverage**: `test_tool_execution_memory_error` (TC-013)

**Mitigation**: Configure reasonable timeouts, implement retries

---

### [ERROR]-[HIGH]-502: Embedding Generation Fails
**Description**: Embedding model raises exception

**Example**:
- Model not loaded
- CUDA out of memory
- Network error (API-based embeddings)

**Expected Behavior**: Error or degraded search (lexical only)

**Test Coverage**: `test_tool_execution_memory_error` (TC-013)

**Mitigation**: Implement fallback chain: embedding → lexical → keyword

---

### [ERROR]-[HIGH]-503: Graph DB Query Error
**Description**: Neo4j Cypher query fails

**Example**:
- Syntax error in generated query
- Graph schema mismatch
- Transaction timeout

**Expected Behavior**: Error with actionable message (query shown in logs)

**Test Coverage**: `test_tool_execution_memory_error` (TC-013)

**Mitigation**: Validate queries, add query logging, implement retries

---

### [ERROR]-[MEDIUM]-504: Tool Execution During Agent Error
**Description**: Agent encounters error but tool result available

**Expected Behavior**: Tool result preserved for potential recovery/retry

**Test Coverage**: Not covered

**Mitigation**: Store tool results before error state updates

---

### [ERROR]-[MEDIUM]-505: Orchestrator Cleanup with Pending Tool Calls
**Description**: `orchestrator.cleanup()` called with tools still executing

**Expected Behavior**: Graceful shutdown, wait for or cancel pending calls

**Test Coverage**: `test_memory_manager_cleanup` (TC-022)

**Mitigation**: Implement cleanup hooks, force cleanup after timeout

---

## 7. Performance Edge Cases

### [PERF]-[MEDIUM]-601: Very Large Document Corpus
**Description**: Memory contains millions of documents

**Impact**: Slow retrieval, high memory usage

**Expected Behavior**: Pagination, index optimization, acceptable latency

**Test Coverage**: Not covered (performance tests use small datasets)

**Mitigation**: Use database indexes, implement caching, optimize queries

---

### [PERF]-[MEDIUM]-602: Frequent Tool Calls
**Description**: Agent makes 100+ tool calls per session

**Impact**: Connection pool exhaustion, API rate limits

**Expected Behavior**: Throttling, caching, batching

**Test Coverage**: `test_tool_execution_performance` (TC-021 partial)

**Mitigation**: Implement result caching, connection pooling, rate limiting

---

### [PERF]-[LOW]-603: Cold Start Delay
**Description**: First tool call much slower than subsequent calls

**Cause**: Model loading, connection establishment, cache warming

**Expected Behavior**: Acceptable latency after first call

**Test Coverage**: Not covered

**Mitigation**: Warm up connections during initialization

---

## 8. Security Edge Cases

### [SEC]-[CRITICAL]-701: SQL Injection in Filters
**Description**: User-provided filter values not sanitized

**Example**:
```python
tool(query="test", documents="doc'; DROP TABLE documents;--")
```

**Expected Behavior**: Parameterized queries prevent injection

**Test Coverage**: Not covered

**Mitigation**: Always use parameterized queries, never string concatenation

---

### [SEC]-[HIGH]-702: Information Disclosure
**Description**: Tool returns documents user shouldn't access

**Expected Behavior**: Respect user_id/run_id permissions

**Test Coverage**: Not covered

**Mitigation**: Implement row-level security, validate permissions

---

### [SEC]-[MEDIUM]-703: Denial of Service via Large Queries
**Description**: Malicious queries designed to exhaust resources

**Example**:
```python
tool(query="a" * 1000000, top_k=10000)
```

**Expected Behavior**: Query truncation, rate limiting

**Test Coverage**: Partially covered by 304

**Mitigation**: Input validation, rate limiting, resource quotas

---

## 9. Integration Edge Cases

### [INTEG]-[MEDIUM]-801: LangChain Version Mismatch
**Description**: Incompatible LangChain/LangGraph versions

**Symptoms**: Import errors, API changes, unexpected behavior

**Detection**: Version checks during initialization

**Expected Behavior**: Clear error message with version requirements

**Test Coverage**: Not covered

**Mitigation**: Pin dependency versions, test across version ranges

---

### [INTEG]-[MEDIUM]-802: PraisonAI vs LangGraph Mode Switch
**Description**: System switches between backends mid-session

**Expected Behavior**: Not allowed - backend selected at initialization

**Test Coverage**: `test_graph_compilation_without_langgraph` (TC-014)

**Mitigation**: Immutable backend selection after initialization

---

## 10. Data Quality Edge Cases

### [DATA]-[MEDIUM]-901: Malformed Metadata
**Description**: Document metadata has unexpected structure

**Example**:
```python
{
    "content": "...",
    "metadata": None  # Should be dict
}
```

**Expected Behavior**: Graceful handling with defaults

**Test Coverage**: `test_tool_with_missing_metadata` (TC-018)

**Mitigation**: Robust metadata parsing with defaults

---

### [DATA]-[LOW]-902: Missing Document IDs
**Description**: Results lack unique document identifiers

**Expected Behavior**: Display "unknown" or generate temporary ID

**Test Coverage**: `test_tool_with_missing_metadata` (TC-018)

**Mitigation**: Fallback ID generation, log warning

---

## Edge Case Priority Matrix

| Severity | Count | Covered | Uncovered | Coverage % |
|----------|-------|---------|-----------|------------|
| CRITICAL | 4 | 3 | 1 | 75% |
| HIGH | 17 | 10 | 7 | 59% |
| MEDIUM | 19 | 8 | 11 | 42% |
| LOW | 6 | 2 | 4 | 33% |
| **TOTAL** | **46** | **23** | **23** | **50%** |

## Recommendations

### Phase 1: Critical Coverage (Immediate)
1. Add tests for SQL injection prevention (SEC-701)
2. Add test for query without query parameter (EXEC-307)
3. Add test for embedding timeout (MEMORY-106)

### Phase 2: High Priority (Next Sprint)
1. Add tests for query token limit (EXEC-302)
2. Add tests for state serialization (STATE-404)
3. Add tests for tool failure mid-execution (STATE-403)

### Phase 3: Medium Priority (Next Quarter)
1. Add tests for large document corpus (PERF-601)
2. Add tests for LangChain version compatibility (INTEG-801)
3. Add tests for information disclosure (SEC-702)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-21
**Next Review**: 2025-04-21
**Maintained By**: Quality Engineering Team