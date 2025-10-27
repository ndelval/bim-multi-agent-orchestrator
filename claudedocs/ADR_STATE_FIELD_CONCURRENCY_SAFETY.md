# ADR: State Field Concurrency Safety Requirements

**Status**: ACCEPTED  
**Date**: 2025-10-04  
**Decision Makers**: Architecture Team  
**Context**: LangGraph StateGraph Implementation for Multi-Agent Orchestration

---

## Context and Problem Statement

LangGraph StateGraphs enable **structural parallelism** where multiple nodes can execute concurrently in the same step when graph topology permits. This execution model requires careful state field design to prevent concurrent write conflicts.

**Problem**: The `InvalidUpdateError: Can receive only one value per step` error occurred when parallel nodes attempted to write to the `current_iteration` field, which lacked proper concurrent write handling (Annotated reducer).

**Root Cause**: State schema was designed with sequential execution assumptions, not adapted for LangGraph's structural parallelism model.

---

## Decision Drivers

1. **Framework Semantics**: LangGraph's execution model (parallel by default when topology allows)
2. **Type Safety**: Python's type system cannot validate reducer requirements statically
3. **Developer Experience**: Clear, enforceable guidelines to prevent future bugs
4. **Maintainability**: Self-documenting state schema with explicit safety guarantees
5. **Error Prevention**: Proactive detection of concurrency issues before runtime

---

## Decision

We establish **mandatory concurrency safety requirements** for all LangGraph StateGraph state fields:

### Rule 1: Multi-Writer Fields MUST Have Reducers

**Requirement**:
```python
# ❌ INVALID - No reducer for multi-writer field
field_name: Dict[str, Any] = field(default_factory=dict)

# ✅ VALID - Annotated reducer for concurrent writes
field_name: Annotated[Dict[str, Any], merge_dicts] = field(default_factory=dict)
```

**Applies To**:
- Any field written by 2+ nodes
- Collection types (List, Dict) where multiple nodes append/add
- Aggregation fields that combine results from parallel branches

### Rule 2: Scalar Fields MUST Be Safe

**Requirement**: Scalar types (int, str, bool, Optional[T]) MUST be:
- **Read-only** (configuration constants, never written by nodes) OR
- **Proven single-writer** via graph topology analysis OR  
- **Use custom Annotated reducer** if multiple writers exist

**Examples**:
```python
# ✅ SAFE - Configuration constant
max_iterations: int = 10

# ✅ SAFE - Proven single-writer (only router writes)
current_route: Optional[str] = None

# ❌ UNSAFE - Multiple nodes could write this
current_iteration: int = 0  # No reducer, parallel writes fail
```

### Rule 3: Semantic Compatibility Required

**Requirement**: Field concepts MUST be semantically compatible with parallel execution

**Invalid Concepts**:
- "Iteration counter" (ambiguous: is 3 parallel nodes = 1 iteration or 3?)
- "Current step" (undefined when nodes execute concurrently)
- Any concept assuming linear sequential progression

**Valid Concepts**:
- "Execution depth" (clear: number of nodes in execution path)
- "Completed count" (clear: number of finished agents)
- Topology-aware metrics derived from graph state

### Rule 4: Mandatory Documentation

**Requirement**: EVERY state field MUST have inline documentation specifying concurrency safety

**Format**:
```python
# SAFE: <justification for single-writer or read-only>
field_name: Type = value

# PARALLEL-SAFE: <reducer type and concurrent write behavior>
field_name: Annotated[Type, reducer] = field(...)
```

**Example**:
```python
@dataclass
class MyState:
    # SAFE: configuration constant, never written
    max_retries: int = 3
    
    # SAFE: only router node writes (single-writer)
    route: Optional[str] = None
    
    # PARALLEL-SAFE: reducer merges concurrent writes
    outputs: Annotated[Dict[str, str], merge_dicts] = field(default_factory=dict)
```

---

## Compliance Mechanisms

### 1. Pre-Merge Checklist

**Before adding ANY new state field**:

- [ ] Field written by multiple nodes? → REQUIRES Annotated reducer
- [ ] Field semantically compatible with parallel execution? → No ambiguous concepts
- [ ] Field read-only or proven single-writer? → Document with `# SAFE:` comment
- [ ] Added inline concurrency safety documentation? → Required for ALL fields
- [ ] Tested with parallel node execution? → Required validation
- [ ] Updated this ADR if new patterns emerge? → Keep guidelines current

### 2. Code Review Requirements

**Reviewers MUST verify**:

- [ ] All new fields have concurrency safety comments (`# SAFE:` or `# PARALLEL-SAFE:`)
- [ ] Multi-writer fields have Annotated reducers (merge_dicts, merge_lists, custom)
- [ ] Single-writer claims validated via graph topology analysis
- [ ] No "iteration counter" or similar parallel-incompatible concepts
- [ ] Parallel execution test coverage exists

### 3. Testing Requirements

**Mandatory Test Coverage**:
- Unit tests for new fields with parallel write scenarios
- Integration tests with 3+ parallel nodes writing to same field
- Stress tests for high-concurrency scenarios (10+ parallel nodes)

---

## Reducer Types Reference

### Built-In Reducers

**`merge_dicts`** - Dictionary merge (for Dict types):
```python
# Behavior: Merge keys from all concurrent writes
outputs: Annotated[Dict[str, str], merge_dicts] = field(default_factory=dict)

# Example: 3 parallel nodes write different keys → all merged
# Node A: {"a": "1"}, Node B: {"b": "2"}, Node C: {"c": "3"}
# Result: {"a": "1", "b": "2", "c": "3"}
```

**`merge_lists`** - List concatenation (for List types):
```python
# Behavior: Concatenate all lists from concurrent writes
completed: Annotated[List[str], merge_lists] = field(default_factory=list)

# Example: 3 parallel nodes append items → all concatenated
# Node A: ["a"], Node B: ["b"], Node C: ["c"]
# Result: ["a", "b", "c"]
```

**`add_messages`** - Message list (for BaseMessage types):
```python
# Behavior: Append messages with deduplication
messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
```

### Custom Reducers

**When to Create**:
- Domain-specific merge logic needed
- Complex aggregation beyond dict/list merge
- Conflict resolution rules (e.g., max value, priority-based)

**Example**:
```python
def merge_metrics(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Custom reducer: take max value for each metric."""
    result = a.copy()
    for key, value in b.items():
        result[key] = max(result.get(key, 0), value)
    return result

metrics: Annotated[Dict[str, float], merge_metrics] = field(default_factory=dict)
```

---

## Consequences

### Positive

1. **Runtime Safety**: Prevents `InvalidUpdateError` from concurrent writes
2. **Self-Documenting Code**: Concurrency safety explicit in state schema
3. **Developer Guidance**: Clear rules eliminate ambiguity in field design
4. **Maintainability**: Future developers understand concurrency semantics
5. **Architectural Alignment**: State schema matches LangGraph execution model

### Negative

1. **Initial Complexity**: Developers must understand reducer concepts upfront
2. **Documentation Overhead**: Every field requires safety comment
3. **Review Burden**: Reviewers must validate concurrency safety claims
4. **Migration Effort**: Existing code may need updates to comply

### Mitigation Strategies

- **Training**: Onboard developers on LangGraph execution model
- **Templates**: Provide state schema templates with examples
- **Tooling**: Create lint rules to auto-detect violations
- **Examples**: Maintain reference implementations demonstrating patterns

---

## Examples and Anti-Patterns

### ✅ Good Examples

**Collection with Reducer**:
```python
# Multiple agents write results → reducer merges
agent_outputs: Annotated[Dict[str, str], merge_dicts] = field(default_factory=dict)  # PARALLEL-SAFE
```

**Proven Single-Writer**:
```python
# Only router node writes route decision → safe
current_route: Optional[str] = None  # SAFE: only router writes (single-writer)
```

**Read-Only Configuration**:
```python
# Configuration set at init, never modified → safe
max_iterations: int = 10  # SAFE: configuration constant, never written
```

**Derived Property Instead of Counter**:
```python
@property
def execution_depth(self) -> int:
    """Derived from execution_path, no concurrent write issues."""
    return len(self.execution_path)  # SAFE: computed, not stored
```

### ❌ Anti-Patterns

**Scalar Without Reducer**:
```python
# ❌ WRONG - Multiple nodes increment → concurrent write error
current_iteration: int = 0  # No reducer, parallel nodes fail
```

**Ambiguous Parallel Concept**:
```python
# ❌ WRONG - "Step" is undefined when 3 nodes execute in parallel
current_step: int = 0  # Semantic incompatibility
```

**Collection Without Reducer**:
```python
# ❌ WRONG - Multiple nodes add items → concurrent write error
results: List[str] = field(default_factory=list)  # Missing merge_lists
```

**Undocumented Safety**:
```python
# ❌ WRONG - No concurrency safety documentation
status: str = "pending"  # Is this safe? Unknown.
```

---

## Migration Guide

### For Existing `current_iteration` References

**Replace with derived properties**:

```python
# OLD (removed):
state.current_iteration += 1
if state.current_iteration >= state.max_iterations:
    raise MaxIterationsError()

# NEW:
# Iteration tracking happens via execution_path (has reducer)
# Use derived property:
if state.execution_depth >= state.max_iterations:
    raise MaxIterationsError()
```

### For New State Fields

**Decision Tree**:

```
Adding new field?
│
├─ Written by multiple nodes?
│  ├─ YES → Requires Annotated reducer
│  │        ├─ Dict type? → Use merge_dicts
│  │        ├─ List type? → Use merge_lists
│  │        └─ Other? → Create custom reducer
│  │
│  └─ NO → Is it read-only OR proven single-writer?
│           ├─ Read-only → Document as "SAFE: configuration/never written"
│           ├─ Single-writer → Document as "SAFE: only X node writes"
│           └─ Uncertain → Default to Annotated reducer (safe choice)
│
└─ Add concurrency safety comment (MANDATORY)
```

---

## Enforcement and Monitoring

### Static Analysis (Future Work)

**Proposed Lint Rule**:
```python
# Detect: StateGraph state classes with non-annotated collections
# Pattern: field_name: (Dict|List)[...] = field(...) without Annotated
# Action: Raise error requiring Annotated reducer or safety justification
```

### Runtime Monitoring

**LangGraph provides**:
- `InvalidUpdateError` when concurrent writes occur without reducer
- Error message explicitly requests Annotated reducer

**We add**:
- Pre-deployment validation: Test all routes with parallel execution
- Continuous monitoring: Track concurrent write patterns in production

---

## Related Decisions

- **Migration from PraisonAI to LangGraph**: ADR-001 (if exists)
- **Multi-Agent Orchestration Architecture**: ADR-002 (if exists)
- **Graph-Based Workflow Design**: ADR-003 (if exists)

---

## References

1. **LangGraph Documentation**: State Management and Reducers
2. **Root Cause Analysis**: `claudedocs/CURRENT_ITERATION_CONCURRENT_WRITE_ROOT_CAUSE.md`
3. **Implementation Report**: `claudedocs/CONCURRENT_STATE_FIX_IMPLEMENTATION.md`
4. **LangGraph Channels**: `langgraph/channels/last_value.py` - LastValue channel rejection logic

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-10-04 | Initial ADR created from bug fix learnings | Architecture Team |

---

**Decision Status**: ✅ ACCEPTED  
**Compliance**: MANDATORY for all StateGraph implementations  
**Review Cycle**: Quarterly (or when new patterns emerge)
