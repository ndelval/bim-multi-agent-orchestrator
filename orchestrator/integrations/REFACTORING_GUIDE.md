# OrchestratorState Refactoring Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the current `OrchestratorState` implementation to the refactored version with improved type safety, validation, and maintainability.

**Reference Implementation**: See `langchain_state_refactored.py` for the complete refactored code.

---

## Migration Strategy

### Phase 1: Backward-Compatible Additions (Safe)
Add new fields and helper methods without changing existing behavior.

### Phase 2: Type Safety Improvements (Low Risk)
Update type annotations and add TypedDict definitions.

### Phase 3: Validation Integration (Medium Risk)
Add validation logic with optional disable flag for gradual rollout.

### Phase 4: Replace Direct Access (Refactoring)
Update client code to use helper methods.

---

## Phase 1: Add Missing Fields (Priority: HIGH, Risk: LOW)

### Changes Required

**File**: `orchestrator/integrations/langchain_integration.py`

**Location**: Lines 64-104 (OrchestratorState class)

### Step 1.1: Add Graph Execution Tracking Fields

```python
# Add after line 86 (after completed_agents field)

# Graph Execution Tracking
current_node: Optional[str] = None
execution_path: List[str] = field(default_factory=list)
node_outputs: Dict[str, str] = field(default_factory=dict)
condition_results: Dict[str, bool] = field(default_factory=dict)
parallel_execution_active: bool = False
```

**Rationale**: These fields are used 7-8+ times in `graph_compiler.py` but missing from schema.

**Testing**: Run existing tests - should pass with no changes since fields now have explicit defaults.

```bash
python orchestrator/planning/test_tot_graph_integration.py
python orchestrator/planning/test_structure_only.py
```

### Step 1.2: Fix Mutable Default Pattern

**Before**:
```python
agent_outputs: Dict[str, Any] = None
completed_agents: List[str] = None
recall_items: List[str] = None

def __post_init__(self):
    if self.agent_outputs is None:
        self.agent_outputs = {}
    if self.completed_agents is None:
        self.completed_agents = []
    if self.recall_items is None:
        self.recall_items = []
```

**After**:
```python
from dataclasses import field

agent_outputs: Dict[str, str] = field(default_factory=dict)  # Note: str not Any
completed_agents: List[str] = field(default_factory=list)
recall_items: List[str] = field(default_factory=list)

# Remove __post_init__ entirely (or keep empty for Phase 3)
def __post_init__(self):
    pass  # Keep for Phase 3 validation logic
```

**Testing**: All existing code should work identically.

---

## Phase 2: Type Safety Improvements (Priority: HIGH, Risk: LOW)

### Step 2.1: Add TypedDict Definitions

**File**: `orchestrator/integrations/langchain_integration.py`

**Location**: Add before `OrchestratorState` class (around line 60)

```python
from typing_extensions import TypedDict
from typing import Literal

class RouterDecision(TypedDict, total=False):
    """Structured router decision output."""
    route: Literal["quick", "research", "analysis", "standards"]
    confidence: float
    reasoning: str
    assigned_agents: List[str]

class Assignment(TypedDict):
    """Task assignment structure."""
    agent_name: str
    task_description: str
    dependencies: List[str]
    priority: int
```

**Rationale**: Provides type safety for structured data without runtime overhead.

### Step 2.2: Update Type Annotations

**Before**:
```python
current_route: Optional[str] = None
router_decision: Optional[Dict[str, Any]] = None
assignments: Optional[List[Dict[str, Any]]] = None
agent_outputs: Dict[str, Any] = field(default_factory=dict)
```

**After**:
```python
current_route: Optional[Literal["quick", "research", "analysis", "standards"]] = None
router_decision: Optional[RouterDecision] = None
assignments: List[Assignment] = field(default_factory=list)
agent_outputs: Dict[str, str] = field(default_factory=dict)
```

**Testing**:
- Run type checker: `mypy orchestrator/integrations/langchain_integration.py`
- Verify no type errors in client code

---

## Phase 3: Add Structured Error Handling (Priority: MEDIUM, Risk: MEDIUM)

### Step 3.1: Create ExecutionError Class

**File**: `orchestrator/integrations/langchain_integration.py`

**Location**: Add after TypedDict definitions (around line 80)

```python
@dataclass(frozen=True)
class ExecutionError:
    """Structured error information for workflow failures."""
    error_type: Type[Exception]
    error_message: str
    node_name: Optional[str] = None
    agent_name: Optional[str] = None
    stack_trace: Optional[str] = None
    is_recoverable: bool = False
    recovery_hint: Optional[str] = None

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        is_recoverable: bool = False
    ) -> "ExecutionError":
        """Create ExecutionError from caught exception."""
        import traceback
        return cls(
            error_type=type(exc),
            error_message=str(exc),
            node_name=node_name,
            agent_name=agent_name,
            stack_trace=traceback.format_exc(),
            is_recoverable=is_recoverable,
            recovery_hint=cls._suggest_recovery(exc)
        )

    @staticmethod
    def _suggest_recovery(exc: Exception) -> Optional[str]:
        """Suggest recovery actions based on error type."""
        from orchestrator.core.exceptions import TaskExecutionError, MemoryError, ValidationError

        if isinstance(exc, TaskExecutionError):
            return "Retry with modified task parameters or reduced scope"
        elif isinstance(exc, MemoryError):
            return "Check memory provider connectivity and configuration"
        elif isinstance(exc, ValidationError):
            return "Review input parameters and state consistency"
        return "Review error details and check system logs"
```

### Step 3.2: Update Error Field Type

**Before**:
```python
error_state: Optional[str] = None
```

**After**:
```python
error_state: Optional[ExecutionError] = None
```

### Step 3.3: Update Error Handling in graph_compiler.py (BREAKING CHANGE)

**File**: `orchestrator/planning/graph_compiler.py`

**Before** (line 242-246):
```python
except Exception as e:
    error_msg = f"Agent {agent_name} failed in node {node_spec.name}: {str(e)}"
    state.error_state = error_msg
    state.node_outputs[node_spec.name] = error_msg
    logger.error(error_msg)
    return state
```

**After**:
```python
except Exception as e:
    state.error_state = ExecutionError.from_exception(
        e,
        node_name=node_spec.name,
        agent_name=agent_name,
        is_recoverable=isinstance(e, TaskExecutionError)
    )
    state.node_outputs[node_spec.name] = state.error_state.error_message
    logger.error(f"Execution failed: {state.error_state}")
    return state
```

**Testing**:
- Trigger error conditions in tests
- Verify `ExecutionError` captures stack traces correctly
- Test error recovery logic with `is_recoverable` flag

---

## Phase 4: Add Validation Logic (Priority: MEDIUM, Risk: MEDIUM)

### Step 4.1: Implement Validation Methods

**File**: `orchestrator/integrations/langchain_integration.py`

**Location**: Inside `OrchestratorState` class, in `__post_init__` method

```python
def __post_init__(self):
    """Validate state consistency after initialization."""
    # Make validation optional during migration
    if not getattr(self, '_skip_validation', False):
        self._validate_iteration_bounds()
        self._validate_route_consistency()
        self._validate_required_fields()

def _validate_iteration_bounds(self) -> None:
    """Ensure iteration counts are within valid bounds."""
    if self.current_iteration < 0:
        raise ValidationError(
            f"current_iteration cannot be negative: {self.current_iteration}"
        )

    if self.current_iteration > self.max_iterations:
        raise ValidationError(
            f"current_iteration ({self.current_iteration}) exceeds "
            f"max_iterations ({self.max_iterations})"
        )

    if self.max_iterations <= 0:
        raise ValidationError(
            f"max_iterations must be positive: {self.max_iterations}"
        )

def _validate_route_consistency(self) -> None:
    """Ensure routing state is consistent."""
    if self.current_route and not self.router_decision:
        logger.warning(
            f"current_route set to '{self.current_route}' without router_decision"
        )

    if self.router_decision:
        decision_route = self.router_decision.get("route")
        if decision_route != self.current_route:
            raise ValidationError(
                f"Inconsistent routing: current_route='{self.current_route}', "
                f"router_decision route='{decision_route}'"
            )

def _validate_required_fields(self) -> None:
    """Validate required fields are present and non-empty."""
    if not self.messages:
        raise ValidationError("messages list cannot be empty")

    if not self.user_prompt or not self.user_prompt.strip():
        raise ValidationError("user_prompt is required and cannot be empty")
```

**Testing Strategy**:

1. **Test Valid States** (should pass):
```python
# test_state_validation.py
from orchestrator.integrations.langchain_integration import OrchestratorState
from langchain_core.messages import HumanMessage

def test_valid_state_creation():
    state = OrchestratorState(
        messages=[HumanMessage(content="test")],
        user_prompt="test query",
        max_iterations=5,
        current_iteration=0
    )
    assert state.current_iteration == 0
    assert state.max_iterations == 5
```

2. **Test Invalid States** (should raise ValidationError):
```python
import pytest
from orchestrator.core.exceptions import ValidationError

def test_negative_iteration_rejected():
    with pytest.raises(ValidationError, match="cannot be negative"):
        OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test",
            current_iteration=-1
        )

def test_empty_messages_rejected():
    with pytest.raises(ValidationError, match="messages list cannot be empty"):
        OrchestratorState(
            messages=[],
            user_prompt="test"
        )
```

3. **Test Gradual Rollout** (skip validation during migration):
```python
def test_skip_validation_during_migration():
    state = OrchestratorState(
        messages=[],  # Would normally fail
        user_prompt="",
    )
    state._skip_validation = True
    # Should not raise during migration period
```

---

## Phase 5: Add Helper Methods (Priority: LOW, Risk: LOW)

### Step 5.1: Add State Management Helpers

**File**: `orchestrator/integrations/langchain_integration.py`

**Location**: Inside `OrchestratorState` class (after validation methods)

```python
def record_node_execution(
    self,
    node_name: str,
    output: str,
    update_current: bool = True
) -> None:
    """Record execution of a graph node."""
    if update_current:
        self.current_node = node_name
    self.execution_path.append(node_name)
    self.node_outputs[node_name] = output
    logger.debug(f"Recorded execution: {node_name}")

def record_agent_completion(
    self,
    agent_name: str,
    result: str
) -> None:
    """Record successful agent execution."""
    self.agent_outputs[agent_name] = result
    self.completed_agents.append(agent_name)
    self.current_iteration += 1
    logger.info(f"Agent {agent_name} completed (iteration {self.current_iteration})")

def is_iteration_limit_reached(self) -> bool:
    """Check if maximum iterations reached."""
    return self.current_iteration >= self.max_iterations

def has_error(self) -> bool:
    """Check if workflow is in error state."""
    return self.error_state is not None

def get_last_agent_output(self) -> Optional[str]:
    """Get most recent agent output."""
    if not self.completed_agents:
        return None
    last_agent = self.completed_agents[-1]
    return self.agent_outputs.get(last_agent)
```

### Step 5.2: Refactor graph_compiler.py to Use Helpers

**Before** (lines 220-232):
```python
# Update execution state
state.current_node = node_spec.name
state.execution_path.append(node_spec.name)

# ... execution logic ...

# Update state with results
state.agent_outputs[agent_name] = result
state.node_outputs[node_spec.name] = result
state.completed_agents.append(agent_name)
state.current_iteration += 1
```

**After**:
```python
# Update execution state
state.record_node_execution(node_spec.name, "Agent execution started")

# ... execution logic ...

# Update state with results
state.record_agent_completion(agent_name, result)
state.node_outputs[node_spec.name] = result  # Still track node output separately
```

**Benefits**:
- ✅ Eliminates repeated 4-5 line patterns
- ✅ Ensures consistent state updates
- ✅ Centralizes logging
- ✅ Easier to maintain

---

## Testing Strategy

### Unit Tests (orchestrator/tests/test_orchestrator_state.py)

```python
import pytest
from orchestrator.integrations.langchain_integration import (
    OrchestratorState,
    ExecutionError,
    RouterDecision,
    Assignment
)
from orchestrator.core.exceptions import ValidationError
from langchain_core.messages import HumanMessage


class TestOrchestratorStateValidation:
    """Test state validation logic."""

    def test_valid_state_creation(self):
        """Test creating a valid state succeeds."""
        state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test query"
        )
        assert state.current_iteration == 0
        assert state.max_iterations == 10

    def test_negative_iteration_rejected(self):
        """Test negative iteration count is rejected."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            OrchestratorState(
                messages=[HumanMessage(content="test")],
                user_prompt="test",
                current_iteration=-1
            )

    def test_iteration_exceeds_max_rejected(self):
        """Test iteration > max_iterations is rejected."""
        with pytest.raises(ValidationError, match="exceeds max_iterations"):
            OrchestratorState(
                messages=[HumanMessage(content="test")],
                user_prompt="test",
                current_iteration=15,
                max_iterations=10
            )

    def test_empty_messages_rejected(self):
        """Test empty messages list is rejected."""
        with pytest.raises(ValidationError, match="messages list cannot be empty"):
            OrchestratorState(messages=[], user_prompt="test")

    def test_empty_prompt_rejected(self):
        """Test empty user_prompt is rejected."""
        with pytest.raises(ValidationError, match="user_prompt is required"):
            OrchestratorState(
                messages=[HumanMessage(content="test")],
                user_prompt=""
            )


class TestOrchestratorStateHelpers:
    """Test helper methods."""

    def test_record_node_execution(self):
        """Test recording node execution updates state correctly."""
        state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test"
        )

        state.record_node_execution("router", "Routing decision made")

        assert state.current_node == "router"
        assert "router" in state.execution_path
        assert state.node_outputs["router"] == "Routing decision made"

    def test_record_agent_completion(self):
        """Test recording agent completion increments iteration."""
        state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test"
        )

        state.record_agent_completion("researcher", "Research complete")

        assert state.current_iteration == 1
        assert "researcher" in state.completed_agents
        assert state.agent_outputs["researcher"] == "Research complete"

    def test_iteration_limit_check(self):
        """Test iteration limit detection."""
        state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test",
            max_iterations=3,
            current_iteration=3
        )

        assert state.is_iteration_limit_reached()

    def test_get_last_agent_output(self):
        """Test retrieving last agent output."""
        state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test"
        )

        state.record_agent_completion("agent1", "result1")
        state.record_agent_completion("agent2", "result2")

        assert state.get_last_agent_output() == "result2"


class TestExecutionError:
    """Test ExecutionError structured error handling."""

    def test_from_exception_captures_context(self):
        """Test creating ExecutionError from exception."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            error = ExecutionError.from_exception(
                e,
                node_name="test_node",
                agent_name="test_agent"
            )

        assert error.error_type == ValueError
        assert error.error_message == "Test error"
        assert error.node_name == "test_node"
        assert error.agent_name == "test_agent"
        assert error.stack_trace is not None

    def test_recovery_hint_for_task_error(self):
        """Test recovery hints are generated correctly."""
        from orchestrator.core.exceptions import TaskExecutionError

        try:
            raise TaskExecutionError("Task failed")
        except Exception as e:
            error = ExecutionError.from_exception(e)

        assert "Retry" in error.recovery_hint
        assert error.is_recoverable is False  # Default unless specified
```

### Integration Tests (orchestrator/tests/test_graph_compiler_integration.py)

```python
import pytest
from orchestrator.planning.graph_compiler import GraphCompiler
from orchestrator.planning.graph_specifications import WorkflowSpec, NodeSpec, NodeType
from orchestrator.integrations.langchain_integration import OrchestratorState
from langchain_core.messages import HumanMessage


class TestGraphCompilerWithRefactoredState:
    """Test GraphCompiler works with refactored OrchestratorState."""

    def test_simple_workflow_execution(self):
        """Test executing a simple workflow updates state correctly."""
        # Create workflow spec
        spec = WorkflowSpec(
            name="test_workflow",
            nodes=[
                NodeSpec(name="start", type=NodeType.START),
                NodeSpec(name="agent1", type=NodeType.AGENT, agent_name="researcher"),
                NodeSpec(name="end", type=NodeType.END)
            ],
            edges=[
                ("start", "agent1"),
                ("agent1", "end")
            ]
        )

        # Create compiler and initial state
        compiler = GraphCompiler(spec)
        initial_state = OrchestratorState(
            messages=[HumanMessage(content="test query")],
            user_prompt="test query"
        )

        # Execute workflow
        graph = compiler.compile()
        result = graph.invoke(initial_state)

        # Verify state updates
        assert "start" in result.execution_path
        assert "agent1" in result.execution_path
        assert "end" in result.execution_path
        assert result.current_iteration >= 1
        assert not result.has_error()

    def test_error_handling_creates_execution_error(self):
        """Test errors create structured ExecutionError objects."""
        # Create workflow that will fail
        spec = WorkflowSpec(
            name="failing_workflow",
            nodes=[
                NodeSpec(name="start", type=NodeType.START),
                NodeSpec(name="bad_agent", type=NodeType.AGENT, agent_name="nonexistent"),
                NodeSpec(name="end", type=NodeType.END)
            ],
            edges=[("start", "bad_agent"), ("bad_agent", "end")]
        )

        compiler = GraphCompiler(spec)
        initial_state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            user_prompt="test"
        )

        graph = compiler.compile()
        result = graph.invoke(initial_state)

        # Verify structured error
        assert result.has_error()
        assert result.error_state.node_name == "bad_agent"
        assert result.error_state.stack_trace is not None
        assert result.error_state.recovery_hint is not None
```

---

## Rollback Plan

If issues arise during migration:

### Quick Rollback (Phase 1-2)
1. Revert file: `git checkout HEAD -- orchestrator/integrations/langchain_integration.py`
2. No client code changes needed

### Partial Rollback (Phase 3-4)
1. Keep new fields but revert to string error handling:
   ```python
   error_state: Optional[str] = None  # Temporarily revert
   ```
2. Comment out validation in `__post_init__`:
   ```python
   def __post_init__(self):
       pass  # Temporarily disable validation
   ```

### Full Migration Testing
Before deploying to production:
1. Run all existing tests: `pytest orchestrator/`
2. Run new validation tests: `pytest orchestrator/tests/test_orchestrator_state.py`
3. Integration test with real workflows: `python orchestrator/cli/main.py chat --test-mode`

---

## Expected Benefits After Migration

### Immediate Benefits (Phase 1-2)
- ✅ Type safety catches bugs at development time
- ✅ No more AttributeErrors from missing fields
- ✅ Consistent mutable default handling

### Medium-Term Benefits (Phase 3-4)
- ✅ Structured error handling enables automated recovery
- ✅ Validation catches invalid states early
- ✅ Better debugging with complete error context

### Long-Term Benefits (Phase 5)
- ✅ Cleaner client code (30% less boilerplate)
- ✅ Easier maintenance (centralized state operations)
- ✅ Self-documenting helper methods

---

## Timeline Recommendation

- **Week 1**: Phase 1 (Add fields, fix defaults) - Deploy to dev
- **Week 2**: Phase 2 (Type safety) - Deploy to staging
- **Week 3**: Phase 3 (Error handling) - Deploy to staging with monitoring
- **Week 4**: Phase 4 (Validation) - Deploy to production
- **Week 5**: Phase 5 (Helpers) - Gradual client code refactoring

**Total Migration Time**: 5 weeks with conservative testing between phases

---

## Questions or Issues?

Contact the refactoring team or review:
- Reference implementation: `orchestrator/integrations/langchain_state_refactored.py`
- Test examples: `orchestrator/tests/test_orchestrator_state.py`
- This guide: `orchestrator/integrations/REFACTORING_GUIDE.md`