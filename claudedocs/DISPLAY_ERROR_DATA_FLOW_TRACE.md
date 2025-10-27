# Complete Data Flow Trace: Display AttributeError

**Purpose**: Detailed step-by-step trace of data transformation from workflow completion to crash

---

## Scenario: Multi-Agent Router Workflow

**User Query**: "Who is the best Formula 1 team and driver?"
**Workflow**: Router → Researcher → Analyst → Quality Assurance
**Expected**: Display quality assurance report
**Actual**: AttributeError crash

---

## Step-by-Step Data Flow

### Step 1: Workflow Execution Success

**Location**: `graph_adapter.py:run_multi_agent_workflow()`

```python
# Lines 492-560: Workflow execution
result = app.invoke(
    initial_state,
    config={"recursion_limit": 50}
)
```

**State at completion**:
```python
result = StateSnapshot(
    values={
        'messages': [...],
        'agent_outputs': {
            'router': "...",
            'research_best_team': "...",
            'research_f1_teams': "...",
            'analyze_best_driver': "...",
            'quality_assurance': "2592 chars report"
        },
        'current_route': 'analysis',
        'router_decision': {...}
    },
    next=['end'],
    ...
)
```

**Log Output**:
```
✓ Agent quality_assurance completed
✓ Workflow completed at node: end
```

---

### Step 2: State Extraction

**Location**: `graph_adapter.py:562`

```python
final_output = self._extract_result_from_state(result)
```

**Function Call**: `_extract_result_from_state(state=result)`

**Processing** (lines 585-676):

1. **Routing Metadata Extraction** (lines 620-630):
```python
routing_metadata = {}
if 'current_route' in state:
    routing_metadata['decision'] = state['current_route']  # 'analysis'
if 'router_decision' in state:
    routing_metadata['router_decision'] = state['router_decision']  # {...}
```

2. **Text Extraction** (lines 632-654):
```python
raw_output = state.get('agent_outputs')['quality_assurance']
# raw_output = "2592 character quality assurance report text..."
```

3. **Text Cleaning** (lines 655-666):
```python
clean_output = re.sub(r'^\*\*\w+\*\*:\s*', '', raw_output.strip())
# Removes agent name prefix
# clean_output = "Quality assurance report text..."
```

4. **Return Decision** (lines 669-676):
```python
if routing_metadata:  # True! We have routing metadata
    return {
        'output': clean_output,  # "Quality assurance report..."
        'current_route': 'analysis',
        'router_decision': {...}
    }
```

**Result**:
```python
final_output = {
    'output': "Quality Assurance Report\n\n**Final Summary**\n...[2592 chars]",
    'current_route': 'analysis',
    'router_decision': {...}
}
```

**Type**: `Dict[str, Any]`

---

### Step 3: Display Call (THE BUG)

**Location**: `graph_adapter.py:565-566`

```python
if display_adapter:  # True, we have RichDisplayAdapter
    display_adapter.show_final_answer(final_output)  # ← DICT PASSED!
```

**What should happen**:
```python
# Should extract text first!
display_text = _extract_text(final_output)  # → "Quality Assurance Report..."
display_adapter.show_final_answer(display_text)
```

**What actually happens**:
```python
# Dict passed directly!
display_adapter.show_final_answer({
    'output': "Quality Assurance Report...",
    'current_route': 'analysis',
    'router_decision': {...}
})
```

---

### Step 4: Display Adapter Receives Dict

**Location**: `display_adapter.py:177-181`

```python
def show_final_answer(self, answer: str) -> None:  # ← Type says str!
    """Show final assistant response."""
    from .events import emit_final_answer
    emit_final_answer(answer)  # ← Dict passed to events
    self.rich_display.show_final_output(answer)  # ← Dict passed to Rich
```

**Actual parameter**:
```python
answer = {
    'output': "Quality Assurance Report...",
    'current_route': 'analysis',
    'router_decision': {...}
}
# Type: dict, NOT str!
```

**Problem**: Type annotation says `str`, but receives `dict`.

---

### Step 5: Rich Display Receives Dict

**Location**: `rich_display.py:514-521`

```python
def show_final_output(self, output: str) -> None:  # ← Type says str!
    """Display final output."""
    self.console.print(
        Panel(
            Text(output, style="white"),  # ← Dict passed to Text()!
            title="💬 [bold green]Assistant Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
```

**Actual parameter**:
```python
output = {
    'output': "Quality Assurance Report...",
    'current_route': 'analysis',
    'router_decision': {...}
}
```

---

### Step 6: Rich Text Constructor (THE CRASH)

**Location**: `rich/text.py:156`

```python
class Text:
    def __init__(self, text, style="white"):
        sanitized_text = strip_control_codes(text)  # ← Dict passed!
```

**Function Call**: `strip_control_codes(text={...})`

**Location**: `rich/control.py:192`

```python
def strip_control_codes(text: str) -> str:
    return text.translate(_translate_table)  # ← CRASH!
```

**Error**:
```
AttributeError: 'dict' object has no attribute 'translate'
```

**Why**: Dict objects don't have a `.translate()` method. Only strings do.

---

## What Should Have Happened

### Correct Flow

```python
# Step 1: Extract state
final_output = self._extract_result_from_state(result)
# Returns: {'output': "text", 'current_route': ..., 'router_decision': ...}

# Step 2: Extract text before display
if display_adapter:
    from orchestrator.cli.main import _extract_text
    display_text = _extract_text(final_output)
    # Returns: "Quality Assurance Report..."
    display_adapter.show_final_answer(display_text)
    # Receives: string

# Step 3: Display receives string
def show_final_answer(self, answer: str):
    self.rich_display.show_final_output(answer)  # String passed

# Step 4: Rich receives string
def show_final_output(self, output: str):
    Text(output, style="white")  # String passed - NO CRASH!
```

---

## The _extract_text() Helper

**Location**: `main.py:325-354`

**What it does**:

```python
def _extract_text(output: Any) -> str:
    """Extract text content from various output formats."""

    # Case 1: Already a string
    if isinstance(output, str):
        return output  # "Quality Assurance Report..."

    # Case 2: Dict with 'output' key (OUR CASE!)
    if isinstance(output, dict):
        if "output" in output:
            return str(output["output"])  # ← WOULD WORK!
            # Returns: "Quality Assurance Report..."

        # Fallback: Try common text fields
        for key in ["text", "content", "response", "result"]:
            if key in output:
                return str(output[key])

    # Case 3: Object with attributes
    if hasattr(output, "final_output"):
        return str(output.final_output)

    # Fallback: Convert to string
    return str(output)
```

**Why it would work**: It checks for `dict` and extracts the `'output'` key!

**Current usage** (chat_orchestrator.py:310):
```python
workflow_result = self.adapter.run_multi_agent_workflow(
    display_adapter=self.display,
)
return _extract_text(workflow_result)  # ← TOO LATE! Display already crashed
```

---

## Comparison: Before vs After Fix

### BEFORE (Broken)

```
_extract_result_from_state(result)
    ↓
{'output': "text", 'current_route': ..., 'router_decision': ...}
    ↓
display_adapter.show_final_answer(dict)  ← Dict!
    ↓
rich_display.show_final_output(dict)  ← Dict!
    ↓
Text(dict, style="white")  ← CRASH!
```

### AFTER (Fixed)

```
_extract_result_from_state(result)
    ↓
{'output': "text", 'current_route': ..., 'router_decision': ...}
    ↓
_extract_text(dict)  ← NEW STEP!
    ↓
"Quality Assurance Report..."
    ↓
display_adapter.show_final_answer(string)  ← String!
    ↓
rich_display.show_final_output(string)  ← String!
    ↓
Text(string, style="white")  ← SUCCESS!
```

---

## Type Analysis

### Current Type Flow (Broken)

```python
_extract_result_from_state() → Union[str, Dict[str, Any]]
                                ↓ (Dict in our case)
show_final_answer(answer: str)  ← Type mismatch!
                                ↓
show_final_output(output: str)  ← Type mismatch!
                                ↓
Text(text: str)                 ← Type mismatch! CRASH!
```

### Fixed Type Flow

```python
_extract_result_from_state() → Union[str, Dict[str, Any]]
                                ↓
_extract_text(output: Any) → str  ← Conversion!
                                ↓
show_final_answer(answer: str)  ← Type match ✓
                                ↓
show_final_output(output: str)  ← Type match ✓
                                ↓
Text(text: str)                 ← Type match ✓ SUCCESS!
```

---

## Why This Bug Exists

### PHASE 3 Enhancement

**Comment**: "PHASE 3 FIX: Returns routing metadata when present to support proper decision extraction."

**Change**: Function modified to return dict with routing metadata:

```python
# Before PHASE 3: Always returned string
return clean_output

# After PHASE 3: Returns dict when routing metadata exists
if routing_metadata:
    return {
        'output': clean_output,
        'current_route': routing_metadata.get('decision'),
        'router_decision': routing_metadata.get('router_decision')
    }
return clean_output  # Legacy format for backward compatibility
```

**Problem**: Display layer not updated to handle new dict format.

---

## Impact on Different Workflows

### Simple Workflow (No Router)

```python
# Query: "What is 2+2?"
# Flow: Researcher only

_extract_result_from_state(result)
    ↓
routing_metadata = {}  # Empty! No router
    ↓
return clean_output  # String "4"
    ↓
display_adapter.show_final_answer("4")  ← String! Works!
```

**Result**: ✅ Works (no routing metadata, returns string)

### Multi-Agent Router Workflow

```python
# Query: "Who is the best F1 team?"
# Flow: Router → Researcher → Analyst → QA

_extract_result_from_state(result)
    ↓
routing_metadata = {'decision': 'analysis', ...}  # Has metadata!
    ↓
return {'output': "...", 'current_route': ..., ...}  # Dict!
    ↓
display_adapter.show_final_answer(dict)  ← Dict! CRASH!
```

**Result**: ❌ Crashes (has routing metadata, returns dict)

---

## The Fix (Detailed)

### Current Code (Lines 564-567)

```python
# Show final answer via display adapter if available
if display_adapter:
    display_adapter.show_final_answer(final_output)

return final_output  # Return clean text, not state object
```

### Fixed Code

```python
# Show final answer via display adapter if available
if display_adapter:
    # DISPLAY FIX: Extract text from dict/string before display
    # _extract_result_from_state() returns dict with routing metadata,
    # but display layer requires string. Use _extract_text() to convert.
    from orchestrator.cli.main import _extract_text
    display_text = _extract_text(final_output)
    display_adapter.show_final_answer(display_text)

return final_output  # Return clean text, not state object
```

### What Changes

**Added**:
- Import of `_extract_text` helper
- Conversion: `display_text = _extract_text(final_output)`
- Pass converted text instead of raw output

**Unchanged**:
- Return value (still returns dict/string for metadata consumers)
- Control flow
- Error handling

---

## Verification Steps

### Test 1: Multi-Agent Router (The Broken Case)

```bash
python -m orchestrator.cli chat --memory-provider hybrid
```

**Query**: `"Who is the best Formula 1 team and driver?"`

**Before Fix**:
```
✓ Router completed
✓ Researcher completed
✓ Analyst completed
✓ Quality Assurance completed
✓ Workflow completed
AttributeError: 'dict' object has no attribute 'translate'
```

**After Fix**:
```
✓ Router completed
✓ Researcher completed
✓ Analyst completed
✓ Quality Assurance completed
✓ Workflow completed
┌─ 💬 Assistant Response ─────────────────────────┐
│ Quality Assurance Report                        │
│                                                  │
│ **Final Summary**                               │
│ [2592 characters of report displayed cleanly]   │
└─────────────────────────────────────────────────┘
```

### Test 2: Simple Query (Backward Compatibility)

**Query**: `"What is 2+2?"`

**Before and After Fix**:
```
✓ Researcher completed
┌─ 💬 Assistant Response ─────────────────────────┐
│ 4                                                │
└─────────────────────────────────────────────────┘
```

**Status**: ✅ No regression

---

## Conclusion

The error is a **type mismatch** caused by an **incomplete PHASE 3 migration**. The function returns a dict for advanced workflows, but the display layer expects a string. The fix is simple: extract text before display using the existing `_extract_text()` helper.

**Root Cause**: Data flow coordination issue
**Fix Complexity**: LOW (3 lines)
**Risk Level**: LOW (uses existing helper)
**Impact**: HIGH (100% failure on router workflows)

**Recommended Action**: Apply fix immediately.
