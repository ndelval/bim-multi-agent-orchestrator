# Fix Implementation: Display Dict Error

**Issue**: AttributeError when displaying workflow results
**Root Cause**: Dict passed to Rich Text() instead of string
**Solution**: Extract text before display using existing `_extract_text()` helper

---

## Quick Fix (Recommended)

### File: `orchestrator/cli/graph_adapter.py`

**Lines 564-567** - Change from:

```python
# Show final answer via display adapter if available
if display_adapter:
    display_adapter.show_final_answer(final_output)
```

**To**:

```python
# Show final answer via display adapter if available
if display_adapter:
    from orchestrator.cli.main import _extract_text
    display_text = _extract_text(final_output)
    display_adapter.show_final_answer(display_text)
```

---

## Complete Fixed Code Block

```python
# PRIORITY 1 FIX: Extract final output from state before returning
final_output = self._extract_result_from_state(result)

# Show final answer via display adapter if available
if display_adapter:
    from orchestrator.cli.main import _extract_text
    display_text = _extract_text(final_output)
    display_adapter.show_final_answer(display_text)

return final_output  # Return clean text, not state object
```

---

## Why This Works

1. **`_extract_result_from_state()`** returns either:
   - String: `"output text"` (simple workflows)
   - Dict: `{'output': "text", 'current_route': ..., 'router_decision': ...}` (router workflows)

2. **`_extract_text()`** handles both cases:
   - If string: returns as-is
   - If dict: extracts `output['output']` or other text fields

3. **`show_final_answer()`** receives clean string text for display

4. **Return value** still preserves dict with metadata for callers that need it

---

## Testing Commands

### Test Multi-Agent Router Workflow
```bash
python -m orchestrator.cli chat --memory-provider hybrid
# Query: "Who is the best Formula 1 team and driver?"
# Expected: Clean display of quality assurance report
```

### Test Simple Workflow (Backward Compatibility)
```bash
python -m orchestrator.cli chat --memory-provider hybrid
# Query: "What is 2+2?"
# Expected: Works as before
```

---

## Verification Checklist

- [ ] No AttributeError during display
- [ ] Text displayed correctly in Rich Panel
- [ ] Quality assurance report visible
- [ ] Router workflows work end-to-end
- [ ] Simple workflows still work (backward compatibility)
- [ ] Return value preserves dict structure for metadata extraction

---

## Alternative: Robust Display Layer (Optional Enhancement)

If you want to make the display layer more robust, also update:

### File: `orchestrator/cli/display_adapter.py`

**Lines 177-181** - Change type annotation and add text extraction:

```python
def show_final_answer(self, answer: Union[str, Dict[str, Any]]) -> None:
    """Show final assistant response."""
    from .events import emit_final_answer

    # Extract text if dict (defensive programming)
    if isinstance(answer, dict):
        display_text = answer.get('output', str(answer))
    else:
        display_text = str(answer)

    emit_final_answer(display_text)
    self.rich_display.show_final_output(display_text)
```

This adds defense-in-depth but is **not required** if the quick fix is applied.

---

## Code Comments to Add

Add this comment above the fix for future maintainers:

```python
# DISPLAY FIX: Extract text from dict/string before display
# _extract_result_from_state() returns dict with routing metadata,
# but display layer requires string. Use _extract_text() to convert.
if display_adapter:
    from orchestrator.cli.main import _extract_text
    display_text = _extract_text(final_output)
    display_adapter.show_final_answer(display_text)
```

---

## Impact Summary

**Files Changed**: 1 (graph_adapter.py)
**Lines Changed**: 3 (add 2 lines)
**Risk Level**: LOW
**Testing Required**: Multi-agent workflows with router
**Backward Compatibility**: MAINTAINED

---

## Follow-Up Improvements (Optional)

1. Update type annotations in `show_final_answer()` to `Union[str, Dict[str, Any]]`
2. Add runtime type validation in display layer
3. Create integration tests for dict and string display paths
4. Document the two-format return pattern in code comments
5. Consider refactoring to eliminate mixed return types (PHASE 4)
