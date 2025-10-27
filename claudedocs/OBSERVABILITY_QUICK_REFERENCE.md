# Observability Issues - Quick Reference Guide

**Last Updated**: 2025-10-04

---

## 🎯 Three Issues Identified

### Issue 1: Final Output Not Displayed ❌

**Symptom**: Workflow completes but results are invisible to user

**Root Cause**: Results extracted but never printed to console

**Fix Location**: `/orchestrator/cli/main.py` (chat loop)

**Complexity**: ⭐ (Simple - add print statement)

---

### Issue 2: No Mermaid Diagrams 📊

**Symptom**: Cannot visualize workflow graph structure

**Root Cause**: LangGraph has `draw_mermaid()` but not integrated

**Fix Location**:
- New file: `/orchestrator/cli/mermaid_utils.py`
- Integration: `/orchestrator/cli/graph_adapter.py:375`

**Complexity**: ⭐⭐ (Medium - create utils + integrate)

---

### Issue 3: Agent Execution Black Box 🔍

**Symptom**: No visibility into prompts, tools, or responses during agent execution

**Root Cause**: No logging/tracing in agent execution layer

**Fix Location**:
- `/orchestrator/integrations/langchain_integration.py:341-374`
- `/orchestrator/planning/graph_compiler.py:220-256`

**Complexity**: ⭐⭐⭐ (High - modify multiple layers + callbacks)

---

## 🔧 Quick Fixes

### Fix 1: Display Final Output (5 minutes)

```python
# In orchestrator/cli/main.py, after workflow execution:

result = adapter.execute_route(...)

# ADD THIS:
final_text = _extract_text(result)
if final_text:
    console.print(Panel(
        Markdown(final_text),
        title="[bold green]Workflow Result[/bold green]",
        border_style="green"
    ))
```

### Fix 2: Generate Mermaid Diagrams (30 minutes)

**Step 1**: Create `orchestrator/cli/mermaid_utils.py`:

```python
from pathlib import Path
from datetime import datetime

def save_mermaid_diagram(compiled_graph, output_dir="claudedocs/graphs"):
    """Generate and save Mermaid diagram."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate diagram
    graph_obj = compiled_graph.get_graph()
    mermaid_code = graph_obj.draw_mermaid()

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagram_file = Path(output_dir) / f"workflow_{timestamp}.mmd"
    diagram_file.write_text(mermaid_code)

    return diagram_file
```

**Step 2**: Integrate in `orchestrator/cli/graph_adapter.py:375`:

```python
from .mermaid_utils import save_mermaid_diagram

compiled_graph = compile_tot_graph(graph_spec, agent_configs)

# ADD THIS:
diagram_path = save_mermaid_diagram(compiled_graph)
logger.info(f"📊 Workflow diagram: {diagram_path}")
```

### Fix 3: Add Agent Tracing (1-2 hours)

**Minimal Implementation**:

```python
# In orchestrator/integrations/langchain_integration.py, LangChainAgent.execute():

def execute(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
    # ADD: Log prompt
    logger.info(f"[AGENT:{self.name}] Executing task")
    logger.debug(f"[AGENT:{self.name}] Prompt: {task_description[:500]}...")

    # ADD: Log tools
    if self.tools:
        tool_names = [t.name for t in self.tools if hasattr(t, 'name')]
        logger.info(f"[AGENT:{self.name}] Tools: {', '.join(tool_names)}")

    # EXISTING: Execute agent
    result = self.agent.invoke({"messages": [HumanMessage(content=task_description)]})

    # ADD: Log result
    logger.info(f"[AGENT:{self.name}] Completed")
    logger.debug(f"[AGENT:{self.name}] Response: {str(result)[:500]}...")

    return str(result)
```

---

## 🧪 Testing Commands

### Test Final Output Display

```bash
python -m orchestrator.cli chat --memory-provider hybrid

# Enter query: "What is Python?"
# Expected: See result in green Panel after workflow completes
```

### Test Mermaid Generation

```bash
python -m orchestrator.cli chat --memory-provider hybrid

# After workflow:
ls claudedocs/graphs/
# Should see: workflow_YYYYMMDD_HHMMSS.mmd

cat claudedocs/graphs/workflow_*.mmd
# Should see: graph TD syntax with nodes and edges
```

### Test Agent Tracing

```bash
export LOG_LEVEL=DEBUG
python -m orchestrator.cli chat --memory-provider hybrid

# Expected logs:
# [AGENT:Researcher] Executing task
# [AGENT:Researcher] Prompt: ...
# [AGENT:Researcher] Tools: duckduckgo, wikipedia
# [AGENT:Researcher] Completed
# [AGENT:Researcher] Response: ...
```

---

## 📊 LangGraph Visualization API

### Available Methods (Verified)

```python
compiled_graph = workflow.compile()
graph_obj = compiled_graph.get_graph()

# Text representations
graph_obj.draw_mermaid()        # Mermaid diagram (string)
graph_obj.draw_ascii()          # ASCII art (string)
graph_obj.print_ascii()         # Print ASCII to console

# Image generation
graph_obj.draw_mermaid_png()    # PNG image (bytes)
graph_obj.draw_png()            # PNG visualization (bytes)

# Serialization
graph_obj.to_json()             # JSON representation (dict)
```

### Example Mermaid Output

```mermaid
graph TD
    start --> router_node
    router_node --> researcher_node
    router_node --> analyst_node
    researcher_node --> end
    analyst_node --> end
```

---

## 🗂️ File Locations Reference

### Current Files (Modify)

| File | Purpose | Lines to Modify |
|------|---------|-----------------|
| `orchestrator/cli/main.py` | CLI entry point | ~450-500 (chat loop) |
| `orchestrator/cli/graph_adapter.py` | Graph execution | ~375 (after compilation) |
| `orchestrator/integrations/langchain_integration.py` | Agent execution | 341-374 (execute method) |
| `orchestrator/planning/graph_compiler.py` | Graph compilation | 220-256 (agent_function) |

### New Files (Create)

| File | Purpose | Size |
|------|---------|------|
| `orchestrator/cli/mermaid_utils.py` | Mermaid generation utilities | ~50-100 lines |
| `claudedocs/graphs/` | Diagram storage directory | N/A |

---

## 🚀 Implementation Priority

### Phase 1: CRITICAL (Do First)
- **Final Output Display**
- **Effort**: 5 minutes
- **Impact**: High (users can see results)
- **Risk**: None

### Phase 2: HIGH (Do Soon)
- **Mermaid Diagram Generation**
- **Effort**: 30 minutes
- **Impact**: High (workflow visualization)
- **Risk**: Low

### Phase 3: MEDIUM (Nice to Have)
- **Agent Execution Tracing**
- **Effort**: 1-2 hours
- **Impact**: Medium (debugging/observability)
- **Risk**: Medium (test callbacks carefully)

---

## 🔍 Debugging Hints

### If Final Output Still Missing

1. Check `_extract_text()` is being called
2. Verify `final_output` field exists in state
3. Add debug print: `print(f"DEBUG: result={result}")`
4. Check Rich display event handling

### If Mermaid Generation Fails

1. Verify LangGraph installation: `pip show langgraph`
2. Check graph compilation succeeds
3. Test manually: `compiled_graph.get_graph().draw_mermaid()`
4. Verify write permissions for `claudedocs/graphs/`

### If Agent Tracing Missing

1. Check log level: `export LOG_LEVEL=DEBUG`
2. Verify logger name matches: `logger = logging.getLogger(__name__)`
3. Test agent execution directly: `agent.execute("test query")`
4. Check if LangChain callbacks are supported by agent type

---

## 📝 Code Patterns

### Pattern: Safe Output Extraction

```python
def _extract_text(output: Any) -> str:
    """Robust extraction from various formats."""
    if hasattr(output, "final_output") and output.final_output:
        return str(output.final_output)
    if isinstance(output, dict) and "final_output" in output:
        return str(output["final_output"])
    # ... more fallbacks ...
    return str(output)
```

### Pattern: Timestamped File Generation

```python
from datetime import datetime
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = Path(output_dir) / f"workflow_{timestamp}.mmd"
file_path.write_text(content)
```

### Pattern: Hierarchical Logging

```python
# Node level
logger.info(f"[NODE:{node_spec.name}] Executing agent: {agent_name}")

# Agent level
logger.info(f"[AGENT:{self.name}] Executing task")

# Tool level
logger.info(f"[AGENT:{self.name}] 🔧 Tool start: {tool_name}")
```

---

## ✅ Success Criteria

### Issue 1 Fixed When:
- [ ] Workflow completes
- [ ] Console shows green Panel with result
- [ ] User can read the workflow output

### Issue 2 Fixed When:
- [ ] Workflow completes
- [ ] `claudedocs/graphs/workflow_*.mmd` file exists
- [ ] File contains valid Mermaid syntax
- [ ] Can be visualized (e.g., in VS Code Mermaid preview)

### Issue 3 Fixed When:
- [ ] Logs show agent execution start
- [ ] Logs show prompt content (truncated)
- [ ] Logs show tools used (if any)
- [ ] Logs show agent completion
- [ ] Logs show response content (truncated)

---

## 🔗 Related Documentation

- Full Analysis: `OBSERVABILITY_ISSUES_ROOT_CAUSE_ANALYSIS.md`
- LangGraph API: https://langchain-ai.github.io/langgraph/
- Mermaid Syntax: https://mermaid.js.org/syntax/flowchart.html
