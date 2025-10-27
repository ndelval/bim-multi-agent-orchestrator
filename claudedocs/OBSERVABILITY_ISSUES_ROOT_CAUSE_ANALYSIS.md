# Observability Issues - Root Cause Analysis

**Date**: 2025-10-04
**Analysis Type**: Root Cause Analysis & Implementation Plan
**Components**: Graph Execution, Mermaid Visualization, Agent Tracing

---

## Executive Summary

Analysis reveals **three distinct observability gaps** in the LangGraph workflow system:

1. **Final Output Visibility Issue**: Results are extracted but **not displayed** to the user
2. **Missing Mermaid Diagram Generation**: LangGraph has built-in `draw_mermaid()` support but **not integrated**
3. **Agent Execution Tracing Gap**: No hooks to capture prompts, tools, and responses during agent execution

All issues have **clear solutions** with specific file locations identified.

---

## Issue 1: Final Output Visibility

### Root Cause

**Location**: `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/cli/graph_adapter.py`

**The Problem**: Results are extracted correctly but execution flow never displays them.

#### Current Flow (Lines 331-422):

```python
def _execute_stategraph_route(self, ...):
    # ...
    result = compiled_graph.invoke(initial_state)  # Line 397

    # Post-execution validation exists
    final_depth = len(execution_path) if execution_path else 0
    logger.debug(f"Workflow completed successfully in {final_depth} execution steps")  # Line 420

    return self._extract_result_from_state(result)  # Line 422 - RETURNS but never displayed
```

#### Why Output is Missing:

1. **`_execute_stategraph_route()` returns the result** to `execute_route()` (line 213)
2. **`execute_route()` is called by `run_multi_agent_workflow()`** (line 658)
3. **`run_multi_agent_workflow()` extracts and emits the result** (lines 675-682):
   ```python
   final_output = self._extract_result_from_state(result)
   if rich_display:
       from .events import emit_final_answer
       emit_final_answer(final_output)
   return final_output
   ```
4. **BUT: The CLI entry point (`main.py`) doesn't call `run_multi_agent_workflow()`** - it calls `execute_route()` directly or through router logic

#### Evidence from Code:

**`orchestrator/cli/main.py`** (lines 328-367):
```python
def _extract_text(output: Any) -> str:
    """Extract text content from various output formats."""
    # Correctly handles final_output field
    if hasattr(output, "final_output") and output.final_output:
        return str(output.final_output)
    # ... fallback logic ...
```

**The extraction logic exists and works correctly**. The issue is **the extracted output is never printed or displayed in the CLI loop**.

### Validation

**Graph Compiler (lines 189-194)** correctly generates final output:
```python
if not state.final_output and state.agent_outputs:
    final_output = self._generate_final_output(state)
    updates["final_output"] = final_output
    updates["messages"] = [AIMessage(content=final_output)]

logger.info(f"Workflow completed at node: {node_spec.name}")  # Line 194
```

**Log shows completion but no result display**.

---

## Issue 2: Mermaid Diagram Generation

### Root Cause

**LangGraph has built-in Mermaid support but it's not integrated into the workflow**.

#### Discovery Evidence:

```python
# LangGraph API (verified via runtime inspection)
compiled_graph = workflow.compile()
graph_obj = compiled_graph.get_graph()

# Available methods:
graph_obj.draw_mermaid()         # Returns Mermaid diagram string
graph_obj.draw_mermaid_png()     # Generates PNG visualization
graph_obj.draw_ascii()           # ASCII art representation
graph_obj.print_ascii()          # Print ASCII to console
```

#### Current State:

- **No integration in GraphCompiler** (`orchestrator/planning/graph_compiler.py`)
- **No integration in CLI** (`orchestrator/cli/graph_adapter.py`)
- **No save functionality** for diagrams

#### Where to Integrate:

**Option 1: After Compilation (Recommended)**
```python
# orchestrator/cli/graph_adapter.py, line ~375
compiled_graph = compile_tot_graph(graph_spec, agent_configs)

# ADD: Generate and save Mermaid diagram
_save_mermaid_diagram(compiled_graph, "graphs/workflow_execution.mmd")
```

**Option 2: In GraphCompiler**
```python
# orchestrator/planning/graph_compiler.py, line ~106
workflow = compiler.compile_graph_spec(graph_spec, agent_configs)

# ADD: Generate diagram as part of compilation
mermaid_code = workflow.get_graph().draw_mermaid()
return workflow, mermaid_code
```

### Suggested Directory Structure

```
claudedocs/
└── graphs/
    ├── workflow_TIMESTAMP.mmd          # Mermaid source
    ├── workflow_TIMESTAMP.png          # Visual diagram (optional)
    └── latest_workflow.mmd             # Symlink to latest
```

---

## Issue 3: Agent Execution Tracing

### Root Cause

**Location**: `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/planning/graph_compiler.py`

**The Problem**: Agent execution happens in a black box with no visibility into LLM interactions.

#### Current Agent Execution (Lines 220-256):

```python
def agent_function(state: OrchestratorState) -> Dict[str, Any]:
    try:
        logger.debug(f"Executing agent node: {node_spec.name} (agent: {agent_name}")  # Line 222

        # Build task description
        task_description = self._build_task_description(node_spec, state)  # Line 225

        # Execute agent (BLACK BOX - no visibility)
        result = agent.execute(task_description, {"state": asdict(state)})  # Line 228

        logger.debug(f"Agent {agent_name} completed execution in node {node_spec.name}")  # Line 230

        # Return result
        return {
            "agent_outputs": {**state.agent_outputs, agent_name: result},
            # ...
        }
```

#### Missing Observability:

1. **No logging of the actual prompt sent to the agent** (line 228)
2. **No logging of tools used by the agent**
3. **No logging of the agent's raw response**
4. **No execution metadata** (duration, tokens, tool calls)

### Agent Backend Implementation

**Location**: `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/factories/agent_backends.py`

**LangChain Agent Execution** (lines 131-166):

```python
def create_agent(self, config: 'AgentConfig', **kwargs) -> Any:
    # Creates agent with tools
    agent = self._agent_class(
        name=config.name,
        role=config.role,
        # ...
        llm=llm,
        tools=tools,  # Line 159 - tools are attached here
    )
```

**Agent Execute Method** (in `langchain_integration.py`, lines 341-374):

```python
def execute(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
    try:
        # THIS IS WHERE WE NEED TRACING
        if hasattr(self.agent, 'invoke'):
            result = self.agent.invoke({
                "messages": [HumanMessage(content=task_description)]
            })
        # ...
        return str(result)
    except Exception as e:
        logger.error(f"Agent {self.name} execution failed: {e}")
        return f"Error: Agent execution failed - {str(e)}"
```

### What Needs to Be Captured:

1. **Prompt Construction**:
   - System prompt (role, goal, backstory, instructions)
   - User prompt (task_description)
   - Full combined prompt sent to LLM

2. **Tool Invocation**:
   - Tool names used
   - Tool inputs
   - Tool outputs

3. **Agent Response**:
   - Raw LLM output
   - Parsed result
   - Error details (if any)

4. **Execution Metadata**:
   - Agent name and node
   - Execution duration
   - Token usage (if available)

---

## Implementation Plan

### Phase 1: Final Output Display (Priority: CRITICAL)

**Files to Modify**:
1. `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/cli/main.py`

**Changes**:

```python
# Around line 450-500 (in the CLI chat loop)
# After workflow execution completes:

# BEFORE (current - no display):
result = adapter.execute_route(...)

# AFTER (add display):
result = adapter.execute_route(...)

# Extract and display final output
final_text = _extract_text(result)
if final_text:
    console.print("\n")
    console.print(Panel(
        Markdown(final_text),
        title="[bold green]Workflow Result[/bold green]",
        border_style="green"
    ))
else:
    logger.warning("No output generated from workflow")
```

**Alternative: Enhance `emit_final_answer` Event**

If Rich display is used:
```python
# orchestrator/cli/graph_adapter.py, line 680
emit_final_answer(final_output)

# Ensure rich_display.py properly handles this event
# Check orchestrator/cli/rich_display.py, line 507:
def show_final_output(self, output: str) -> None:
    """Display final workflow output."""
    # Ensure this is called by event handler
```

### Phase 2: Mermaid Diagram Generation (Priority: HIGH)

**Files to Create**:
1. `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/cli/mermaid_utils.py`

**Implementation**:

```python
"""Mermaid diagram generation utilities for LangGraph visualization."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def save_mermaid_diagram(
    compiled_graph,
    output_dir: str = "claudedocs/graphs",
    filename: Optional[str] = None
) -> Path:
    """
    Generate and save Mermaid diagram from compiled LangGraph.

    Args:
        compiled_graph: Compiled LangGraph StateGraph
        output_dir: Directory to save diagram (created if missing)
        filename: Optional filename (default: workflow_TIMESTAMP.mmd)

    Returns:
        Path to saved Mermaid file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_{timestamp}.mmd"

    # Get Mermaid diagram
    try:
        graph_obj = compiled_graph.get_graph()
        mermaid_code = graph_obj.draw_mermaid()

        # Save to file
        diagram_file = output_path / filename
        diagram_file.write_text(mermaid_code, encoding='utf-8')

        # Create 'latest' symlink
        latest_link = output_path / "latest_workflow.mmd"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(diagram_file.name)

        logger.info(f"Mermaid diagram saved: {diagram_file}")
        return diagram_file

    except Exception as e:
        logger.error(f"Failed to generate Mermaid diagram: {e}")
        raise


def print_ascii_graph(compiled_graph) -> None:
    """Print ASCII representation of graph to console."""
    try:
        graph_obj = compiled_graph.get_graph()
        graph_obj.print_ascii()
    except Exception as e:
        logger.error(f"Failed to print ASCII graph: {e}")
```

**Integration Point**:

```python
# orchestrator/cli/graph_adapter.py, line ~375
from .mermaid_utils import save_mermaid_diagram

compiled_graph = compile_tot_graph(graph_spec, agent_configs)

# Generate and save Mermaid diagram
try:
    diagram_path = save_mermaid_diagram(compiled_graph)
    logger.info(f"📊 Workflow diagram: {diagram_path}")
except Exception as e:
    logger.warning(f"Could not generate diagram: {e}")
```

### Phase 3: Agent Execution Tracing (Priority: MEDIUM)

**Files to Modify**:
1. `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/integrations/langchain_integration.py`
2. `/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/orchestrator/planning/graph_compiler.py`

**Strategy 1: Add Logging Callbacks**

```python
# orchestrator/integrations/langchain_integration.py, in LangChainAgent.execute()

def execute(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Execute task with detailed tracing."""

    # Log the prompt being sent
    logger.info(f"[AGENT:{self.name}] Executing task")
    logger.debug(f"[AGENT:{self.name}] Prompt: {task_description[:500]}...")  # Truncate for logs

    # Log tools available
    if self.tools:
        tool_names = [t.name if hasattr(t, 'name') else str(t) for t in self.tools]
        logger.info(f"[AGENT:{self.name}] Tools: {', '.join(tool_names)}")

    try:
        start_time = time.time()

        # Execute with callbacks for tool tracing
        if hasattr(self.agent, 'invoke'):
            result = self.agent.invoke({
                "messages": [HumanMessage(content=task_description)]
            })

        duration = time.time() - start_time

        # Log execution result
        logger.info(f"[AGENT:{self.name}] Completed in {duration:.2f}s")

        # Extract and log response
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]
            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            logger.debug(f"[AGENT:{self.name}] Response: {response_text[:500]}...")
            return response_text

        # ... existing return logic ...

    except Exception as e:
        logger.error(f"[AGENT:{self.name}] Execution failed: {e}")
        return f"Error: Agent execution failed - {str(e)}"
```

**Strategy 2: Enhanced Graph Compiler Logging**

```python
# orchestrator/planning/graph_compiler.py, in agent_function()

def agent_function(state: OrchestratorState) -> Dict[str, Any]:
    try:
        # Build task description
        task_description = self._build_task_description(node_spec, state)

        # LOG PROMPT CONSTRUCTION
        logger.info(f"[NODE:{node_spec.name}] Executing agent: {agent_name}")
        logger.debug(f"[NODE:{node_spec.name}] Task description:\n{task_description}")

        # LOG STATE CONTEXT
        logger.debug(f"[NODE:{node_spec.name}] State context: "
                    f"completed_agents={state.completed_agents}, "
                    f"execution_depth={state.execution_depth}")

        # Execute agent
        result = agent.execute(task_description, {"state": asdict(state)})

        # LOG RESULT
        logger.info(f"[NODE:{node_spec.name}] Agent completed successfully")
        logger.debug(f"[NODE:{node_spec.name}] Result: {result[:500]}...")

        return {
            "agent_outputs": {**state.agent_outputs, agent_name: result},
            "node_outputs": {**state.node_outputs, node_spec.name: result},
            "completed_agents": state.completed_agents + [agent_name],
            "execution_path": state.execution_path + [node_spec.name],
            "messages": [AIMessage(content=result)]
        }
```

**Strategy 3: Tool Call Tracing (Advanced)**

For LangChain ReAct agents with tools:

```python
# orchestrator/integrations/langchain_integration.py

from langchain.callbacks.base import BaseCallbackHandler

class AgentTracingCallback(BaseCallbackHandler):
    """Callback handler for detailed agent execution tracing."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Log tool invocation start."""
        tool_name = serialized.get("name", "unknown")
        logger.info(f"[AGENT:{self.agent_name}] 🔧 Tool start: {tool_name}")
        logger.debug(f"[AGENT:{self.agent_name}] Tool input: {input_str}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log tool invocation result."""
        logger.info(f"[AGENT:{self.agent_name}] ✓ Tool complete")
        logger.debug(f"[AGENT:{self.agent_name}] Tool output: {output[:200]}...")

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """Log LLM prompt."""
        logger.debug(f"[AGENT:{self.agent_name}] LLM prompt: {prompts[0][:500]}...")

# In LangChainAgent.execute():
callbacks = [AgentTracingCallback(self.name)]
result = self.agent.invoke(
    {"messages": [HumanMessage(content=task_description)]},
    config={"callbacks": callbacks}
)
```

---

## Testing Strategy

### Test 1: Final Output Display

```bash
# Run workflow and verify output is displayed
python -m orchestrator.cli chat --memory-provider hybrid

# Expected behavior:
# 1. Workflow executes
# 2. Logger shows: "Workflow completed at node: end"
# 3. Console displays Panel with Markdown result
# 4. Result is visible to user
```

### Test 2: Mermaid Diagram Generation

```bash
# Run workflow
python -m orchestrator.cli chat --memory-provider hybrid

# Verify files created:
ls -la claudedocs/graphs/
# Should see:
# - workflow_YYYYMMDD_HHMMSS.mmd
# - latest_workflow.mmd (symlink)

# Verify diagram contents:
cat claudedocs/graphs/latest_workflow.mmd
# Should see valid Mermaid syntax:
# graph TD
#   start --> ...
#   ... --> end
```

### Test 3: Agent Execution Tracing

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python -m orchestrator.cli chat --memory-provider hybrid

# Expected log output:
# [AGENT:Researcher] Executing task
# [AGENT:Researcher] Prompt: ...
# [AGENT:Researcher] Tools: duckduckgo, wikipedia
# [AGENT:Researcher] 🔧 Tool start: duckduckgo
# [AGENT:Researcher] ✓ Tool complete
# [AGENT:Researcher] Completed in 2.34s
# [AGENT:Researcher] Response: ...
```

---

## Priority Ranking

1. **CRITICAL**: Final Output Display
   - **Impact**: Users cannot see workflow results
   - **Effort**: Low (add print statement in CLI loop)
   - **Files**: 1 file modification

2. **HIGH**: Mermaid Diagram Generation
   - **Impact**: No workflow visualization
   - **Effort**: Medium (create utils, integrate)
   - **Files**: 1 new file, 1 file modification

3. **MEDIUM**: Agent Execution Tracing
   - **Impact**: Limited debugging capability
   - **Effort**: High (modify multiple layers, test callbacks)
   - **Files**: 2-3 file modifications

---

## File Change Summary

### Files to Modify

1. **`orchestrator/cli/main.py`**
   - Add final output display in chat loop
   - Lines: ~450-500

2. **`orchestrator/cli/graph_adapter.py`**
   - Integrate Mermaid diagram generation
   - Lines: ~375 (after graph compilation)

3. **`orchestrator/integrations/langchain_integration.py`**
   - Add agent execution tracing
   - Lines: 341-374 (LangChainAgent.execute)

4. **`orchestrator/planning/graph_compiler.py`**
   - Add node-level execution logging
   - Lines: 220-256 (agent_function)

### Files to Create

1. **`orchestrator/cli/mermaid_utils.py`**
   - Mermaid diagram generation utilities
   - ~50-100 lines

2. **`claudedocs/graphs/` (directory)**
   - Storage for generated diagrams

---

## Code Snippet Reference

### Key Execution Flow

```
main.py:chat_command()
    ↓
graph_adapter.py:execute_route()
    ↓
graph_adapter.py:_execute_stategraph_route()
    ↓
graph_compiler.py:compile_tot_graph()
    ↓
[Compiled Graph].invoke(state)
    ↓
graph_compiler.py:agent_function()
    ↓
langchain_integration.py:LangChainAgent.execute()
    ↓
[LangChain ReAct Agent].invoke()
    ↓
RESULT (currently not displayed)
```

### Critical Line Numbers

- **Graph Execution**: `graph_adapter.py:397` - `compiled_graph.invoke(initial_state)`
- **Result Extraction**: `graph_adapter.py:422` - `self._extract_result_from_state(result)`
- **Graph Compilation**: `graph_compiler.py:516` - `workflow.compile()`
- **Agent Execution**: `graph_compiler.py:228` - `agent.execute(task_description, ...)`
- **LLM Invocation**: `langchain_integration.py:347` - `self.agent.invoke(...)`

---

## Next Steps

1. **Implement Phase 1** (Final Output Display) - Immediate fix
2. **Implement Phase 2** (Mermaid Diagrams) - High value for debugging
3. **Implement Phase 3** (Agent Tracing) - Enhanced observability

Each phase is **independent** and can be implemented separately.
