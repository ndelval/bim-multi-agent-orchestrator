# ToT Reasoning & Troubleshooting UI Visualization Specification

**Document Purpose**: Technical specification for exposing Tree-of-Thought reasoning and troubleshooting information in CLI UI

**Target Audience**: Frontend developer, UI architect

**Architecture Context**: ToT Planning → StateGraph Compilation → Agent Execution (with fallback to PraisonAI)

---

## 1. ToT Reasoning Flow Extraction Points

### 1.1 Thought Generation Process

**Primary Location**: `orchestrator/planning/tot_planner.py:432-436`

```python
# Current code (line 432-436):
try:
    plans, info = solve(args, task, idx=0, to_print=False)
except Exception as exc:
    logger.exception("ToT planner failed during solve()")
    raise
```

**Extraction Point #1: Raw ToT Solver Output**
```python
# File: orchestrator/planning/tot_planner.py
# Function: generate_plan_with_tot()
# Lines: 432-454

# What to extract:
{
    "plans": plans,              # List[str] - All generated plan candidates
    "info": {
        "steps": [                # List of step information from BFS solver
            {
                "step": 0,         # int - Planning step number
                "x": "problem...", # str - Original problem statement
                "ys": [""],        # List[str] - Input candidates for this step
                "new_ys": [...],   # List[str] - Generated candidate extensions
                "values": [...],   # List[float] - Evaluation scores (0-10)
                "select_new_ys": [...] # List[str] - Selected candidates after pruning
            }
        ]
    },
    "best_plan": plans[0],       # str - Highest scoring final plan
    "tot_usage": gpt_usage(backend) # Dict - Token usage statistics
}
```

**UI Event to Emit**:
```typescript
interface ToTGenerationEvent {
  timestamp: string;
  stage: "generation" | "evaluation" | "selection";
  step: number;
  details: {
    generation?: {
      input_candidates: string[];
      generated_candidates: string[];
      generation_mode: "sample" | "propose";
      samples_per_candidate: number;
    };
    evaluation?: {
      scores: number[];
      score_range: [number, number];
      evaluation_mode: "value" | "vote";
      evaluation_samples: number;
    };
    selection?: {
      selected_candidates: string[];
      selected_indices: number[];
      pruned_count: number;
      selection_mode: "greedy" | "sample";
    };
  };
}
```

### 1.2 Thought Evaluation Process

**Primary Location**: `orchestrator/planning/tot_planner.py:231-252`

```python
# Value-based evaluation logic (lines 231-252)
def value_prompt_wrap(self, x: str, y: str) -> str:
    # Evaluation criteria exposed here
    return (
        "Evalúa la calidad del siguiente plan parcial para resolver la solicitud."
        ' Responde en JSON estricto con el formato {"score": <0-10>, "reason": "..."}.'
        # ... criteria ...
    )

def value_outputs_unwrap(self, x: str, y: str, value_outputs: List[str]) -> float:
    # Score extraction with fallback logic
```

**Extraction Point #2: Evaluation Criteria & Scores**
```python
# What to extract:
{
    "evaluation_criteria": [
        "cobertura del problema",    # Problem coverage
        "orden lógico",              # Logical ordering
        "claridad",                  # Clarity
        "uso adecuado de agentes"   # Agent utilization
    ],
    "raw_llm_responses": value_outputs,  # List[str] - Raw LLM evaluation responses
    "parsed_scores": scores,             # List[float] - Parsed numeric scores
    "best_score": max(scores),          # float - Highest score
    "parsing_failures": failures_count   # int - JSON parse failures
}
```

**UI Visualization**: Score distribution chart showing how each criterion influenced final score

### 1.3 Path Selection & Pruning

**Primary Location**: `tree-of-thought-llm/src/tot/methods/bfs.py:71-76` (external dependency)

**Extraction Point #3: Pruning Decisions**
```python
# Reasoning exposed in selection phase:
{
    "selection_strategy": "greedy" | "sample",
    "candidates_before_pruning": len(new_ys),
    "candidates_after_pruning": len(select_new_ys),
    "pruned_indices": [i for i in range(len(new_ys)) if i not in select_ids],
    "pruning_reason": "score_below_threshold" | "not_in_top_n",
    "pruning_details": {
        "threshold": min_score_selected,
        "pruned_scores": [values[i] for i in pruned_indices]
    }
}
```

**UI Visualization**: Thought tree with pruned branches shown in gray/strikethrough

---

## 2. Troubleshooting Visualization

### 2.1 Error Detection & Diagnosis

**Fallback Mechanism Flow**:
```
StateGraph Planning
    ↓ (try)
ToT Graph Planning (orchestrator/planning/tot_graph_planner.py:291-391)
    ↓ (fail)
Sequential Fallback Graph (tot_graph_planner.py:625-655)
    ↓ (fail)
PraisonAI Legacy System (orchestrator/cli/graph_adapter.py:165-174)
```

**Extraction Point #4: Fallback Detection**

**File**: `orchestrator/cli/graph_adapter.py`
**Lines**: 111-123, 165-174, 211-221

```python
# Current fallback logic (lines 116-123):
except Exception as e:
    logger.error(f"Router execution failed with {self.backend_type}: {e}")
    # Fallback to the other backend if possible
    if self.use_stategraph and self.backend_info["praisonai"]["available"]:
        logger.info("Falling back to PraisonAI for router execution")
        return self._execute_legacy_router(router_config, timeout)
    else:
        raise
```

**UI Event to Emit**:
```typescript
interface FallbackEvent {
  timestamp: string;
  stage: "planning" | "compilation" | "execution";
  fallback_level: 1 | 2 | 3; // 1=ToT→Sequential, 2=StateGraph→PraisonAI, 3=Critical failure
  original_backend: "stategraph" | "praisonai";
  fallback_backend: "sequential_graph" | "praisonai" | "none";
  error: {
    type: string;
    message: string;
    stack_trace?: string;
  };
  recovery_action: string;
  user_impact: "none" | "degraded" | "partial" | "critical";
}
```

### 2.2 Recovery Strategies Visualization

**Extraction Point #5: Recovery Decision Tree**

**File**: `orchestrator/planning/tot_planner.py`
**Lines**: 487-489, 534-539

```python
# Graph planning fallback logic (lines 487-489, 534-539):
if not _TOT_AVAILABLE and not _try_import_tot():
    logger.warning("ToT not available, falling back to assignments")
    enable_graph_planning = False

# Later fallback (lines 534-539):
except Exception as e:
    logger.warning(f"Graph planning failed: {e}, falling back to assignments")
    enable_graph_planning = False
```

**Recovery Strategy Taxonomy**:
```typescript
interface RecoveryStrategy {
  strategy_id: string;
  trigger_condition: string;
  decision_logic: string;
  recovery_path: string[];
  confidence_level: number; // 0.0-1.0
  expected_quality_degradation: number; // 0.0-1.0 (0=no degradation, 1=total loss)
}

// Example strategies:
const RECOVERY_STRATEGIES = [
  {
    strategy_id: "TOT_UNAVAILABLE",
    trigger_condition: "_TOT_AVAILABLE == False",
    decision_logic: "Check if tree-of-thought-llm package can be imported",
    recovery_path: ["disable_graph_planning", "use_sequential_fallback"],
    confidence_level: 0.9,
    expected_quality_degradation: 0.3
  },
  {
    strategy_id: "GRAPH_PLANNING_FAILED",
    trigger_condition: "Exception during generate_graph_with_tot()",
    decision_logic: "Catch exception, log warning, fallback to assignments",
    recovery_path: ["generate_fallback_graph", "create_simple_sequential"],
    confidence_level: 0.8,
    expected_quality_degradation: 0.4
  },
  {
    strategy_id: "STATEGRAPH_COMPILATION_FAILED",
    trigger_condition: "GraphCreationError or compilation exception",
    decision_logic: "Check PraisonAI availability, switch backend",
    recovery_path: ["fallback_to_praisonai", "execute_legacy_route"],
    confidence_level: 0.95,
    expected_quality_degradation: 0.2
  }
]
```

**UI Component**: Decision tree diagram showing current recovery path

### 2.3 Debugging Information Exposure

**Extraction Point #6: Compilation Trace**

**File**: `orchestrator/planning/graph_compiler.py`
**Lines**: 112-129 (node compilation), 336-353 (edge compilation)

```python
# Current node compilation (lines 118-129):
for node_spec in graph_spec.nodes:
    try:
        node_function = self._create_node_function(node_spec, agent_config_map)
        workflow.add_node(node_spec.name, node_function)
        self._node_functions[node_spec.name] = node_function

        logger.debug(f"Compiled node: {node_spec.name} ({node_spec.type.value})")

    except Exception as e:
        logger.error(f"Failed to compile node {node_spec.name}: {str(e)}")
        raise GraphCreationError(f"Node compilation failed for {node_spec.name}: {str(e)}")
```

**Enhanced Trace Structure**:
```typescript
interface CompilationTrace {
  graph_name: string;
  validation: {
    passed: boolean;
    errors: string[];
    timestamp: string;
  };
  node_compilation: NodeCompilationTrace[];
  edge_compilation: EdgeCompilationTrace[];
  entry_exit_setup: {
    entry_point: string;
    exit_points: string[];
    auto_detected: boolean;
  };
  summary: {
    total_nodes: number;
    nodes_compiled: number;
    total_edges: number;
    edges_compiled: number;
    total_errors: number;
    compilation_successful: boolean;
  };
}

interface NodeCompilationTrace {
  name: string;
  type: "START" | "END" | "AGENT" | "ROUTER" | "CONDITION" | "PARALLEL" | "AGGREGATOR";
  status: "success" | "failed";
  start_time: string;
  end_time: string;
  duration_ms: number;
  details?: {
    agent?: string;
    routing_strategy?: "LLM_BASED" | "STATE_BASED" | "RULE_BASED";
    agent_resolved?: boolean;
    function_created?: boolean;
  };
  error?: string;
}

interface EdgeCompilationTrace {
  from: string;
  to: string;
  type: "DIRECT" | "CONDITIONAL" | "PARALLEL" | "AGGREGATION";
  status: "success" | "failed" | "skipped";
  start_time: string;
  end_time: string;
  duration_ms: number;
  error?: string;
}
```

**UI Component**: Step-by-step compilation log with collapsible sections

---

## 3. Timeline Events Structure

### 3.1 Complete Event Taxonomy

```typescript
// Base event interface
interface BaseEvent {
  event_id: string;
  timestamp: string;
  elapsed_ms: number; // Time since planning started
  stage: EventStage;
  status: EventStatus;
}

enum EventStage {
  PLANNING_INIT = "planning_init",
  TOT_GENERATION = "tot_generation",
  TOT_EVALUATION = "tot_evaluation",
  TOT_SELECTION = "tot_selection",
  PLANNING_COMPLETE = "planning_complete",
  GRAPH_VALIDATION = "graph_validation",
  GRAPH_COMPILATION = "graph_compilation",
  EXECUTION_INIT = "execution_init",
  NODE_EXECUTION = "node_execution",
  EXECUTION_COMPLETE = "execution_complete",
  ERROR_RECOVERY = "error_recovery"
}

enum EventStatus {
  STARTED = "started",
  IN_PROGRESS = "in_progress",
  COMPLETED = "completed",
  FAILED = "failed",
  RECOVERED = "recovered"
}

// Specific event types
interface PlanningInitEvent extends BaseEvent {
  stage: EventStage.PLANNING_INIT;
  details: {
    agent_count: number;
    memory_provider: string;
    backend: string;
    max_steps: number;
    graph_planning_enabled: boolean;
  };
}

interface ToTGenerationEvent extends BaseEvent {
  stage: EventStage.TOT_GENERATION;
  details: {
    step: number;
    current_candidates: number;
    generated_candidates: number;
    generation_mode: "sample" | "propose";
    tokens_used: number;
  };
}

interface NodeExecutionEvent extends BaseEvent {
  stage: EventStage.NODE_EXECUTION;
  details: {
    node_name: string;
    node_type: string;
    agent_name?: string;
    execution_path: string[];
    output_preview?: string; // First 200 chars
  };
}

interface ErrorRecoveryEvent extends BaseEvent {
  stage: EventStage.ERROR_RECOVERY;
  details: {
    error_type: string;
    original_operation: string;
    recovery_strategy: string;
    fallback_path: string[];
    success: boolean;
  };
}

// Union type for all events
type TimelineEvent =
  | PlanningInitEvent
  | ToTGenerationEvent
  | ToTEvaluationEvent
  | ToTSelectionEvent
  | GraphValidationEvent
  | GraphCompilationEvent
  | NodeExecutionEvent
  | ErrorRecoveryEvent
  | ExecutionCompleteEvent;
```

### 3.2 Event Emission Points

**Planning Phase Events**:

| Event Type | File | Line | Function | When to Emit |
|------------|------|------|----------|--------------|
| PLANNING_INIT | tot_planner.py | 406-411 | generate_plan_with_tot() | Before ToT solver starts |
| TOT_GENERATION | tot_planner.py | 432 | generate_plan_with_tot() | After each generation step |
| TOT_EVALUATION | tot_planner.py | 432 | generate_plan_with_tot() | After each evaluation step |
| TOT_SELECTION | tot_planner.py | 432 | generate_plan_with_tot() | After each selection step |
| PLANNING_COMPLETE | tot_planner.py | 438-446 | generate_plan_with_tot() | After best plan selected |

**Compilation Phase Events**:

| Event Type | File | Line | Function | When to Emit |
|------------|------|------|----------|--------------|
| GRAPH_VALIDATION | graph_compiler.py | 77-80 | compile_graph_spec() | After validation check |
| NODE_COMPILATION | graph_compiler.py | 118-129 | _compile_nodes() | After each node compiled |
| EDGE_COMPILATION | graph_compiler.py | 338-353 | _compile_edges() | After each edge compiled |
| ENTRY_EXIT_SETUP | graph_compiler.py | 370-387 | _set_entry_exit_points() | After entry/exit configured |

**Execution Phase Events**:

| Event Type | File | Line | Function | When to Emit |
|------------|------|------|----------|--------------|
| EXECUTION_INIT | graph_adapter.py | 324 | _execute_stategraph_route() | Before graph.invoke() |
| NODE_EXECUTION | graph_compiler.py | 215-239 | agent_function() | After each node executes |
| EXECUTION_COMPLETE | graph_compiler.py | 179-191 | end_function() | After END node reached |

**Error & Recovery Events**:

| Event Type | File | Line | Function | When to Emit |
|------------|------|------|----------|--------------|
| ERROR_RECOVERY | graph_adapter.py | 116-123 | execute_router() | When fallback triggered |
| ERROR_RECOVERY | tot_planner.py | 534-539 | generate_graph_with_tot() | When planning fails |
| ERROR_RECOVERY | graph_compiler.py | 241-246 | agent_function() | When agent fails |

---

## 4. UI Visualization Recommendations

### 4.1 ToT Reasoning Tree Component

**Purpose**: Show how ToT explored different planning paths

**Visual Design**:
```
┌─────────────────────────────────────────────────────────────┐
│ Tree-of-Thought Planning Explorer                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 0 (Initial)                                            │
│  └─ [empty plan]                                             │
│      ├─ Generated 3 candidates ───────────────┐              │
│      │                                         │              │
│      ├─ Candidate A (score: 8.5) ●────────────┼─ SELECTED   │
│      │   "Agent: Researcher | Objective..."   │              │
│      │                                         │              │
│      ├─ Candidate B (score: 7.2) ○────────────┼─ PRUNED     │
│      │   "Agent: Analyst | Objective..."      │              │
│      │                                         │              │
│      └─ Candidate C (score: 6.1) ○────────────┘  PRUNED     │
│          "Agent: StandardsAgent | Objective..."              │
│                                                               │
│  Step 1 (Expansion)                                          │
│  ├─ Candidate A (from step 0)                                │
│  │   ├─ Extension A1 (score: 9.1) ●──────── SELECTED        │
│  │   │   + "Agent: Analyst | Objective..."                   │
│  │   │                                                        │
│  │   ├─ Extension A2 (score: 8.8) ○──────── PRUNED          │
│  │   │   + "Agent: StandardsAgent | Objective..."            │
│  │   │                                                        │
│  │   └─ Extension A3 (score: 7.5) ○──────── PRUNED          │
│  │       + "Agent: Researcher | Objective..."                │
│  │                                                            │
│  └─ [other branches pruned at step 0]                        │
│                                                               │
│  ✓ Final Plan (score: 9.1, 2 steps)                         │
│    Agent: Researcher | Objective: Gather sources...          │
│    Agent: Analyst | Objective: Synthesize findings...        │
│                                                               │
│  [View Raw Plan] [View Prompts] [View Evaluations]          │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Library: D3.js tree layout or React Flow
- Color coding: Green = selected (score >8), Yellow = considered (score 6-8), Red = pruned (score <6)
- Interactive: Click node to see full prompt/response
- Collapsible: Expand/collapse branches

### 4.2 Timeline Visualization

**Purpose**: Show chronological sequence of all planning/execution events

**Visual Design**:
```
┌─────────────────────────────────────────────────────────────────────┐
│ Execution Timeline                                    Total: 12.4s  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  0s ───┬─ PLANNING_INIT (0.1s) ──────────────────────────────┐      │
│        │   Backend: gpt-4o-mini, Agents: 3                     │      │
│        │                                                        │      │
│  0.1s ─┼─ TOT_GENERATION Step 0 (2.3s) ─────────────────────┐│      │
│        │   Generated 3 candidates                             ││      │
│        │                                                       ││      │
│  2.4s ─┼─ TOT_EVALUATION Step 0 (1.8s) ────────────────────┐││      │
│        │   Scores: [8.5, 7.2, 6.1]                          │││      │
│        │                                                      │││      │
│  4.2s ─┼─ TOT_SELECTION Step 0 (0.1s) ───────────────────┐ │││      │
│        │   Selected top 2 candidates                       │ │││      │
│        │                                                    │ │││      │
│  4.3s ─┼─ TOT_GENERATION Step 1 (2.5s) ─────────────────┐ │││      │
│        │   Generated 6 extensions (2 per candidate)      │ │││      │
│        │                                                  │ │││      │
│  6.8s ─┼─ TOT_EVALUATION Step 1 (2.0s) ───────────────┐│││││      │
│        │   Scores: [9.1, 8.8, 7.5, 7.8, 7.0, 6.3]      │││││││      │
│        │                                                 │││││││      │
│  8.8s ─┼─ PLANNING_COMPLETE (0.2s) ────────────────────┘││││││      │
│        │   Best plan: 2 steps, score: 9.1                 │││││      │
│        │                                                    │││││      │
│  9.0s ─┼─ GRAPH_COMPILATION (0.5s) ───────────────────────┘│││      │
│        │   Nodes: 5, Edges: 4                               │││      │
│        │                                                     │││      │
│  9.5s ─┼─ EXECUTION_INIT (0.1s) ──────────────────────────┘││      │
│        │   Entry: start, State initialized                  ││      │
│        │                                                     ││      │
│ 10.0s ─┼─ NODE: start (0.2s) ─────────────────────────────┘│      │
│        │   Workflow initialized                              │      │
│        │                                                      │      │
│ 10.2s ─┼─ NODE: agent_researcher (1.8s) ──────────────────┐│      │
│        │   Agent: Researcher, Output: "Found 15 sources..." ││      │
│        │                                                     ││      │
│ 12.0s ─┼─ NODE: agent_analyst (0.3s) ─────────────────────┘│      │
│        │   Agent: Analyst, Output: "Synthesized..."         │      │
│        │                                                      │      │
│ 12.3s ─┼─ NODE: end (0.1s) ─────────────────────────────────┘      │
│        │   Workflow completed                                        │
│        │                                                              │
│ 12.4s ──── EXECUTION_COMPLETE                                        │
│                                                                       │
│  [Export Events] [Filter Stages] [Zoom Timeline]                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Library: vis-timeline or react-chrono
- Interactive: Click event to see full details
- Filterable: Show/hide specific stages
- Zoomable: Focus on planning vs execution

### 4.3 Graph Structure Visualization

**Purpose**: Show compiled StateGraph structure with execution path

**Visual Design**:
```
┌─────────────────────────────────────────────────────────┐
│ StateGraph Structure                    Status: Running │
├─────────────────────────────────────────────────────────┤
│                                                           │
│          ┌─────────┐                                     │
│          │  START  │                                     │
│          │  (✓)    │                                     │
│          └────┬────┘                                     │
│               │                                           │
│               │ DIRECT                                    │
│               ↓                                           │
│          ┌─────────┐                                     │
│          │ ROUTER  │                                     │
│          │  (✓)    │                                     │
│          └────┬────┘                                     │
│               │                                           │
│         ┌─────┴──────┐                                   │
│         │            │                                    │
│    "research"   "analysis"                               │
│         │            │                                    │
│         ↓            ↓                                    │
│   ┌──────────┐  ┌──────────┐                            │
│   │Researcher│  │ Analyst  │                             │
│   │   (●)    │  │   ( )    │                             │
│   └────┬─────┘  └────┬─────┘                            │
│        │             │                                    │
│        └──────┬──────┘                                   │
│               │                                           │
│               ↓                                           │
│          ┌─────────┐                                     │
│          │   END   │                                     │
│          │   ( )   │                                     │
│          └─────────┘                                     │
│                                                           │
│  Legend:                                                 │
│  (✓) Completed  (●) Running  ( ) Pending                │
│  ─── Direct Edge  ⋯⋯ Conditional Edge                   │
│                                                           │
│  [Export Graph] [View State] [Step Through]             │
└─────────────────────────────────────────────────────────┘
```

**Implementation**:
- Library: Cytoscape.js or Mermaid (for static), React Flow (for interactive)
- Node colors: Blue (START), Green (AGENT), Yellow (ROUTER), Red (END)
- Edge styles: Solid (DIRECT), Dashed (CONDITIONAL)
- Highlight: Current executing node in animated border

### 4.4 Troubleshooting Panel

**Purpose**: Show errors, recovery strategies, and debugging info

**Visual Design**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Troubleshooting & Recovery                           Status: OK │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ ⚠️  Fallback Activated (1 recovery action)                      │
│                                                                   │
│ [09:30:15] ERROR: ToT graph planning failed                     │
│ ├─ Error Type: ConnectionError                                  │
│ ├─ Message: "LLM API timeout after 30s"                         │
│ ├─ Location: orchestrator/planning/tot_graph_planner.py:354     │
│ └─ Stack: [View Full Trace]                                     │
│                                                                   │
│ [09:30:15] RECOVERY: Falling back to sequential planning        │
│ ├─ Strategy: GRAPH_PLANNING_FAILED                              │
│ ├─ Confidence: 80%                                               │
│ ├─ Quality Degradation: 40%                                      │
│ └─ Actions:                                                      │
│    1. Generate fallback graph from agent catalog                │
│    2. Create simple sequential workflow                          │
│    3. Continue with PraisonAI backend                            │
│                                                                   │
│ [09:30:16] SUCCESS: Recovery completed                          │
│ ├─ Fallback Graph: Sequential, 3 nodes, 2 edges                 │
│ ├─ Backend: PraisonAI Legacy                                    │
│ └─ Impact: Reduced parallelization, no conditional routing      │
│                                                                   │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Recovery Decision Tree                                     │  │
│ ├───────────────────────────────────────────────────────────┤  │
│ │                                                             │  │
│ │  ToT Graph Planning                                        │  │
│ │         │                                                   │  │
│ │         ├─ SUCCESS ──→ Compile StateGraph                  │  │
│ │         │                                                   │  │
│ │         └─ FAIL ──→ Fallback: Sequential Graph            │  │
│ │                           │                                 │  │
│ │                           ├─ SUCCESS ──→ Continue          │  │
│ │                           │                                 │  │
│ │                           └─ FAIL ──→ Fallback: PraisonAI │  │
│ │                                             │               │  │
│ │                                             ├─ SUCCESS ──→ Continue│
│ │                                             │               │  │
│ │                                             └─ FAIL ──→ CRITICAL ERROR│
│ │                                                             │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│ [View Logs] [Retry Original] [Export Debug Info]                │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Real-time log stream with severity levels
- Collapsible error details
- Recovery action timeline
- Decision tree diagram (Mermaid or D3.js)

### 4.5 Performance Metrics Dashboard

**Purpose**: Show resource usage and efficiency metrics

**Visual Design**:
```
┌─────────────────────────────────────────────────────────────┐
│ Performance Metrics                                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ Planning Phase                                               │
│ ├─ Total Duration: 8.8s                                      │
│ ├─ ToT Steps: 2                                              │
│ ├─ Candidates Generated: 9                                   │
│ ├─ Candidates Evaluated: 9                                   │
│ └─ Candidates Selected: 1                                    │
│                                                               │
│ Compilation Phase                                            │
│ ├─ Duration: 0.5s                                            │
│ ├─ Nodes Compiled: 5/5 (100%)                                │
│ ├─ Edges Compiled: 4/4 (100%)                                │
│ └─ Validation Errors: 0                                      │
│                                                               │
│ Execution Phase                                              │
│ ├─ Duration: 2.4s                                            │
│ ├─ Nodes Executed: 4                                         │
│ ├─ Agents Completed: 2                                       │
│ └─ Iterations Used: 2/10                                     │
│                                                               │
│ Token Usage                                                  │
│ ├─ Planning: 2,450 tokens                                    │
│ ├─ Execution: 1,820 tokens                                   │
│ └─ Total: 4,270 tokens                                       │
│                                                               │
│ Score Progression                                            │
│ Step 0: avg=7.3 max=8.5 ███████▌                            │
│ Step 1: avg=7.8 max=9.1 ████████                            │
│                                                               │
│ [Export Metrics] [Compare Runs]                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Real-time metric updates
- Score progression chart (Chart.js or Recharts)
- Token usage breakdown
- Historical comparison (if multi-run)

---

## 5. Code Instrumentation Strategy

### 5.1 Event Emitter Implementation

**New Module**: `orchestrator/planning/event_emitter.py`

```python
"""
Event emission system for UI visualization.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import json

class EventStage(Enum):
    PLANNING_INIT = "planning_init"
    TOT_GENERATION = "tot_generation"
    TOT_EVALUATION = "tot_evaluation"
    TOT_SELECTION = "tot_selection"
    PLANNING_COMPLETE = "planning_complete"
    GRAPH_VALIDATION = "graph_validation"
    GRAPH_COMPILATION = "graph_compilation"
    NODE_EXECUTION = "node_execution"
    ERROR_RECOVERY = "error_recovery"

class EventStatus(Enum):
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"

class PlanningEventEmitter:
    """Central event emitter for planning/execution events."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.callbacks: List[Callable] = []

    def start_session(self):
        """Start a new planning session."""
        self.start_time = datetime.now()
        self.events = []

    def emit(self, stage: EventStage, status: EventStatus, details: Dict[str, Any]) -> None:
        """Emit an event to all registered callbacks."""
        if not self.start_time:
            self.start_time = datetime.now()

        event = {
            "event_id": f"{stage.value}_{len(self.events)}",
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": int((datetime.now() - self.start_time).total_seconds() * 1000),
            "stage": stage.value,
            "status": status.value,
            "details": details
        }

        self.events.append(event)

        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                # Don't let callback errors stop event emission
                print(f"Event callback error: {e}")

    def register_callback(self, callback: Callable) -> None:
        """Register a callback function to receive events."""
        self.callbacks.append(callback)

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events emitted during this session."""
        return self.events.copy()

    def export_json(self, filepath: str) -> None:
        """Export events to JSON file for offline analysis."""
        with open(filepath, 'w') as f:
            json.dump({
                "session_start": self.start_time.isoformat() if self.start_time else None,
                "total_events": len(self.events),
                "events": self.events
            }, f, indent=2)

# Global singleton instance
_event_emitter: Optional[PlanningEventEmitter] = None

def get_event_emitter() -> PlanningEventEmitter:
    """Get the global event emitter instance."""
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = PlanningEventEmitter()
    return _event_emitter
```

### 5.2 Instrumentation Points

**File: `orchestrator/planning/tot_planner.py`**

Add at line 406 (before ToT solver):
```python
from .event_emitter import get_event_emitter, EventStage, EventStatus

emitter = get_event_emitter()
emitter.start_session()
emitter.emit(EventStage.PLANNING_INIT, EventStatus.STARTED, {
    "agent_count": len(agent_catalog),
    "backend": settings.backend,
    "max_steps": settings.max_steps,
    "memory_provider": memory_config.provider.value if memory_config else "none"
})
```

Add at line 432 (wrapping ToT solver):
```python
try:
    emitter.emit(EventStage.TOT_GENERATION, EventStatus.STARTED, {
        "step": 0,
        "generation_mode": args.method_generate
    })

    plans, info = solve(args, task, idx=0, to_print=False)

    # Emit events for each ToT step
    for step_info in info.get("steps", []):
        step = step_info["step"]

        emitter.emit(EventStage.TOT_GENERATION, EventStatus.COMPLETED, {
            "step": step,
            "generated_candidates": len(step_info["new_ys"])
        })

        emitter.emit(EventStage.TOT_EVALUATION, EventStatus.COMPLETED, {
            "step": step,
            "scores": step_info["values"],
            "score_range": [min(step_info["values"]), max(step_info["values"])]
        })

        emitter.emit(EventStage.TOT_SELECTION, EventStatus.COMPLETED, {
            "step": step,
            "selected_count": len(step_info["select_new_ys"]),
            "pruned_count": len(step_info["new_ys"]) - len(step_info["select_new_ys"])
        })

    emitter.emit(EventStage.PLANNING_COMPLETE, EventStatus.COMPLETED, {
        "total_steps": len(info.get("steps", [])),
        "best_score": max(step_info["values"]) if info.get("steps") else 0,
        "plan_length": len(assignments)
    })

except Exception as exc:
    emitter.emit(EventStage.PLANNING_COMPLETE, EventStatus.FAILED, {
        "error": str(exc),
        "error_type": type(exc).__name__
    })
    logger.exception("ToT planner failed during solve()")
    raise
```

**File: `orchestrator/planning/graph_compiler.py`**

Add at line 74 (start of compilation):
```python
from .event_emitter import get_event_emitter, EventStage, EventStatus

emitter = get_event_emitter()
emitter.emit(EventStage.GRAPH_COMPILATION, EventStatus.STARTED, {
    "graph_name": graph_spec.name,
    "nodes_count": len(graph_spec.nodes),
    "edges_count": len(graph_spec.edges)
})
```

Add at line 118 (each node compilation):
```python
node_start = datetime.now()
try:
    node_function = self._create_node_function(node_spec, agent_config_map)
    workflow.add_node(node_spec.name, node_function)

    duration_ms = int((datetime.now() - node_start).total_seconds() * 1000)
    emitter.emit(EventStage.GRAPH_COMPILATION, EventStatus.IN_PROGRESS, {
        "operation": "node_compiled",
        "node_name": node_spec.name,
        "node_type": node_spec.type.value,
        "duration_ms": duration_ms
    })
except Exception as e:
    duration_ms = int((datetime.now() - node_start).total_seconds() * 1000)
    emitter.emit(EventStage.GRAPH_COMPILATION, EventStatus.FAILED, {
        "operation": "node_compilation_failed",
        "node_name": node_spec.name,
        "error": str(e),
        "duration_ms": duration_ms
    })
    raise
```

**File: `orchestrator/cli/graph_adapter.py`**

Add at line 116 (fallback detection):
```python
from ..planning.event_emitter import get_event_emitter, EventStage, EventStatus

emitter = get_event_emitter()
emitter.emit(EventStage.ERROR_RECOVERY, EventStatus.STARTED, {
    "original_backend": self.backend_type,
    "error": str(e),
    "fallback_backend": "praisonai" if self.backend_info["praisonai"]["available"] else "none"
})

# ... existing fallback logic ...

emitter.emit(EventStage.ERROR_RECOVERY, EventStatus.RECOVERED, {
    "recovery_successful": True,
    "fallback_backend": "praisonai"
})
```

### 5.3 CLI Output Integration

**File: `orchestrator/cli/main.py`**

Add callback registration for real-time CLI output:

```python
from orchestrator.planning.event_emitter import get_event_emitter, EventStage

def cli_event_callback(event: Dict[str, Any]):
    """Print event to CLI in real-time."""
    stage = event["stage"]
    status = event["status"]
    elapsed = event["elapsed_ms"] / 1000.0

    # Planning events
    if stage == "tot_generation" and status == "completed":
        step = event["details"]["step"]
        count = event["details"]["generated_candidates"]
        print(f"[{elapsed:.1f}s] 🧠 ToT Step {step}: Generated {count} candidates")

    elif stage == "tot_evaluation" and status == "completed":
        scores = event["details"]["scores"]
        print(f"[{elapsed:.1f}s] 📊 Evaluation: scores {min(scores):.1f}-{max(scores):.1f}")

    elif stage == "planning_complete" and status == "completed":
        steps = event["details"]["total_steps"]
        score = event["details"]["best_score"]
        print(f"[{elapsed:.1f}s] ✅ Planning complete: {steps} steps, score {score:.1f}")

    # Compilation events
    elif stage == "graph_compilation" and "node_compiled" in event["details"].get("operation", ""):
        node = event["details"]["node_name"]
        print(f"[{elapsed:.1f}s] 🔧 Compiled node: {node}")

    # Error recovery
    elif stage == "error_recovery":
        if status == "started":
            print(f"[{elapsed:.1f}s] ⚠️  Error detected: {event['details']['error'][:80]}...")
        elif status == "recovered":
            backend = event["details"]["fallback_backend"]
            print(f"[{elapsed:.1f}s] 🔄 Recovered: using {backend}")

# In run_chat() function, before router execution:
emitter = get_event_emitter()
emitter.register_callback(cli_event_callback)
```

---

## 6. Example UI Output Scenarios

### 6.1 Successful Planning Scenario

**Console Output**:
```
» Design authentication system with JWT

[0.1s] 🧠 ToT Step 0: Generated 3 candidates
[1.9s] 📊 Evaluation: scores 6.1-8.5
[2.0s] 🔍 Selection: kept 2, pruned 1
[2.1s] 🧠 ToT Step 1: Generated 6 candidates
[4.1s] 📊 Evaluation: scores 6.3-9.1
[4.2s] 🔍 Selection: kept 1, pruned 5
[4.3s] ✅ Planning complete: 2 steps, score 9.1
[4.8s] 🔧 Compiled node: start
[4.9s] 🔧 Compiled node: router_node
[5.0s] 🔧 Compiled node: agent_researcher
[5.1s] 🔧 Compiled node: agent_analyst
[5.2s] 🔧 Compiled node: end
[5.3s] 🚀 Executing StateGraph...
[7.1s] ✓ Researcher: Found 15 sources on JWT authentication
[7.4s] ✓ Analyst: Synthesized implementation strategy
[7.5s] ✅ Execution complete

Final output:
JWT authentication implementation strategy based on 15 sources...
```

**Timeline Visualization** (shows 8 events, 7.5s total)

**Thought Tree** (shows 2 steps, 9 candidates, 1 final plan)

### 6.2 Fallback Recovery Scenario

**Console Output**:
```
» Analyze system performance bottlenecks

[0.1s] 🧠 ToT Step 0: Generated 3 candidates
[30.5s] ⚠️  Error detected: LLM API timeout after 30s
[30.5s] 🔄 Recovering: falling back to sequential planning
[31.0s] ✅ Fallback planning complete: 3 steps (sequential)
[31.5s] 🔧 Compiled node: start
[31.6s] 🔧 Compiled node: agent_researcher (sequential)
[31.7s] 🔧 Compiled node: agent_analyst (sequential)
[31.8s] 🔧 Compiled node: agent_standards (sequential)
[31.9s] 🔧 Compiled node: end
[32.0s] ⚠️  Using PraisonAI backend (StateGraph compilation skipped)
[32.1s] 🚀 Executing workflow...
[35.2s] ✓ Researcher: Performance data collected
[37.5s] ✓ Analyst: Bottleneck analysis complete
[39.1s] ✓ StandardsAgent: Recommendations provided
[39.2s] ✅ Execution complete (degraded mode)

⚠️  Quality Note: Sequential execution used (no parallelization)

Final output:
Performance bottleneck analysis based on collected data...
```

**Troubleshooting Panel** (shows error, recovery path, impact assessment)

---

## 7. Implementation Checklist

### Phase 1: Event Infrastructure
- [ ] Create `event_emitter.py` module
- [ ] Define event schemas (TypeScript interfaces for frontend)
- [ ] Implement global emitter singleton
- [ ] Add callback registration system
- [ ] Test event emission without UI

### Phase 2: Instrumentation
- [ ] Instrument `tot_planner.py` (lines 406, 432)
- [ ] Instrument `tot_graph_planner.py` (similar points)
- [ ] Instrument `graph_compiler.py` (lines 74, 118, 338)
- [ ] Instrument `graph_adapter.py` (lines 116, 165, 211)
- [ ] Add CLI console output callbacks

### Phase 3: UI Components (Frontend)
- [ ] ToT Reasoning Tree (D3.js/React Flow)
- [ ] Timeline Visualization (vis-timeline)
- [ ] Graph Structure Viewer (Cytoscape.js)
- [ ] Troubleshooting Panel (custom React)
- [ ] Performance Metrics Dashboard (Chart.js)

### Phase 4: Integration & Testing
- [ ] Wire backend event stream to frontend
- [ ] Test with various planning scenarios
- [ ] Test with failure injection
- [ ] Performance optimization (event buffering)
- [ ] Documentation and examples

---

## 8. Technical Requirements

### Backend Requirements
- Python 3.8+
- JSON serialization for events
- Optional: WebSocket server for real-time streaming
- Optional: SQLite for event persistence

### Frontend Requirements
- React 18+ or Vue 3+
- D3.js v7+ for tree/graph visualization
- vis-timeline for timeline component
- Cytoscape.js for graph structure
- WebSocket client (if real-time)
- TypeScript for type safety

### Performance Considerations
- Event buffering (max 100 events in memory)
- Lazy loading for large thought trees (>50 nodes)
- Virtual scrolling for timeline (>100 events)
- Debounce real-time updates (every 100ms)

---

## 9. File Summary

**Key Files for Instrumentation**:

| File | Purpose | Lines to Modify |
|------|---------|-----------------|
| `orchestrator/planning/tot_planner.py` | ToT planning events | 406, 432-454 |
| `orchestrator/planning/tot_graph_planner.py` | Graph planning events | 291-391 |
| `orchestrator/planning/graph_compiler.py` | Compilation events | 74-129, 336-353 |
| `orchestrator/cli/graph_adapter.py` | Fallback/recovery events | 111-123, 165-174, 211-221 |
| `orchestrator/cli/main.py` | CLI output integration | 676-831 (run_chat) |

**New Files to Create**:
- `orchestrator/planning/event_emitter.py` - Event emission system (250 lines)
- `orchestrator/planning/event_schemas.py` - TypeScript type definitions (150 lines)

---

## 10. Next Steps

**Immediate Actions**:
1. Review this specification with frontend developer
2. Implement `event_emitter.py` module
3. Add instrumentation to `tot_planner.py` first (highest value)
4. Create prototype CLI output with events
5. Build simple timeline visualization to validate event structure

**Follow-Up**:
1. Extend instrumentation to compilation phase
2. Build thought tree component
3. Integrate with troubleshooting panel
4. Add performance metrics
5. Full end-to-end testing

---

**Document Version**: 1.0
**Created**: 2025-09-30
**Author**: Root Cause Analyst Agent
**Audience**: Frontend Developer, UI Architect
**Purpose**: Complete technical specification for ToT reasoning visualization
