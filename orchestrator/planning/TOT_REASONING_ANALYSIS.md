# Tree-of-Thought Reasoning Analysis & Troubleshooting Guide

**Document Purpose**: Expose ToT planning reasoning for UI visualization and debugging

**Architecture**: Prompt → ToT Planning → StateGraph Compilation → Agent Execution

---

## 1. ToT Planning Reasoning Flow

### 1.1 Core ToT Algorithm (BFS Method)

The Tree-of-Thought planner uses a Breadth-First Search algorithm to explore planning space:

```
Input: Problem statement, Agent catalog, Settings
→ Initialize: ys = [''] (empty plan candidates)
→ For each step (max_steps iterations):
    1. GENERATION: Generate new plan candidates
       - Sample mode: Generate n_generate_sample variations per candidate
       - Propose mode: Generate proposals based on current state
       - Concatenate: Append new steps to existing candidates

    2. EVALUATION: Score all new candidates
       - Value mode: LLM scores each candidate (0-10)
       - Vote mode: LLM votes for best candidates
       - Cache: Reuse scores for duplicate candidates

    3. SELECTION: Choose best candidates for next iteration
       - Greedy mode: Select top n_select_sample by score
       - Sample mode: Probabilistically select based on scores
       - Prune: Discard low-scoring branches

    4. ITERATION: Continue with selected candidates

→ Output: Best plan (highest scoring final candidate)
```

**Key Insight**: ToT explores multiple planning paths simultaneously, pruning bad paths and expanding good ones.

---

## 2. Thought Generation Process

### 2.1 Standard Prompt Mode

**Location**: `orchestrator/planning/tot_planner.py:180-196`

```python
def _base_prompt(self, x: str, y: str = "") -> str:
    # Components:
    # 1. Role definition: "Eres un orquestador de agentes de ingeniería"
    # 2. Available agents: List with name, role, goal
    # 3. Current plan: What's been decided so far
    # 4. Task: "Propón el siguiente paso del plan"
    # 5. Format constraint: "Agent: <nombre> | Objective: <acción>..."
```

**Reasoning Exposed**:
- **Context Awareness**: LLM sees all available agents and their capabilities
- **Incremental Planning**: Each step builds on previous decisions
- **Format Enforcement**: Structured output ensures parseability
- **No Explanations**: LLM focuses on decisions, not justifications

### 2.2 Chain-of-Thought (CoT) Prompt Mode

**Location**: `orchestrator/planning/tot_planner.py:219-224`

```python
def cot_prompt_wrap(self, x: str, y: str = "") -> str:
    # Enhancement: "Piensa paso a paso antes de decidir"
    # Forces explicit reasoning before decision
```

**Reasoning Exposed**:
- **Explicit Thinking**: LLM shows reasoning process
- **Quality Improvement**: CoT mode typically produces higher-quality plans
- **Debugging Aid**: Reasoning is visible if LLM fails

### 2.3 Graph Planning Mode (Advanced)

**Location**: `orchestrator/planning/tot_graph_planner.py:91-123`

```python
def _base_graph_prompt(self, x: str, y: str = "") -> str:
    # Differences from standard planning:
    # 1. Graph components: nodes, edges, parallel_groups
    # 2. JSON format: Structured component specification
    # 3. Explicit questions: "¿Se puede ejecutar en paralelo?"
    # 4. Architecture focus: "Diseña un grafo StateGraph estructurado"
```

**Advanced Reasoning**:
- **Parallelization Opportunities**: LLM identifies concurrent tasks
- **Conditional Routing**: LLM designs decision nodes
- **Resource Efficiency**: LLM optimizes execution flow
- **Error Recovery**: LLM plans fallback paths

---

## 3. Evaluation and Scoring Process

### 3.1 Value-Based Evaluation

**Location**: `orchestrator/planning/tot_planner.py:231-252`

```python
def value_prompt_wrap(self, x: str, y: str) -> str:
    # Evaluation criteria:
    # - Cobertura del problema (completeness)
    # - Orden lógico (logical ordering)
    # - Claridad (clarity)
    # - Uso adecuado de agentes (agent utilization)
    # Output: {"score": <0-10>, "reason": "..."}

def value_outputs_unwrap(self, x: str, y: str, value_outputs: List[str]) -> float:
    # Scoring logic:
    # 1. Parse JSON response
    # 2. Extract score field
    # 3. Fallback: regex search for numbers if JSON fails
    # 4. Return maximum score across n_evaluate_sample attempts
```

**Reasoning Exposed**:
- **Multi-Criteria Evaluation**: Not just "is it correct?" but quality assessment
- **Numeric Scoring**: Enables objective comparison between candidates
- **Fallback Parsing**: Handles LLM output variations robustly
- **Best-of-N**: Multiple evaluation attempts reduce LLM variance

### 3.2 Graph Planning Evaluation

**Location**: `orchestrator/planning/tot_graph_planner.py:143-172`

```python
def value_prompt_wrap(self, x: str, y: str) -> str:
    # Additional graph criteria:
    # - Estructura del grafo (graph structure quality)
    # - Oportunidades de paralelización (parallelization opportunities)
    # - Manejo de condiciones y routing (control flow)
```

**Advanced Evaluation**:
- **Structural Quality**: Assesses graph topology, not just linear steps
- **Efficiency Focus**: Rewards parallel execution opportunities
- **Control Flow**: Evaluates conditional routing decisions

---

## 4. Selection and Pruning Decisions

### 4.1 Greedy Selection (Default)

**Location**: `tree-of-thought-llm/src/tot/methods/bfs.py:74-76`

```python
if args.method_select == 'greedy':
    select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
    select_new_ys = [new_ys[select_id] for select_id in select_ids]
```

**Pruning Logic**:
- **Deterministic**: Always selects top N candidates
- **Efficient**: No randomness means faster convergence
- **Risk**: May miss creative solutions if early scores are misleading

### 4.2 Sample Selection (Exploratory)

**Location**: `tree-of-thought-llm/src/tot/methods/bfs.py:71-73`

```python
if args.method_select == 'sample':
    ps = np.array(values) / sum(values)
    select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
```

**Exploration Logic**:
- **Probabilistic**: Higher scores = higher probability
- **Diverse**: Can explore lower-scoring but promising paths
- **Use Case**: When creative solutions are needed

---

## 5. Graph Compilation Reasoning

### 5.1 Node Compilation Process

**Location**: `orchestrator/planning/graph_compiler.py:112-154`

```python
def _compile_nodes(self, workflow: StateGraph, graph_spec: StateGraphSpec,
                   agent_config_map: Dict[str, AgentConfig]) -> None:
    # For each node in specification:
    # 1. Identify node type (START, END, AGENT, ROUTER, CONDITION, etc.)
    # 2. Create appropriate node function
    # 3. Register function with StateGraph workflow
    # 4. Cache for reuse
```

**Compilation Reasoning**:
- **Type-Based Dispatch**: Different node types get different implementations
- **Agent Resolution**: Maps agent names to actual agent configurations
- **Function Wrapping**: Converts declarative specs to executable functions
- **Error Isolation**: Each node compilation is independent (partial success possible)

### 5.2 Node Function Creation Logic

**Node Type → Function Mapping**:

```python
START → _create_start_function()
    # Reasoning: Initialize state, add user message, mark entry
    # State updates: current_node, execution_path, messages

END → _create_end_function()
    # Reasoning: Finalize output, aggregate agent results
    # State updates: final_output, completion markers

AGENT → _create_agent_function()
    # Reasoning: Execute agent with context, handle errors, update state
    # State updates: agent_outputs, completed_agents, messages
    # Error handling: Catch exceptions, store in error_state

ROUTER → _create_router_function()
    # Reasoning: Decide next path based on routing strategy
    # Strategies: LLM-based (ask LLM), state-based (check state), rule-based (keywords)
    # State updates: router_decision, current_route

CONDITION → _create_condition_function()
    # Reasoning: Evaluate boolean conditions, store results
    # State updates: condition_results (for edge routing)

PARALLEL → _create_parallel_function()
    # Reasoning: Mark parallel execution start
    # State updates: parallel_execution_active flag

AGGREGATOR → _create_aggregator_function()
    # Reasoning: Combine results from parallel execution
    # State updates: aggregated_result, clear parallel flag
```

### 5.3 Edge Compilation Process

**Location**: `orchestrator/planning/graph_compiler.py:336-368`

```python
def _compile_edges(self, workflow: StateGraph, graph_spec: StateGraphSpec) -> None:
    # For each edge:
    # 1. Identify edge type (DIRECT, CONDITIONAL, PARALLEL, etc.)
    # 2. Add to StateGraph:
    #    - Direct: workflow.add_edge(from, to)
    #    - Conditional: workflow.add_conditional_edges(from, condition_func)
    # 3. Handle failures gracefully (continue with other edges)
```

**Edge Reasoning**:
- **Direct Edges**: Unconditional transitions (A always goes to B)
- **Conditional Edges**: Function returns next node based on state
- **Fallback**: If condition function fails, route to END
- **Partial Success**: Continue compiling even if some edges fail

### 5.4 Entry/Exit Point Resolution

**Location**: `orchestrator/planning/graph_compiler.py:370-387`

```python
def _set_entry_exit_points(self, workflow: StateGraph, graph_spec: StateGraphSpec) -> None:
    # Entry point logic:
    # 1. Use explicit entry_point if specified
    # 2. Fallback: Find START node
    # 3. Fallback: Use first node

    # Exit point logic:
    # 1. Add edges from all exit_points to END
    # 2. Avoid self-reference (end → END)
```

**Resolution Reasoning**:
- **Explicit > Implicit**: Prefer spec-defined entry/exit
- **Smart Defaults**: Find START/END nodes automatically
- **Graph Validity**: Ensure all paths have entry and exit

---

## 6. Troubleshooting Decision Trees

### 6.1 Planning Failure Modes

#### **Failure Mode 1: ToT Package Not Available**

**Detection**: `orchestrator/planning/tot_planner.py:394-397`

```python
if not _TOT_AVAILABLE and not _try_import_tot():
    raise RuntimeError("tree-of-thought-llm package is not available")
```

**Root Causes**:
- Tree-of-thought-llm not installed
- Import path incorrect
- OpenAI shim not found

**Recovery Strategy**:
```python
# In generate_graph_with_tot():
if not _TOT_AVAILABLE:
    logger.warning("ToT not available, falling back to assignments")
    enable_graph_planning = False
    # → Fallback to sequential planning
```

**Diagnosis Steps**:
1. Check if `tree-of-thought-llm/src` exists
2. Verify `sys.path` includes ToT directory
3. Check for `openai/__init__.py` shim file
4. Review import error stack trace

**User Impact**: **MEDIUM** - Fallback to sequential planning works but loses ToT benefits

---

#### **Failure Mode 2: ToT Solver Exception**

**Detection**: `orchestrator/planning/tot_planner.py:432-436`

```python
try:
    plans, info = solve(args, task, idx=0, to_print=False)
except Exception as exc:
    logger.exception("ToT planner failed during solve()")
    raise
```

**Root Causes**:
- LLM API failure (rate limit, timeout, auth error)
- Invalid prompt format
- Task configuration error (bad max_steps, n_samples)

**Recovery Strategy**:
```python
# In tot_graph_planner.py:351-356
try:
    plans, info = solve(args, task, idx=0, to_print=False)
except Exception as exc:
    logger.exception("ToT graph planner failed")
    # → Fallback to sequential graph
    return _generate_fallback_graph(prompt, recall_snippets, agent_catalog)
```

**Diagnosis Steps**:
1. Check LLM API credentials (OPENAI_API_KEY)
2. Review LLM API logs for rate limiting
3. Inspect prompt length (may exceed token limits)
4. Verify task configuration (settings.max_steps, n_generate_sample)
5. Check for empty agent_catalog

**User Impact**: **HIGH** - Planning fails completely without fallback

---

#### **Failure Mode 3: Plan Parsing Failure**

**Detection**: `orchestrator/planning/tot_planner.py:355-383`

```python
pattern = re.compile(
    r"Agent:\s*(?P<agent>[^|]+)\|\s*Objective:\s*(?P<objective>[^|]+)...",
    re.IGNORECASE,
)
# If regex doesn't match, assignment not created
```

**Root Causes**:
- LLM doesn't follow format instructions
- CoT mode outputs too much reasoning text
- LLM outputs malformed JSON (in graph mode)

**Recovery Strategy**:
```python
# In tot_graph_planner.py:465-472
try:
    component = json.loads(line)
    # ... process component
except json.JSONDecodeError:
    logger.warning(f"Failed to parse graph component: {line}")
    continue  # Skip invalid lines

# If no valid components found:
if not components["nodes"]:
    logger.info("No graph components found, creating sequential fallback")
    return _create_sequential_graph_from_agents(agent_catalog, graph_name)
```

**Diagnosis Steps**:
1. Enable debug logging to see raw LLM output
2. Check if LLM used correct format
3. Review prompt clarity (are format instructions clear?)
4. Try switching prompt_style ('cot' → 'standard' or vice versa)
5. Increase n_generate_sample for more format variations

**User Impact**: **MEDIUM** - Fallback creates simple sequential graph

---

#### **Failure Mode 4: Empty Agent Catalog**

**Detection**: `orchestrator/planning/tot_planner.py:400-401`

```python
if not agent_catalog:
    raise ValueError("agent_catalog must not be empty")
```

**Root Causes**:
- No agents configured in orchestrator
- Agent loading failed earlier in pipeline
- Filtering removed all agents

**Recovery Strategy**: **NONE** (immediate failure)

**Diagnosis Steps**:
1. Check orchestrator initialization
2. Verify agent configuration files exist
3. Review agent filtering logic (if any)
4. Check for agent factory errors

**User Impact**: **CRITICAL** - Cannot plan without agents

---

### 6.2 Compilation Failure Modes

#### **Failure Mode 5: Graph Validation Errors**

**Detection**: `orchestrator/planning/graph_compiler.py:77-80`

```python
if validation_mode:
    validation_errors = graph_spec.validate()
    if validation_errors:
        raise GraphCreationError(f"Graph validation failed: {validation_errors}")
```

**Root Causes**:
- Missing nodes referenced in edges
- Duplicate node names
- No entry/exit points
- Circular dependencies without termination

**Recovery Strategy**:
```python
# In graph_specifications.py (validate method):
errors = graph_spec.validate()
if errors:
    logger.warning(f"Graph validation errors: {errors}")
    _auto_fix_graph(graph_spec)  # Attempt automatic fixes
```

**Diagnosis Steps**:
1. Review graph_spec.validate() output
2. Check for orphaned edges (from_node or to_node not in nodes)
3. Verify entry_point and exit_points are valid node names
4. Visualize graph structure to identify issues

**User Impact**: **MEDIUM** - Auto-fix may resolve, else need manual correction

---

#### **Failure Mode 6: Agent Not Found**

**Detection**: `orchestrator/planning/graph_compiler.py:203-205`

```python
agent_name = node_spec.agent
if not agent_name or agent_name not in agent_config_map:
    raise GraphCreationError(f"Agent '{agent_name}' not found in configuration")
```

**Root Causes**:
- ToT planner generated invalid agent name
- Agent configuration mismatch between planning and compilation
- Typo in agent name

**Recovery Strategy**: **NONE** (immediate failure)

**Diagnosis Steps**:
1. Compare node_spec.agent names to agent_config_map keys
2. Check for case sensitivity issues
3. Review ToT prompt (does it list correct agent names?)
4. Verify agent_catalog passed to planner matches compilation catalog

**User Impact**: **HIGH** - Cannot compile graph without valid agents

---

#### **Failure Mode 7: Node Compilation Exception**

**Detection**: `orchestrator/planning/graph_compiler.py:127-129`

```python
except Exception as e:
    logger.error(f"Failed to compile node {node_spec.name}: {str(e)}")
    raise GraphCreationError(f"Node compilation failed for {node_spec.name}: {str(e)}")
```

**Root Causes**:
- Unsupported node type
- Agent creation failure
- Invalid node configuration (bad timeout, missing objective)

**Recovery Strategy**: **NONE** (fail fast on node compilation)

**Diagnosis Steps**:
1. Check node_spec.type is valid NodeType enum
2. Verify agent_factory can create agent
3. Review node_spec fields for invalid values
4. Check for missing required fields (agent for AGENT type)

**User Impact**: **HIGH** - One bad node fails entire graph compilation

---

#### **Failure Mode 8: Edge Compilation Failure**

**Detection**: `orchestrator/planning/graph_compiler.py:351-353`

```python
except Exception as e:
    logger.error(f"Failed to compile edge {edge_spec.from_node}->{edge_spec.to_node}: {str(e)}")
    # Continue with other edges rather than failing completely
```

**Root Causes**:
- from_node or to_node doesn't exist (already compiled in nodes)
- Invalid condition function
- StateGraph API error

**Recovery Strategy**: **PARTIAL SUCCESS** (skip bad edge, continue with others)

**Diagnosis Steps**:
1. Verify from_node and to_node were compiled as nodes
2. Check condition function syntax if conditional edge
3. Review StateGraph API documentation for edge constraints
4. Check for duplicate edges (may cause API errors)

**User Impact**: **LOW-MEDIUM** - Graph may execute with missing edges (could cause unexpected behavior)

---

### 6.3 Execution Failure Modes

#### **Failure Mode 9: Agent Execution Error**

**Detection**: `orchestrator/planning/graph_compiler.py:241-246`

```python
except Exception as e:
    error_msg = f"Agent {agent_name} failed in node {node_spec.name}: {str(e)}"
    state.error_state = error_msg
    state.node_outputs[node_spec.name] = error_msg
    logger.error(error_msg)
    return state  # Continue execution despite error
```

**Root Causes**:
- Agent LLM API failure
- Tool execution error (if agent uses tools)
- Invalid task description
- Agent timeout

**Recovery Strategy**: **GRACEFUL DEGRADATION**
- Store error in state
- Mark node as failed
- Continue to next node (if graph allows)

**Diagnosis Steps**:
1. Check agent LLM API logs
2. Review task_description passed to agent
3. Verify tool availability and credentials
4. Check for agent timeout configuration
5. Inspect state.agent_outputs for partial results

**User Impact**: **VARIABLE** - Depends on whether subsequent nodes can proceed without this agent's output

---

#### **Failure Mode 10: Routing Decision Failure**

**Detection**: `orchestrator/planning/graph_compiler.py:250-270`

```python
# If routing logic fails, fallback to default route
if node_spec.routing_strategy == RoutingStrategy.LLM_BASED:
    routing_result = self._llm_based_routing(node_spec, state)
    # If LLM fails, fallback implemented in method
```

**Root Causes**:
- LLM-based routing: LLM API failure
- State-based routing: Missing state fields
- Rule-based routing: No matching rules

**Recovery Strategy**: **DEFAULT ROUTE**
```python
# In _create_edge_condition():
if edge_spec.condition:
    condition_result = edge_spec.condition.evaluate(state)
    if condition_result:
        return edge_spec.to_node
    else:
        return END  # Safe default: exit graph
```

**Diagnosis Steps**:
1. Check routing_strategy configuration
2. Verify LLM credentials for LLM_BASED routing
3. Inspect state fields for STATE_BASED routing
4. Review rule keywords for RULE_BASED routing
5. Test condition.evaluate() logic directly

**User Impact**: **MEDIUM** - May skip important nodes or exit prematurely

---

## 7. Reasoning Exposure for UI Visualization

### 7.1 ToT Thought Tree Visualization

**Data Structure** (from `bfs.py:83`):
```python
infos.append({
    'step': step,              # Current planning step (0 to max_steps-1)
    'x': x,                    # Original problem statement
    'ys': ys,                  # Current candidate plans (pre-generation)
    'new_ys': new_ys,          # Generated candidate plans (post-generation)
    'values': values,          # Evaluation scores for each new candidate
    'select_new_ys': select_new_ys  # Selected candidates (post-pruning)
})
```

**UI Visualization Recommendations**:

1. **Tree View**:
   ```
   Problem: "Create auth system"

   Step 0 (Root):
   └─ [empty plan]
       ├─ Generate 3 candidates
       ├─ Evaluate scores: [8.5, 7.2, 6.1]
       └─ Select top 2: [candidate_1, candidate_2]

   Step 1 (Branch):
   ├─ Candidate 1 (score: 8.5)
   │   ├─ "Agent: Security | Objective: Design auth flow..."
   │   ├─ Generate 3 extensions
   │   ├─ Scores: [9.1, 8.8, 7.5]
   │   └─ Select: [extension_1]  ✓ FINAL PLAN
   │
   └─ Candidate 2 (score: 7.2)
       ├─ "Agent: Backend | Objective: Implement JWT..."
       ├─ Generate 3 extensions
       ├─ Scores: [7.8, 7.0, 6.3]
       └─ PRUNED (score < 8.0)
   ```

2. **Timeline View**:
   ```
   [Step 0] → [Step 1] → [Step 2] → [Final]
      3         6         4          1
   candidates  candidates  candidates  plan
   ```

3. **Score Heatmap**:
   ```
   Step  | Candidates Generated | Score Range  | Pruned
   ------|---------------------|--------------|--------
   0     | 3                   | 6.1 - 8.5    | 1
   1     | 6                   | 6.3 - 9.1    | 4
   2     | 4                   | 7.5 - 9.3    | 3
   ```

### 7.2 Planning Stages Timeline

**Event Types**:

```python
# Event schema for UI consumption
class PlanningEvent:
    timestamp: datetime
    stage: Literal["generation", "evaluation", "selection", "compilation", "execution"]
    status: Literal["started", "in_progress", "completed", "failed"]
    details: Dict[str, Any]
    metadata: Dict[str, Any]

# Example events:
[
    {
        "timestamp": "2025-09-30T10:30:00",
        "stage": "generation",
        "status": "started",
        "details": {
            "step": 0,
            "current_candidates": 1,
            "generation_mode": "sample",
            "n_generate_sample": 3
        },
        "metadata": {"prompt_length": 1024, "temperature": 0.7}
    },
    {
        "timestamp": "2025-09-30T10:30:05",
        "stage": "generation",
        "status": "completed",
        "details": {
            "step": 0,
            "generated_candidates": 3,
            "tokens_used": 512
        }
    },
    {
        "timestamp": "2025-09-30T10:30:05",
        "stage": "evaluation",
        "status": "started",
        "details": {
            "candidates_to_evaluate": 3,
            "evaluation_mode": "value",
            "n_evaluate_sample": 2
        }
    },
    {
        "timestamp": "2025-09-30T10:30:10",
        "stage": "evaluation",
        "status": "completed",
        "details": {
            "scores": [8.5, 7.2, 6.1],
            "evaluation_criteria": [
                "problem_coverage",
                "logical_order",
                "clarity",
                "agent_utilization"
            ]
        }
    },
    {
        "timestamp": "2025-09-30T10:30:10",
        "stage": "selection",
        "status": "completed",
        "details": {
            "selection_mode": "greedy",
            "n_select_sample": 2,
            "selected_indices": [0, 1],
            "pruned_count": 1
        }
    },
    {
        "timestamp": "2025-09-30T10:30:45",
        "stage": "compilation",
        "status": "started",
        "details": {
            "graph_spec_name": "tot_graph_3_agents",
            "nodes_count": 5,
            "edges_count": 4,
            "parallel_groups": 0
        }
    },
    {
        "timestamp": "2025-09-30T10:30:47",
        "stage": "compilation",
        "status": "completed",
        "details": {
            "compiled_nodes": 5,
            "compiled_edges": 4,
            "validation_errors": []
        }
    },
    {
        "timestamp": "2025-09-30T10:30:50",
        "stage": "execution",
        "status": "started",
        "details": {
            "entry_point": "start",
            "initial_state": {"user_prompt": "...", "max_iterations": 10}
        }
    }
]
```

### 7.3 Graph Compilation Steps Visualization

**Compilation Pipeline**:

```
1. VALIDATION
   ├─ Check node references
   ├─ Verify entry/exit points
   ├─ Validate edge connections
   └─ Status: PASS / FAIL (with error list)

2. NODE COMPILATION
   For each node in graph_spec.nodes:
   ├─ Node: "start" (START type)
   │   ├─ Create: _create_start_function()
   │   ├─ Register: workflow.add_node("start", start_function)
   │   └─ Status: ✓ COMPILED
   ├─ Node: "router_node" (ROUTER type)
   │   ├─ Create: _create_router_function()
   │   ├─ Strategy: LLM_BASED
   │   ├─ Register: workflow.add_node("router_node", router_function)
   │   └─ Status: ✓ COMPILED
   ├─ Node: "agent_researcher" (AGENT type)
   │   ├─ Resolve: agent="Researcher" → agent_config_map["Researcher"]
   │   ├─ Cache: Create LangChainAgent and cache in _agent_cache
   │   ├─ Create: _create_agent_function(agent=Researcher)
   │   ├─ Register: workflow.add_node("agent_researcher", agent_function)
   │   └─ Status: ✓ COMPILED
   └─ Node: "end" (END type)
       ├─ Create: _create_end_function()
       ├─ Register: workflow.add_node("end", end_function)
       └─ Status: ✓ COMPILED

3. EDGE COMPILATION
   For each edge in graph_spec.edges:
   ├─ Edge: start → router_node (DIRECT)
   │   ├─ Add: workflow.add_edge("start", "router_node")
   │   └─ Status: ✓ COMPILED
   ├─ Edge: router_node → agent_researcher (CONDITIONAL)
   │   ├─ Create: condition_func = _create_edge_condition(edge_spec)
   │   ├─ Add: workflow.add_conditional_edges("router_node", condition_func)
   │   └─ Status: ✓ COMPILED
   └─ Edge: agent_researcher → end (DIRECT)
       ├─ Add: workflow.add_edge("agent_researcher", "end")
       └─ Status: ✓ COMPILED

4. ENTRY/EXIT SETUP
   ├─ Entry: workflow.set_entry_point("start")
   ├─ Exit: workflow.add_edge("end", END)
   └─ Status: ✓ CONFIGURED

5. PARALLEL GROUPS
   ├─ Group: parallel_group_1 (nodes: [node_a, node_b])
   │   └─ Status: ⚠️  NOTED (LangGraph parallel handling not fully implemented)
   └─ Status: PARTIAL

6. FINALIZATION
   ├─ Compile: compiled_graph = workflow.compile()
   ├─ Cache: _compiled_graphs["tot_graph_3_agents"] = compiled_graph
   └─ Status: ✓ READY FOR EXECUTION
```

### 7.4 Execution Monitoring Visualization

**Real-Time Execution State**:

```python
# State snapshot for UI
class ExecutionSnapshot:
    current_node: str             # "agent_researcher"
    execution_path: List[str]     # ["start", "router_node", "agent_researcher"]
    completed_agents: List[str]   # ["Researcher"]
    agent_outputs: Dict[str, str] # {"Researcher": "Research results..."}
    current_iteration: int        # 3
    max_iterations: int           # 10
    errors: List[Dict]            # []
    router_decision: Dict         # {"route": "research", "confidence": 0.92}
```

**Live Progress Display**:

```
Execution Progress: 60% (3/5 nodes completed)

[✓] start          (0.2s)  → Workflow initialized
[✓] router_node    (1.5s)  → Routed to: research (confidence: 0.92)
[●] agent_researcher (currently running, 3.2s elapsed)
[ ] aggregator
[ ] end

Agent Outputs:
  Researcher: "Found 15 relevant sources on authentication patterns..."
  (waiting for agent to complete)

Current State:
  - Iteration: 3/10
  - Messages: 2 (1 Human, 1 AI)
  - Memory context: 3 recall items
  - Errors: None
```

---

## 8. Code Examples: Reasoning Extraction

### 8.1 Extracting ToT Thought Tree

```python
# Location: orchestrator/planning/tot_planner.py
def generate_plan_with_tot_debug(
    prompt: str,
    recall_snippets: Sequence[str],
    agent_catalog: Sequence[AgentConfig],
    settings: Optional[PlanningSettings] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> Dict[str, Any]:
    """Extended version that returns detailed thought tree for UI visualization."""

    # ... standard setup ...

    # Execute ToT solver
    plans, info = solve(args, task, idx=0, to_print=False)

    # Extract thought tree from info
    thought_tree = []
    for step_info in info.get("steps", []):
        step_data = {
            "step": step_info["step"],
            "generation": {
                "input_candidates": step_info["ys"],
                "generated_candidates": step_info["new_ys"],
                "generation_mode": args.method_generate,
                "samples_per_candidate": args.n_generate_sample
            },
            "evaluation": {
                "scores": step_info["values"],
                "evaluation_mode": args.method_evaluate,
                "evaluation_samples": args.n_evaluate_sample,
                "score_range": (min(step_info["values"]), max(step_info["values"]))
            },
            "selection": {
                "selected_candidates": step_info["select_new_ys"],
                "selection_mode": args.method_select,
                "pruned_count": len(step_info["new_ys"]) - len(step_info["select_new_ys"])
            },
            "reasoning": {
                "problem_statement": step_info["x"],
                "best_score": max(step_info["values"]),
                "avg_score": sum(step_info["values"]) / len(step_info["values"])
            }
        }
        thought_tree.append(step_data)

    return {
        "assignments": parse_plan_to_assignments(plans[0]),
        "metadata": {
            "raw_plan": plans[0],
            "tot_usage": gpt_usage(settings.backend),
            "thought_tree": thought_tree,  # Full reasoning trace
            "planning_metrics": {
                "total_steps": len(thought_tree),
                "total_candidates_generated": sum(len(s["generation"]["generated_candidates"]) for s in thought_tree),
                "total_candidates_evaluated": sum(len(s["evaluation"]["scores"]) for s in thought_tree),
                "final_score": thought_tree[-1]["reasoning"]["best_score"] if thought_tree else 0
            }
        }
    }
```

### 8.2 Extracting Compilation Reasoning

```python
# Location: orchestrator/planning/graph_compiler.py
class GraphCompiler:
    def compile_graph_spec_with_trace(
        self,
        graph_spec: StateGraphSpec,
        agent_configs: List[AgentConfig],
        validation_mode: bool = True
    ) -> Tuple[StateGraph, Dict[str, Any]]:
        """Compile with detailed trace for debugging and UI visualization."""

        trace = {
            "validation": {},
            "node_compilation": [],
            "edge_compilation": [],
            "entry_exit_setup": {},
            "errors": []
        }

        # VALIDATION STAGE
        if validation_mode:
            validation_errors = graph_spec.validate()
            trace["validation"] = {
                "passed": not validation_errors,
                "errors": validation_errors,
                "timestamp": datetime.now().isoformat()
            }
            if validation_errors:
                trace["errors"].append({
                    "stage": "validation",
                    "error": f"Graph validation failed: {validation_errors}"
                })

        # NODE COMPILATION STAGE
        agent_config_map = {config.name: config for config in agent_configs}
        workflow = StateGraph(OrchestratorState)

        for node_spec in graph_spec.nodes:
            node_trace = {
                "name": node_spec.name,
                "type": node_spec.type.value,
                "start_time": datetime.now().isoformat()
            }
            try:
                node_function = self._create_node_function(node_spec, agent_config_map)
                workflow.add_node(node_spec.name, node_function)

                node_trace.update({
                    "status": "success",
                    "end_time": datetime.now().isoformat(),
                    "details": {
                        "agent": node_spec.agent if node_spec.type == NodeType.AGENT else None,
                        "routing_strategy": node_spec.routing_strategy.value if node_spec.type == NodeType.ROUTER else None
                    }
                })
            except Exception as e:
                node_trace.update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now().isoformat()
                })
                trace["errors"].append({
                    "stage": "node_compilation",
                    "node": node_spec.name,
                    "error": str(e)
                })

            trace["node_compilation"].append(node_trace)

        # EDGE COMPILATION STAGE
        for edge_spec in graph_spec.edges:
            edge_trace = {
                "from": edge_spec.from_node,
                "to": edge_spec.to_node,
                "type": edge_spec.type.value,
                "start_time": datetime.now().isoformat()
            }
            try:
                if edge_spec.type == EdgeType.DIRECT:
                    workflow.add_edge(edge_spec.from_node, edge_spec.to_node)
                elif edge_spec.type == EdgeType.CONDITIONAL:
                    condition_func = self._create_edge_condition(edge_spec)
                    workflow.add_conditional_edges(edge_spec.from_node, condition_func)

                edge_trace.update({
                    "status": "success",
                    "end_time": datetime.now().isoformat()
                })
            except Exception as e:
                edge_trace.update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now().isoformat()
                })
                # Continue with other edges (non-critical failure)

            trace["edge_compilation"].append(edge_trace)

        # ENTRY/EXIT SETUP
        trace["entry_exit_setup"] = {
            "entry_point": graph_spec.entry_point or "auto-detected",
            "exit_points": graph_spec.exit_points,
            "timestamp": datetime.now().isoformat()
        }
        self._set_entry_exit_points(workflow, graph_spec)

        # FINALIZATION
        compiled_graph = workflow
        trace["summary"] = {
            "total_nodes": len(graph_spec.nodes),
            "nodes_compiled": sum(1 for n in trace["node_compilation"] if n["status"] == "success"),
            "total_edges": len(graph_spec.edges),
            "edges_compiled": sum(1 for e in trace["edge_compilation"] if e["status"] == "success"),
            "total_errors": len(trace["errors"]),
            "compilation_successful": len(trace["errors"]) == 0
        }

        return compiled_graph, trace
```

### 8.3 Extracting Execution State for UI

```python
# Location: orchestrator/cli/graph_adapter.py or new monitoring module
class ExecutionMonitor:
    """Monitor StateGraph execution and expose state for UI visualization."""

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None

    def create_snapshot(self, state: OrchestratorState) -> Dict[str, Any]:
        """Create a point-in-time snapshot of execution state."""
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "current_node": state.current_node,
            "execution_path": list(state.execution_path),
            "completed_agents": list(state.completed_agents),
            "agent_outputs": {agent: output[:200] + "..." if len(output) > 200 else output
                             for agent, output in state.agent_outputs.items()},
            "current_iteration": state.current_iteration,
            "max_iterations": state.max_iterations,
            "progress_pct": (state.current_iteration / state.max_iterations) * 100,
            "errors": list(state.errors) if hasattr(state, 'errors') else [],
            "router_decision": state.router_decision,
            "memory_context_length": len(state.memory_context) if state.memory_context else 0,
            "message_count": len(state.messages)
        }

    def start_monitoring(self, compiled_graph: StateGraph, initial_state: OrchestratorState):
        """Execute graph with monitoring."""
        self.start_time = datetime.now()
        self.snapshots = []

        # Create initial snapshot
        self.snapshots.append(self.create_snapshot(initial_state))

        # Execute graph (would need StateGraph API support for step-by-step execution)
        # For now, we'd need to wrap node functions to capture snapshots
        result = compiled_graph.invoke(initial_state)

        # Create final snapshot
        self.snapshots.append(self.create_snapshot(result))

        return result, self.snapshots

    def get_execution_summary(self) -> Dict[str, Any]:
        """Generate summary of entire execution for UI dashboard."""
        if not self.snapshots:
            return {}

        first = self.snapshots[0]
        last = self.snapshots[-1]

        return {
            "total_duration": last["elapsed_time"],
            "nodes_executed": len(last["execution_path"]),
            "agents_completed": len(last["completed_agents"]),
            "iterations_used": last["current_iteration"],
            "errors_count": len(last["errors"]),
            "success": len(last["errors"]) == 0 and last.get("final_output") is not None,
            "snapshots": self.snapshots,
            "timeline": [
                {
                    "time": snap["elapsed_time"],
                    "node": snap["current_node"],
                    "event": f"Executed {snap['current_node']}"
                }
                for snap in self.snapshots
            ]
        }
```

---

## 9. Summary: Making the Invisible Visible

### Key Reasoning Elements to Expose

1. **ToT Thought Generation**:
   - Show all candidate plans generated at each step
   - Display evaluation scores for each candidate
   - Visualize pruning decisions (why candidates were rejected)

2. **Evaluation Criteria**:
   - Expose scoring rubric (coverage, order, clarity, agent utilization)
   - Show individual criterion scores (not just aggregate)
   - Display LLM reasoning (if available in CoT mode)

3. **Graph Compilation**:
   - Step-by-step node compilation with success/failure status
   - Agent resolution (which agent config matched which node)
   - Edge creation logic (direct vs conditional)

4. **Execution Flow**:
   - Real-time node execution progress
   - Agent outputs as they complete
   - Routing decisions with confidence scores
   - Error states with recovery actions

### UI Implementation Recommendations

1. **Interactive Thought Tree**:
   - Expandable tree view showing all ToT steps
   - Click node to see prompt, LLM response, score
   - Highlight selected path vs pruned branches
   - Color-code by score (green=high, yellow=medium, red=low)

2. **Timeline Visualization**:
   - Horizontal timeline with stages: Generation → Evaluation → Selection → Compilation → Execution
   - Show duration for each stage
   - Click stage to see detailed logs

3. **Graph Visualization**:
   - Interactive graph rendering (nodes and edges)
   - Node colors by type (START=blue, AGENT=green, ROUTER=yellow, END=red)
   - Edge labels showing conditions
   - Highlight execution path in real-time

4. **Debugging Panel**:
   - Live log stream
   - Error highlighting
   - State inspector (view OrchestratorState at any point)
   - Replay functionality (step through execution)

5. **Performance Metrics**:
   - Token usage tracking
   - LLM API call count
   - Execution time per node
   - Score trends across ToT steps

---

## 10. Next Steps for Implementation

1. **Instrument Code**:
   - Add event emitters to ToT solver (generation, evaluation, selection)
   - Add trace logging to graph compiler
   - Add state snapshots to execution loop

2. **Define Event Schema**:
   - Standardize event format for UI consumption
   - Create TypeScript types (if using TS frontend)
   - Document event sequence

3. **Build Data Collection Layer**:
   - Implement ExecutionMonitor class
   - Store events in structured format (JSON, SQLite)
   - Provide query API for UI

4. **Create UI Components**:
   - Thought tree component (D3.js or React Flow)
   - Timeline component (vis-timeline)
   - Graph visualization (Cytoscape.js or Mermaid)
   - Log viewer (react-console-log)

5. **Test with Real Scenarios**:
   - Run planning with different complexity levels
   - Trigger failure modes intentionally
   - Validate UI shows correct reasoning

---

## Appendix: Key Files Reference

- **ToT Planning**: `orchestrator/planning/tot_planner.py`
- **Graph Planning**: `orchestrator/planning/tot_graph_planner.py`
- **Graph Compiler**: `orchestrator/planning/graph_compiler.py`
- **Graph Specifications**: `orchestrator/planning/graph_specifications.py`
- **CLI Adapter**: `orchestrator/cli/graph_adapter.py` (lines 280-340)
- **ToT BFS Solver**: `tree-of-thought-llm/src/tot/methods/bfs.py`
- **LangChain Integration**: `orchestrator/integrations/langchain_integration.py`

---

**Document Version**: 1.0
**Last Updated**: 2025-09-30
**Author**: Root Cause Analyst Agent
**Purpose**: UI Visualization Support & Troubleshooting Reference