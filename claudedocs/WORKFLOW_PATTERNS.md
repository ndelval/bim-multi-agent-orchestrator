# Workflow Patterns Guide

Common patterns and best practices for building AI agent workflows with the Orchestrator framework.

---

## Table of Contents

1. [Basic Patterns](#basic-patterns)
2. [Parallel Execution Patterns](#parallel-execution-patterns)
3. [Conditional Routing Patterns](#conditional-routing-patterns)
4. [Memory Integration Patterns](#memory-integration-patterns)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Advanced Patterns](#advanced-patterns)

---

## Basic Patterns

### Pattern 1: Linear Sequential Workflow

**Use Case**: Simple workflows where each step depends on the previous one.

```python
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig

config = OrchestratorConfig(name="LinearWorkflow", process="sequential")

# Agents
config.agents = [
    AgentConfig(name="StepA", role="First Step", goal="Do A", backstory="Expert A", instructions="Execute A"),
    AgentConfig(name="StepB", role="Second Step", goal="Do B", backstory="Expert B", instructions="Execute B"),
    AgentConfig(name="StepC", role="Third Step", goal="Do C", backstory="Expert C", instructions="Execute C"),
]

# Tasks in sequence
config.tasks = [
    TaskConfig(name="task_a", agent="StepA", description="Step A", expected_output="Output A"),
    TaskConfig(name="task_b", agent="StepB", description="Step B", expected_output="Output B", context=["task_a"]),
    TaskConfig(name="task_c", agent="StepC", description="Step C", expected_output="Output C", context=["task_b"]),
]
```

**Execution Flow:**
```
task_a → task_b → task_c
```

**Best For:**
- Simple workflows
- Strict ordering requirements
- Debugging and testing

---

### Pattern 2: Research → Analysis → Report

**Use Case**: Common knowledge work pattern with research, analysis, and documentation phases.

```python
config = OrchestratorConfig(name="ResearchWorkflow")

config.agents = [
    AgentConfig(
        name="Researcher",
        role="Research Specialist",
        goal="Gather comprehensive information",
        backstory="Expert researcher with web search skills",
        instructions="Search web and memory for relevant information",
        tools=["duckduckgo", "graphrag"]
    ),
    AgentConfig(
        name="Analyst",
        role="Data Analyst",
        goal="Extract insights from research",
        backstory="Analytical expert",
        instructions="Identify patterns and key findings",
        tools=["graphrag"]
    ),
    AgentConfig(
        name="Writer",
        role="Technical Writer",
        goal="Create clear reports",
        backstory="Professional writer",
        instructions="Synthesize findings into clear report",
        tools=[]
    )
]

config.tasks = [
    TaskConfig(
        name="research",
        agent="Researcher",
        description="Research topic: {user_query}",
        expected_output="Research findings with sources"
    ),
    TaskConfig(
        name="analysis",
        agent="Analyst",
        description="Analyze research findings",
        expected_output="Key insights and patterns",
        context=["research"]
    ),
    TaskConfig(
        name="report",
        agent="Writer",
        description="Write executive summary",
        expected_output="Professional report",
        context=["research", "analysis"]
    )
]
```

**Execution Flow:**
```
research → analysis → report
            ↓           ↑
         (context passed)
```

---

## Parallel Execution Patterns

### Pattern 3: Parallel Research with Synthesis

**Use Case**: Multiple independent research streams that merge into a single analysis.

```python
config = OrchestratorConfig(name="ParallelResearch", process="workflow")

config.agents = [
    AgentConfig(name="WebResearcher", role="Web Search", goal="Search internet", ...),
    AgentConfig(name="AcademicResearcher", role="Academic Search", goal="Search papers", ...),
    AgentConfig(name="InternalResearcher", role="Internal Search", goal="Search memory", ...),
    AgentConfig(name="Synthesizer", role="Synthesis Expert", goal="Combine findings", ...),
]

config.tasks = [
    # Three parallel research tasks
    TaskConfig(
        name="web_research",
        agent="WebResearcher",
        description="Search web for information",
        expected_output="Web findings",
        is_start=True,  # Parallel start
        async_execution=True
    ),
    TaskConfig(
        name="academic_research",
        agent="AcademicResearcher",
        description="Search academic papers",
        expected_output="Academic findings",
        is_start=True,  # Parallel start
        async_execution=True
    ),
    TaskConfig(
        name="internal_research",
        agent="InternalResearcher",
        description="Search internal memory",
        expected_output="Internal findings",
        is_start=True,  # Parallel start
        async_execution=True
    ),
    # Synthesis waits for all three
    TaskConfig(
        name="synthesis",
        agent="Synthesizer",
        description="Combine all research findings",
        expected_output="Comprehensive synthesis",
        context=["web_research", "academic_research", "internal_research"]
    )
]
```

**Execution Flow:**
```
web_research ────┐
                 │
academic_research─┼──→ synthesis
                 │
internal_research─┘
```

**Parallel Efficiency**: ~3x speedup vs sequential

---

### Pattern 4: Parallel Processing with Aggregation

**Use Case**: Process multiple items in parallel, then aggregate results.

```python
config = OrchestratorConfig(name="ParallelProcessing", process="workflow")

# Create parallel processing agents
for i in range(5):
    config.agents.append(AgentConfig(
        name=f"Processor{i}",
        role=f"Processor {i}",
        goal="Process assigned items",
        ...
    ))

config.agents.append(AgentConfig(
    name="Aggregator",
    role="Result Aggregator",
    goal="Combine all results",
    ...
))

# Create parallel tasks
parallel_tasks = []
for i in range(5):
    task_name = f"process_{i}"
    config.tasks.append(TaskConfig(
        name=task_name,
        agent=f"Processor{i}",
        description=f"Process batch {i}",
        expected_output=f"Processed batch {i}",
        is_start=True,
        async_execution=True
    ))
    parallel_tasks.append(task_name)

# Aggregation task
config.tasks.append(TaskConfig(
    name="aggregate",
    agent="Aggregator",
    description="Aggregate all processed results",
    expected_output="Final aggregated result",
    context=parallel_tasks
))
```

**Execution Flow:**
```
process_0 ────┐
process_1 ────┤
process_2 ────┼──→ aggregate
process_3 ────┤
process_4 ────┘
```

---

## Conditional Routing Patterns

### Pattern 5: Decision Tree Routing

**Use Case**: Route workflow based on analysis results.

```python
config.agents = [
    AgentConfig(name="Analyzer", role="Risk Analyzer", ...),
    AgentConfig(name="HighRiskHandler", role="Emergency Response", ...),
    AgentConfig(name="MediumRiskHandler", role="Standard Review", ...),
    AgentConfig(name="LowRiskHandler", role="Documentation", ...),
]

config.tasks = [
    # Decision point
    TaskConfig(
        name="risk_analysis",
        agent="Analyzer",
        description="Analyze security risk level",
        expected_output="Risk level: high|medium|low",
        task_type="decision",
        condition={
            "high": ["emergency_response"],
            "medium": ["standard_review"],
            "low": ["documentation_only"]
        }
    ),
    # Conditional branches
    TaskConfig(
        name="emergency_response",
        agent="HighRiskHandler",
        description="Immediate security response",
        expected_output="Emergency action report"
    ),
    TaskConfig(
        name="standard_review",
        agent="MediumRiskHandler",
        description="Standard security review",
        expected_output="Review findings"
    ),
    TaskConfig(
        name="documentation_only",
        agent="LowRiskHandler",
        description="Document findings",
        expected_output="Documentation"
    )
]
```

**Execution Flow:**
```
                    ┌──→ emergency_response (if high)
                    │
risk_analysis ──────┼──→ standard_review (if medium)
                    │
                    └──→ documentation_only (if low)
```

---

## Memory Integration Patterns

### Pattern 6: Memory-Augmented Research

**Use Case**: Combine historical context from memory with fresh research.

```python
from orchestrator.tools.graph_rag_tool import create_graph_rag_tool

# Setup memory manager
orchestrator = Orchestrator(config)
orchestrator.initialize()

# Create GraphRAG tool
graphrag_tool = create_graph_rag_tool(
    orchestrator.memory_manager,
    default_user_id="user123",
    default_run_id="project_A"
)

# Agent with both memory and web search
config.agents.append(AgentConfig(
    name="HybridResearcher",
    role="Hybrid Research Specialist",
    goal="Combine historical and current information",
    instructions=(
        "1. Search GraphRAG memory for historical context\n"
        "2. Search web for current information\n"
        "3. Synthesize both sources with citations"
    ),
    tools=[graphrag_tool, "duckduckgo"],
    llm="gpt-4o-mini"
))
```

**Memory Retrieval Example:**

```python
# Agent automatically calls GraphRAG tool during execution
# The tool filters by tags, documents, sections:

results = graphrag_tool(
    query="authentication security",
    tags=["security", "api"],
    documents=["doc_001"],
    sections=["oauth", "jwt"],
    top_k=5
)
```

---

### Pattern 7: Progressive Memory Building

**Use Case**: Build up knowledge base across multiple workflow executions.

```python
# First execution: Store findings
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()

# Store results in memory
orchestrator.memory_manager.store(
    content=result["final_output"],
    metadata={
        "document_id": "workflow_001",
        "section": "api_security",
        "tags": ["security", "best_practices"],
        "user_id": "user123",
        "run_id": "project_A",
        "timestamp": "2024-01-01T12:00:00Z"
    }
)

# Second execution: Recall previous findings
config.custom_config = {
    "recall": {
        "query": "previous security findings",
        "limit": 5,
        "user_id": "user123",
        "run_id": "project_A",
        "rerank": True
    }
}

orchestrator2 = Orchestrator(config)
result2 = orchestrator2.run_sync()
# Now has access to previous findings via recall context
```

---

## Error Handling Patterns

### Pattern 8: Retry with Fallback

**Use Case**: Retry failed tasks with exponential backoff and fallback agents.

```python
from orchestrator.workflow.workflow_engine import WorkflowEngine

# Configure workflow engine with retries
workflow_engine = WorkflowEngine(
    process_type="workflow",
    max_concurrent_tasks=5,
    max_retries=3,  # Retry failed tasks 3 times
    retry_delay=1.0,  # Start with 1 second, exponential backoff
    timeout=300.0  # 5 minute timeout per task
)

# Primary task with high-capability agent
config.tasks.append(TaskConfig(
    name="complex_analysis",
    agent="AdvancedAnalyst",
    description="Complex analysis requiring GPT-4",
    expected_output="Detailed analysis"
))

# Fallback task with simpler agent
config.tasks.append(TaskConfig(
    name="simple_analysis",
    agent="BasicAnalyst",
    description="Simplified analysis using GPT-3.5",
    expected_output="Basic analysis",
    context=[]  # Independent, runs if complex_analysis fails
))
```

**Retry Behavior:**
```
Attempt 1: Run complex_analysis
  ↓ (fails)
Wait 1 second

Attempt 2: Run complex_analysis
  ↓ (fails)
Wait 2 seconds (exponential backoff)

Attempt 3: Run complex_analysis
  ↓ (fails)
Wait 4 seconds

Attempt 4: Run complex_analysis
  ↓ (fails)
Trigger fallback: simple_analysis
```

---

### Pattern 9: Validation with Guardrails

**Use Case**: Validate outputs before proceeding to next task.

```python
from praisonaiagents import GuardrailResult

def validate_security_analysis(task_output) -> GuardrailResult:
    """Validate security analysis completeness."""
    content = task_output.raw.lower()

    # Check for required sections
    required_sections = ["vulnerabilities", "recommendations", "severity"]
    missing = [s for s in required_sections if s not in content]

    if missing:
        return GuardrailResult(
            success=False,
            error=f"Missing required sections: {', '.join(missing)}"
        )

    # Check minimum length
    if len(content) < 500:
        return GuardrailResult(
            success=False,
            error="Analysis too short, needs more detail"
        )

    return GuardrailResult(success=True, result=task_output)

# Agent with guardrails
config.agents.append(AgentConfig(
    name="SecurityAnalyst",
    role="Security Analysis Expert",
    goal="Comprehensive security analysis",
    instructions="Analyze vulnerabilities, provide recommendations",
    guardrail=validate_security_analysis,
    max_guardrail_retries=3,
    self_reflect=True,
    min_reflect=1,
    max_reflect=3
))
```

---

## Advanced Patterns

### Pattern 10: Dynamic Planning with ToT

**Use Case**: Let the system decide the best task sequence based on the query.

```python
orchestrator = Orchestrator(config)
orchestrator.initialize()

# Dynamic task generation from prompt
orchestrator.plan_from_prompt(
    prompt="Build authentication system with OAuth 2.0",
    agent_sequence=["Researcher", "SecurityAnalyst", "Architect", "Implementer"],
    recall_snippets=["Previous OAuth implementation notes"],
    assignments=[
        {
            "objective": "Research OAuth 2.0 standards and best practices",
            "expected_output": "OAuth 2.0 research report",
            "tags": ["security", "oauth"]
        },
        {
            "objective": "Analyze security requirements and threats",
            "expected_output": "Security threat model",
            "tags": ["security", "threat-model"]
        },
        {
            "objective": "Design authentication architecture",
            "expected_output": "Architecture diagram and specs",
            "tags": ["architecture", "oauth"]
        },
        {
            "objective": "Implement OAuth 2.0 flow",
            "expected_output": "Working implementation",
            "tags": ["implementation"]
        }
    ]
)

result = await orchestrator.run()
```

**Generated Workflow:**
```
researcher_task_1 → analyst_task_2 → architect_task_3 → implementer_task_4
```

---

### Pattern 11: Multi-Tenant Workflows

**Use Case**: Isolate workflows and memory by user/project.

```python
# User A workflow
config_a = OrchestratorConfig(
    name="UserAWorkflow",
    user_id="user_a",
    run_id="project_alpha",
    memory=MemoryConfig(provider="hybrid")
)

# User B workflow
config_b = OrchestratorConfig(
    name="UserBWorkflow",
    user_id="user_b",
    run_id="project_beta",
    memory=MemoryConfig(provider="hybrid")
)

# Memory is automatically scoped by user_id and run_id
orchestrator_a = Orchestrator(config_a)
orchestrator_b = Orchestrator(config_b)

# User A's data is isolated from User B
result_a = orchestrator_a.run_sync()
result_b = orchestrator_b.run_sync()

# Retrieve user-specific memories
memories_a = orchestrator_a.memory_manager.retrieve_filtered(
    query="recent findings",
    user_id="user_a",
    run_id="project_alpha"
)

memories_b = orchestrator_b.memory_manager.retrieve_filtered(
    query="recent findings",
    user_id="user_b",
    run_id="project_beta"
)
```

---

### Pattern 12: Hierarchical Agent System

**Use Case**: Manager agent coordinates specialist agents.

```python
config = OrchestratorConfig(name="HierarchicalSystem", process="hierarchical")

# Manager agent
config.agents.append(AgentConfig(
    name="Manager",
    role="Project Manager",
    goal="Coordinate specialists to complete project",
    instructions=(
        "1. Analyze project requirements\n"
        "2. Assign tasks to appropriate specialists\n"
        "3. Review and integrate specialist outputs\n"
        "4. Ensure quality and completeness"
    ),
    llm="gpt-4o"  # Use more capable model for manager
))

# Specialist agents
specialists = [
    ("FrontendDev", "Frontend Developer", "Build UI components"),
    ("BackendDev", "Backend Developer", "Build API endpoints"),
    ("DatabaseDev", "Database Specialist", "Design data models"),
    ("QAEngineer", "QA Engineer", "Test functionality"),
]

for name, role, goal in specialists:
    config.agents.append(AgentConfig(
        name=name,
        role=role,
        goal=goal,
        llm="gpt-4o-mini"  # Simpler model for specialists
    ))

# Manager delegates to specialists
config.tasks.append(TaskConfig(
    name="project_execution",
    agent="Manager",
    description="Complete full-stack web application",
    expected_output="Working application with tests",
))
```

---

## Performance Optimization Patterns

### Pattern 13: Minimize Context Passing

**Best Practice**: Only pass necessary context between tasks.

```python
# ❌ Bad: Pass everything
TaskConfig(
    name="final_task",
    agent="Writer",
    context=["research1", "research2", "research3", "analysis1", "analysis2"]
)

# ✅ Good: Pass only what's needed
TaskConfig(
    name="final_task",
    agent="Writer",
    context=["analysis1"]  # analysis1 already includes research summaries
)
```

---

### Pattern 14: Cache Expensive Operations

**Best Practice**: Use memory to cache expensive computations.

```python
# Check if result exists in memory
cached_results = orchestrator.memory_manager.retrieve_filtered(
    query="analysis for query X",
    user_id="user123",
    run_id="project_A",
    limit=1
)

if cached_results:
    result = cached_results[0]["content"]
else:
    # Run expensive computation
    result = orchestrator.run_sync()

    # Cache result
    orchestrator.memory_manager.store(
        content=result,
        metadata={
            "query": "analysis for query X",
            "user_id": "user123",
            "run_id": "project_A",
            "cached_at": datetime.now().isoformat()
        }
    )
```

---

## Pattern Selection Guide

| Use Case | Recommended Pattern | Complexity | Performance |
|----------|-------------------|------------|-------------|
| Simple sequential tasks | Pattern 1 (Linear) | Low | Baseline |
| Knowledge work | Pattern 2 (Research→Analysis→Report) | Low | Good |
| Independent parallel tasks | Pattern 3 (Parallel Research) | Medium | 3-4x speedup |
| Large-scale processing | Pattern 4 (Parallel Processing) | Medium | 4-5x speedup |
| Conditional logic | Pattern 5 (Decision Tree) | Medium | Variable |
| Historical context needed | Pattern 6 (Memory-Augmented) | Medium | Good |
| Multi-execution workflows | Pattern 7 (Progressive Memory) | Medium | Excellent |
| Unreliable operations | Pattern 8 (Retry with Fallback) | Medium | Robust |
| Quality requirements | Pattern 9 (Validation Guardrails) | Medium | Reliable |
| Complex planning | Pattern 10 (Dynamic ToT Planning) | High | Adaptive |
| Multi-user systems | Pattern 11 (Multi-Tenant) | High | Scalable |
| Complex coordination | Pattern 12 (Hierarchical) | High | Organized |

---

## Best Practices Summary

1. **Start Simple**: Begin with Pattern 1 or 2, add complexity as needed
2. **Parallelize When Possible**: Use `is_start=True` and `async_execution=True` for independent tasks
3. **Use Memory Wisely**: Cache expensive operations, provide context via `recall_snippets`
4. **Validate Outputs**: Implement guardrails for critical tasks
5. **Handle Errors**: Configure retries and fallback agents
6. **Monitor Performance**: Use workflow engine metrics to identify bottlenecks
7. **Iterate**: Start with MVP workflow, refine based on results

---

For implementation details, see:
- [Orchestrator Architecture](ORCHESTRATOR_ARCHITECTURE.md)
- [Quick Start Guide](QUICK_START_GUIDE.md)
- [Main README](../README.md)
