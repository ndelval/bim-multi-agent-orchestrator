# Orchestrator Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Component Details](#component-details)
4. [Memory System](#memory-system)
5. [Workflow Engine](#workflow-engine)
6. [Agent Factory System](#agent-factory-system)
7. [Planning & Routing](#planning--routing)
8. [CLI Interface](#cli-interface)
9. [Integration Patterns](#integration-patterns)
10. [API Reference](#api-reference)

---

## Overview

The **Orchestrator** is a scalable, class-based architecture for building and managing multi-agent AI systems with support for:

- **Parallel Execution**: DAG-based workflow engine with concurrent task processing
- **Multiple Memory Providers**: Hybrid (ChromaDB + SQLite FTS5 + Neo4j), RAG, Mem0 with graph capabilities
- **Dual Backend Support**: Compatible with both PraisonAI (legacy) and LangGraph (modern) execution backends
- **Advanced Planning**: Tree-of-Thought (ToT) integration for complex reasoning and StateGraph generation
- **GraphRAG Tools**: Memory-augmented retrieval with semantic + lexical + graph search
- **Modular Design**: Factory pattern for agents/tasks, strategy pattern for backends, provider pattern for memory

**Key Capabilities:**
- Coordinate 5-20+ specialized AI agents in structured workflows
- Process 50-1000+ documents with hybrid search (vector + BM25 + graph)
- Execute tasks in parallel with automatic dependency resolution
- Dynamic planning with ToT reasoning for complex multi-step problems
- Real-time Rich CLI with progress tracking and live agent monitoring

---

## Core Architecture

### System Design Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                       Orchestrator                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Agent Factory │  │  Task Factory │  │ Memory Manager│      │
│  │   (Template)  │  │   (Template)  │  │  (Strategy)   │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │Workflow Engine│  │ Router System │  │ GraphRAG Tool │      │
│  │     (DAG)     │  │   (Planning)  │  │   (Memory)    │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼─────────┐    ┌───────▼─────────┐
            │  PraisonAI Core │    │ LangGraph State │
            │  (Legacy Mode)  │    │  (Modern Mode)  │
            └─────────────────┘    └─────────────────┘
```

**Design Patterns Used:**

1. **Factory Pattern** (`factories/`)
   - `AgentFactory`: Dynamic agent creation with template registry
   - `TaskFactory`: Configurable task creation with dependency validation
   - `GraphFactory`: StateGraph generation for LangGraph backend

2. **Strategy Pattern** (`memory/providers/`)
   - `IMemoryProvider`: Interface for pluggable memory systems
   - `HybridRAGMemoryProvider`: Tri-modal search (vector + lexical + graph)
   - `Mem0MemoryProvider`: Graph-based memory with temporal decay
   - `RAGMemoryProvider`: Simple vector-based retrieval

3. **Observer Pattern** (`workflow/`)
   - Workflow callbacks: `on_task_start`, `on_task_complete`, `on_workflow_complete`
   - Event emission system for Rich CLI display

4. **Builder Pattern** (`core/config.py`)
   - Configuration builders with validation and environment loading
   - Hierarchical config structures (Orchestrator → Agent → Task)

### Directory Structure

```
orchestrator/
├── core/                      # Core orchestration logic
│   ├── orchestrator.py       # Main Orchestrator class (925 lines)
│   ├── config.py             # Configuration management with Pydantic
│   ├── exceptions.py         # Custom exception hierarchy
│   └── embedding_utils.py    # Embedding model utilities
│
├── factories/                # Component creation factories
│   ├── agent_factory.py      # Agent creation with template registry
│   ├── task_factory.py       # Task creation and dependency validation
│   ├── agent_backends.py     # Backend strategy implementations
│   └── graph_factory.py      # StateGraph generation for LangGraph
│
├── memory/                   # Memory management system
│   ├── memory_manager.py     # Main memory coordinator
│   ├── document_schema.py    # Document metadata schemas
│   └── providers/            # Pluggable memory providers
│       ├── base.py           # IMemoryProvider interface
│       ├── rag_provider.py   # ChromaDB vector search
│       ├── mem0_provider.py  # Mem0 graph memory integration
│       ├── hybrid_provider.py # Tri-modal search system
│       └── registry.py       # Provider registration system
│
├── workflow/                 # Workflow execution engine
│   └── workflow_engine.py    # DAG-based parallel execution (594 lines)
│
├── planning/                 # Advanced planning systems
│   ├── tot_planner.py        # Tree-of-Thought base implementation
│   ├── tot_graph_planner.py  # ToT for StateGraph generation
│   └── graph_specifications.py # StateGraph spec definitions
│
├── tools/                    # Agent tools
│   └── graph_rag_tool.py     # GraphRAG memory retrieval tool
│
├── cli/                      # Command-line interface
│   ├── main.py               # CLI entry point with chat/info commands
│   ├── events.py             # Event emission system
│   ├── rich_display.py       # Rich terminal UI
│   └── graph_adapter.py      # Backend adapter for LangGraph
│
├── ingestion/                # Document processing pipeline
│   ├── extractors.py         # PDF/text extraction
│   ├── metadata_loader.py    # Metadata enrichment
│   ├── graph_sync.py         # Neo4j graph synchronization
│   └── sample_loader.py      # Document ingestion CLI
│
├── integrations/             # Backend integrations
│   ├── praisonai.py          # PraisonAI integration (legacy)
│   └── langchain_integration.py # LangGraph integration (modern)
│
└── templates/                # Agent/Task templates
    └── mock_engineering.py   # Engineering workflow templates
```

---

## Component Details

### 1. Orchestrator Core (`core/orchestrator.py`)

**Main Class:** `Orchestrator`

**Responsibilities:**
- Coordinate all system components (agents, tasks, memory, workflow)
- Manage dual-backend execution (PraisonAI / LangGraph)
- Provide high-level API for workflow execution
- Handle lifecycle management and resource cleanup

**Key Methods:**

```python
class Orchestrator:
    def __init__(self, config: Optional[OrchestratorConfig] = None)
    def initialize(self) -> None

    async def run(self) -> Any  # Async execution
    def run_sync(self) -> Any    # Sync wrapper

    def add_agent(self, agent_config: AgentConfig) -> Agent
    def add_task(self, task_config: TaskConfig) -> Task

    def plan_from_prompt(self, prompt: str, agent_sequence: Sequence[str], ...) -> None
    def create_graph_tool(self, *, user_id: str, run_id: str)

    def get_workflow_status(self) -> Dict[str, Any]
    def get_system_info(self) -> Dict[str, Any]

    def cleanup(self) -> None  # Resource cleanup
```

**Initialization Flow:**

1. **Config Validation** → Parse and validate `OrchestratorConfig`
2. **Memory Setup** → Initialize memory provider (hybrid/rag/mem0)
3. **Agent Creation** → Create agents from templates via `AgentFactory`
4. **Task Creation** → Create tasks with dependency validation
5. **Backend Selection** → Choose PraisonAI or LangGraph based on availability
6. **Tool Registration** → Attach GraphRAG and web search tools to agents

**Backend Decision Logic:**

```python
# Compatibility layer in orchestrator.py (lines 14-25)
try:
    from ..integrations.langchain_integration import OrchestratorState, StateGraph
    USING_LANGGRAPH = True
except ImportError:
    USING_LANGGRAPH = False
    # Fallback to PraisonAI
```

**Memory Recall System** (`_build_recall_content()`, lines 422-481):

Supports global context injection from memory before workflow execution:

```python
# Example custom_config for memory recall:
{
    "recall": {
        "query": "previous research on topic X",
        "limit": 5,
        "agent_id": "researcher",
        "run_id": "project_2024",
        "rerank": true
    }
}
```

---

### 2. Configuration System (`core/config.py`)

**Dataclass Hierarchy:**

```python
@dataclass
class EmbedderConfig:
    provider: str = "openai"
    config: Dict[str, Any] = field(default_factory=lambda: {
        "model": "text-embedding-3-large"
    })

@dataclass
class MemoryConfig:
    provider: MemoryProvider = MemoryProvider.HYBRID
    use_embedding: bool = True
    embedder: Optional[EmbedderConfig] = None
    # Provider-specific paths
    hybrid_vector_path: Optional[str] = ".praison/hybrid_chroma"
    hybrid_lexical_db_path: Optional[str] = ".praison/hybrid_lexical.db"
    rag_db_path: Optional[str] = ".praison/memory/chroma_db"

@dataclass
class AgentConfig:
    name: str
    role: str
    goal: str
    backstory: str
    instructions: str
    tools: List[str | Callable] = field(default_factory=list)
    enabled: bool = True
    llm: Optional[str] = None  # e.g., "gpt-4o-mini"

@dataclass
class TaskConfig:
    name: str
    description: str
    expected_output: str
    agent: str
    context: List[str] = field(default_factory=list)  # Dependencies
    next_tasks: List[str] = field(default_factory=list)
    async_execution: bool = False
    is_start: bool = False
    task_type: str = "standard"
    condition: Optional[Dict[str, List[str]]] = None

@dataclass
class OrchestratorConfig:
    name: str
    process: str = "sequential"  # workflow | sequential | hierarchical
    agents: List[AgentConfig] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    memory: Optional[MemoryConfig] = None
    verbose: int = 1
    max_iter: int = 25
    user_id: str = "default_user"
    run_id: Optional[str] = None
    async_execution: bool = False
```

**Loading Patterns:**

```python
# From YAML file
config = OrchestratorConfig.from_yaml("config.yaml")

# From dictionary
config_dict = {...}
config = OrchestratorConfig.from_dict(config_dict)

# From environment variables
config = OrchestratorConfig.from_env(prefix="ORCHESTRATOR_")

# Validation
config.validate()  # Raises ValidationError if invalid
```

---

### 3. Agent Factory System (`factories/agent_factory.py`)

**Architecture:** Template Pattern + Strategy Pattern

**Agent Templates:**

1. **OrchestratorAgentTemplate** - Coordinate work among specialists
2. **ResearcherAgentTemplate** - Web search and information gathering
3. **PlannerAgentTemplate** - Create actionable plans
4. **ImplementerAgentTemplate** - Build prototypes
5. **TesterAgentTemplate** - Validate functionality
6. **WriterAgentTemplate** - Technical documentation

**Backend Strategy System:**

```python
class AgentFactory:
    def __init__(self, default_mode: Optional[str] = None):
        self._backend_registry = BackendRegistry()  # langchain | praisonai
        self._default_mode = default_mode or self._detect_default_mode()

    def create_agent(
        self,
        config: AgentConfig,
        mode: Optional[str] = None,  # Override backend
        **kwargs
    ) -> Agent:
        # Auto-detect or use explicit mode
        mode = mode or self._default_mode
        backend = self._backend_registry.get(mode)

        # Backend creates agent with tools
        agent = backend.create_agent(config, **kwargs)
        return agent
```

**Tool Resolution:**

Tools can be specified as:
- **String names**: `["duckduckgo", "wikipedia", "graphrag"]` → Auto-resolved by backend
- **Callable objects**: Pre-instantiated tool functions

---

## Memory System

### Three-Tier Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Store   │    │ Lexical Index   │    │  Graph Store    │
│   (ChromaDB)    │    │  (SQLite FTS5)  │    │    (Neo4j)      │
│  Semantic       │    │  BM25 Ranking   │    │  Relationships  │
│  Similarity     │    │  Exact Match    │    │  Multi-hop      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Memory Manager  │
                    │   (Coordinator) │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Orchestrator  │
                    └─────────────────┘
```

### Provider Implementations

#### 1. **HybridRAGMemoryProvider** (Recommended)

**File:** `memory/providers/hybrid_provider.py`

**Features:**
- **Tri-modal search**: Vector (semantic) + Lexical (BM25) + Graph (relationships)
- **RRF (Reciprocal Rank Fusion)**: Combines rankings from multiple sources
- **Cross-encoder reranking**: Optional final reranking with sentence-transformers
- **Metadata filtering**: Filter by tags, documents, sections, user_id, run_id

**Configuration:**

```python
MemoryConfig(
    provider=MemoryProvider.HYBRID,
    hybrid_vector_path=".praison/hybrid_chroma",
    hybrid_lexical_db_path=".praison/hybrid_lexical.db",
    embedder=EmbedderConfig(
        provider="openai",
        config={"model": "text-embedding-3-large"}
    ),
    config={
        "neo4j_url": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password"
    }
)
```

**Retrieval Example:**

```python
results = memory_manager.retrieve_with_graph(
    query="authentication best practices",
    limit=10,
    tags=["security", "api"],
    document_ids=["doc_001"],
    rerank=True,  # Enable cross-encoder reranking
    user_id="user123",
    run_id="project_A"
)
```

#### 2. **Mem0MemoryProvider**

**File:** `memory/providers/mem0_provider.py`

**Features:**
- **Graph-based memory**: Entity relationships with Neo4j/Memgraph
- **Temporal decay**: Automatic memory aging and importance scoring
- **Multi-level memory**: User, session, agent scopes
- **Cross-session persistence**: Maintain context across conversations

**Configuration:**

```python
MemoryConfig(
    provider=MemoryProvider.MEM0,
    embedder=EmbedderConfig(...),
    config={
        "graph_url": "bolt://localhost:7687",
        "vector_provider": "qdrant",  # Optional
        "qdrant_host": "localhost",
        "qdrant_port": 6333
    }
)
```

#### 3. **RAGMemoryProvider**

**File:** `memory/providers/rag_provider.py`

**Features:**
- **Simple vector search**: ChromaDB-based semantic retrieval
- **Lightweight**: No external dependencies (Neo4j, Qdrant)
- **Fast setup**: Good for prototyping and small-scale deployments

**Configuration:**

```python
MemoryConfig(
    provider=MemoryProvider.RAG,
    rag_db_path=".praison/memory/chroma_db",
    embedder=EmbedderConfig(...)
)
```

### Memory Manager API

```python
class MemoryManager:
    def store(self, content: str, metadata: Optional[Dict] = None) -> str
    def retrieve(self, query: str, limit: int = 10) -> List[Dict]

    def retrieve_filtered(
        self, query: str, *, limit: int = 10,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        rerank: Optional[bool] = None
    ) -> List[Dict]

    def retrieve_with_graph(
        self, query: str, limit: int = 5,
        tags: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict]

    def update(self, ref_id: str, content: str, metadata: Optional[Dict] = None)
    def delete(self, ref_id: str)
    def health_check(self) -> bool
    def cleanup(self)

    def create_graph_tool(self, *, default_user_id: str, default_run_id: str)
```

---

## Workflow Engine

### DAG Execution System

**File:** `workflow/workflow_engine.py` (594 lines)

**Key Features:**
- **Directed Acyclic Graph (DAG)**: Automatic dependency resolution
- **Parallel Execution**: Concurrent task processing with semaphores
- **Retry Logic**: Exponential backoff for failed tasks
- **Execution Strategies**: Sequential, Parallel, Mixed (adaptive)
- **Real-time Monitoring**: Task status tracking and metrics collection

**Task Execution States:**

```python
class TaskStatus(Enum):
    PENDING = "pending"      # Not yet started
    READY = "ready"          # Dependencies completed
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Error occurred
    SKIPPED = "skipped"      # Skipped due to conditions
```

**Workflow Configuration:**

```python
workflow_engine = WorkflowEngine(
    process_type=ProcessType.WORKFLOW,  # or SEQUENTIAL, HIERARCHICAL
    max_concurrent_tasks=5,  # Parallel execution limit
    max_retries=3,  # Retry failed tasks
    retry_delay=1.0,  # Exponential backoff base
    timeout=300.0  # Task timeout in seconds
)
```

**Execution Flow:**

1. **Add Tasks** → Validate dependencies and build DAG
2. **Topological Sort** → Determine execution order in levels
3. **Execute Levels** → Run tasks in parallel where possible
4. **Monitor Progress** → Track status and collect metrics
5. **Handle Failures** → Retry with backoff or skip based on criticality

**Parallel Execution Levels:**

```python
# Example DAG structure:
Level 0 (Parallel):  [task_a, task_b, task_c]  # No dependencies
Level 1 (Parallel):  [task_d, task_e]          # Depend on Level 0
Level 2 (Sequential): [task_f]                  # Depends on task_d, task_e
```

**Metrics Collection:**

```python
@dataclass
class WorkflowMetrics:
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    total_duration: float
    parallel_efficiency: float  # total_task_time / workflow_duration
```

---

## Planning & Routing

### Tree-of-Thought Integration

**File:** `planning/tot_graph_planner.py`

**Purpose:** Generate complex StateGraph specifications using ToT reasoning for advanced multi-agent workflows.

**Configuration:**

```python
@dataclass
class GraphPlanningSettings:
    backend: str = "gpt-4"
    temperature: float = 0.7
    max_steps: int = 5  # Tree depth
    n_generate_sample: int = 3  # Branches per step
    n_evaluate_sample: int = 2  # Evaluations per branch
    n_select_sample: int = 2  # Top branches to keep

    enable_parallel_planning: bool = True
    enable_conditional_routing: bool = True
    max_parallel_groups: int = 3
```

**ToT Search Process:**

```
Step 0: Generate 3 alternative graph components
├─ Option A: {"component_type":"node",...}
├─ Option B: {"component_type":"node",...}
└─ Option C: {"component_type":"parallel_group",...}

Evaluate each option (2 times for averaging)
├─ A: score = 8.5
├─ B: score = 7.0
└─ C: score = 9.0

Select top 2 options (greedy)
├─ A: 8.5 ✓
└─ C: 9.0 ✓

Repeat for max_steps levels...
```

### Router System (CLI)

**File:** `cli/main.py` (lines 141-242)

**Router Agent Template:**

```python
AgentConfig(
    name="Router",
    role="Query Analyzer & Decision Router",
    goal="Analyze user queries and route to appropriate execution path",
    instructions=(
        "Classify queries into:\n"
        "- 'quick': Simple factual questions\n"
        "- 'analysis': Complex research requiring multi-agent workflows\n"
        "- 'planning': Strategic planning requiring ToT reasoning\n"
        "Return JSON: {decision, confidence, rationale}"
    ),
    tools=["graphrag"]
)
```

**Routing Decision Flow:**

1. **Query Analysis** → Router agent analyzes user query
2. **GraphRAG Lookup** → Search memory for relevant context
3. **Classification** → Determine routing decision (quick/analysis/planning)
4. **Execution** → Dispatch to appropriate workflow

---

## CLI Interface

### Commands

**File:** `cli/main.py`

#### 1. **Chat Command**

Start interactive chat with agent orchestration:

```bash
python -m orchestrator.cli chat \
  --memory-provider hybrid \
  --verbose

# Options:
#   --memory-provider {hybrid|rag|mem0}
#   --verbose / --no-verbose
#   --use-tools / --no-use-tools
```

**Execution Flow:**

1. **Initialize Memory** → Setup hybrid/rag/mem0 provider
2. **Build Agents** → Create Router, Researcher, Analyst, Planner, Standards agents
3. **Attach Tools** → GraphRAG + DuckDuckGo + Wikipedia
4. **Start Chat Loop** → Interactive conversation with routing

#### 2. **Memory Info Command**

Display memory provider configuration:

```bash
python -m orchestrator.cli memory-info \
  --memory-provider hybrid
```

**Output:**
- Provider class and module
- Initialization status
- Health check results
- Configuration details

### Rich Display System

**File:** `cli/rich_display.py`

**Features:**
- **Live progress bars** for agent execution
- **Real-time status updates** with color coding
- **Formatted output** with Markdown rendering
- **Event-driven updates** via event emission system

**Event System:**

```python
# events.py
emit_router_decision(decision, confidence, rationale)
emit_agent_start(agent_name, task_description)
emit_agent_progress(agent_name, progress_text)
emit_agent_complete(agent_name, result)
emit_final_answer(answer_text)
```

---

## Integration Patterns

### Dual Backend Support

#### PraisonAI Mode (Legacy)

```python
from praisonaiagents import Agent, Task, PraisonAIAgents

# Create agents and tasks
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()
```

#### LangGraph Mode (Modern)

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

# StateGraph execution with compiled graph
orchestrator = Orchestrator(config)
result = await orchestrator.run()
```

### Memory Integration

**GraphRAG Tool Creation:**

```python
# In orchestrator
graphrag_tool = orchestrator.create_graph_tool(
    user_id="user123",
    run_id="project_A"
)

# Attach to agent
agent_config.tools.append(graphrag_tool)
```

**Tool Usage by Agents:**

```python
# Agent calls GraphRAG tool during execution
def graph_rag_lookup(
    query: str,
    tags: Optional[str] = None,
    documents: Optional[str] = None,
    top_k: int = 5
) -> str:
    """Retrieve document fragments using GraphRAG."""
    # Returns formatted results with citations
```

---

## API Reference

### Quick Start

```python
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig, MemoryConfig

# 1. Create configuration
config = OrchestratorConfig(name="MyWorkflow")

# 2. Add agents
config.agents.append(AgentConfig(
    name="Researcher",
    role="Research Specialist",
    goal="Gather information",
    backstory="Expert researcher",
    instructions="Search web and memory for information",
    tools=["duckduckgo", "graphrag"]
))

# 3. Add tasks
config.tasks.append(TaskConfig(
    name="research_task",
    description="Research best practices",
    expected_output="Research report",
    agent="Researcher",
    is_start=True
))

# 4. Configure memory
config.memory = MemoryConfig(provider="hybrid")

# 5. Execute
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()
```

### Factory Methods

```python
# From file
orchestrator = Orchestrator.from_file("config.yaml")

# From dictionary
orchestrator = Orchestrator.from_dict(config_dict)

# From environment
orchestrator = Orchestrator.from_env(prefix="ORCH_")

# Default setup
orchestrator = Orchestrator.create_default("MyOrchestrator")
```

### Dynamic Planning

```python
# Generate tasks dynamically from prompt
orchestrator.plan_from_prompt(
    prompt="Build authentication system",
    agent_sequence=["Researcher", "Planner", "Implementer"],
    recall_snippets=["previous auth notes"],
    assignments=[
        {"objective": "Research OAuth standards", "tags": ["security"]},
        {"objective": "Create implementation plan"},
        {"objective": "Build prototype"}
    ]
)

# Execute
result = await orchestrator.run()
```

### Memory Operations

```python
# Store documents
memory_manager.store(
    content="Authentication best practices...",
    metadata={
        "document_id": "doc_001",
        "section": "security",
        "tags": ["auth", "security"]
    }
)

# Retrieve with filters
results = memory_manager.retrieve_filtered(
    query="JWT implementation",
    limit=10,
    user_id="user123",
    run_id="project_A",
    rerank=True
)

# Graph-enhanced search
results = memory_manager.retrieve_with_graph(
    query="security vulnerabilities",
    tags=["security"],
    document_ids=["doc_001", "doc_002"],
    sections=["authentication"]
)
```

### Workflow Monitoring

```python
# Get real-time status
status = orchestrator.get_workflow_status()
# Returns: {is_running, total_tasks, pending_tasks, running_tasks, completed_tasks}

# Get system info
info = orchestrator.get_system_info()
# Returns: {name, agents, tasks, execution, memory}

# Export execution graph
graph_data = workflow_engine.export_execution_graph()
# Returns: {nodes, edges, metrics}
```

---

## Performance Characteristics

### Memory System

| Provider | Search Type | Latency (avg) | Throughput | Index Size (1K docs) |
|----------|-------------|---------------|------------|----------------------|
| RAG | Vector | ~50ms | 200 QPS | ~500MB |
| Mem0 | Graph + Vector | ~100ms | 100 QPS | ~1GB |
| Hybrid | Tri-modal + Rerank | ~150ms | 80 QPS | ~1.5GB |

### Workflow Engine

| Execution Mode | Max Concurrent | Overhead | Parallel Efficiency |
|----------------|----------------|----------|---------------------|
| Sequential | 1 | 5% | 1.0 (baseline) |
| Parallel (5 workers) | 5 | 15% | 3.5-4.2x |
| Mixed (adaptive) | 1-15 | 10-20% | 2.8-4.0x |

### Scaling Limits

- **Agents**: 5-20 per workflow (optimal 8-12)
- **Tasks**: 10-100 per workflow (optimal 20-50)
- **Documents**: 100-10,000 (hybrid provider recommended for >1000)
- **Memory Retrieval**: <200ms for 10 results (p95)
- **Workflow Duration**: 10s-30min (depends on LLM latency)

---

## Best Practices

### 1. Memory Provider Selection

- **RAG**: Prototypes, <500 documents, no graph needed
- **Hybrid**: Production, >500 documents, complex queries, multi-modal search
- **Mem0**: Cross-session context, entity relationships, temporal tracking

### 2. Workflow Design

- **Minimize task count**: 20-30 tasks ideal for complex workflows
- **Maximize parallelization**: Use `async_execution=True` and `is_start=True` for independent tasks
- **Explicit dependencies**: Use `context` field to define task dependencies
- **Error handling**: Set `max_retries=3` and implement failure callbacks

### 3. Agent Configuration

- **Specific roles**: Define clear, non-overlapping responsibilities
- **Minimal tools**: Attach only necessary tools (3-5 max per agent)
- **Detailed instructions**: Provide step-by-step guidance in `instructions` field
- **LLM selection**: Use `gpt-4o-mini` for speed, `gpt-4o` for quality

### 4. Performance Optimization

- **Tool caching**: GraphRAG tool caches embeddings automatically
- **Connection pooling**: Memory providers reuse database connections
- **Batch operations**: Use `retrieve_with_graph()` for multi-source queries
- **Async execution**: Enable `async_execution=True` for I/O-bound tasks

---

## Troubleshooting

### Common Issues

**1. "Memory provider not initialized"**
- Check environment variables (NEO4J_URL, OPENAI_API_KEY)
- Verify database services are running (Neo4j, Qdrant)
- Review `memory_manager.health_check()` output

**2. "Circular dependencies detected"**
- Validate task `context` fields don't form cycles
- Use `config.validate()` before initialization
- Review workflow DAG with `workflow_engine.get_execution_order()`

**3. "LangGraph components not available"**
- Install LangGraph dependencies: `pip install langgraph langchain-core`
- Verify imports in `integrations/langchain_integration.py`
- Fallback to PraisonAI mode if LangGraph not needed

**4. "GraphRAG tool not found"**
- Ensure tool is created before agent initialization: `orchestrator.create_graph_tool()`
- Check tool registration in `_create_langgraph_system()` method
- Verify agent `tools` field includes `"graphrag"` string

### Debug Mode

```bash
# Enable detailed logging
python -m orchestrator.cli chat --verbose

# Check component status
python -m orchestrator.cli memory-info --memory-provider hybrid

# Validate configuration
python -c "
from orchestrator import Orchestrator
config = Orchestrator.from_file('config.yaml')
config.config.validate()
"
```

---

## Future Enhancements

### Planned Features

1. **Observability**
   - OpenTelemetry integration for distributed tracing
   - Prometheus metrics export
   - Real-time workflow visualization dashboard

2. **Advanced Routing**
   - ML-based query classification
   - Dynamic agent selection based on performance history
   - A/B testing for routing strategies

3. **Memory Improvements**
   - Automatic index optimization
   - Multi-tenant isolation
   - Distributed caching layer (Redis)

4. **Workflow Features**
   - Human-in-the-loop approval gates
   - Conditional branching with complex predicates
   - Sub-workflow composition

5. **Integration Ecosystem**
   - Kubernetes operator for deployment
   - REST API server with OpenAPI spec
   - LangSmith/LangFuse integration for monitoring

---

**Last Updated:** 2025-01-03
**Version:** 1.0.0
**Maintainer:** PraisonAI Team
