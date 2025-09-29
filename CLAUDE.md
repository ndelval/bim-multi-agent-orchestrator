# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified AI Agents framework combining multiple production-ready components:

- **PraisonAI**: Multi-agent AI framework with self-reflection, supporting 100+ LLMs and various execution patterns
- **Mem0**: Intelligent memory layer for AI agents with multi-level memory and graph database support (+26% accuracy, 91% faster, 90% fewer tokens vs full context)
- **Orchestrator**: Scalable class-based architecture for multi-agent coordination with parallel execution and workflow engine
- **Tree-of-Thought-LLM**: Advanced reasoning system for complex problem solving

## Development Commands

### Environment Setup

```bash
# Install core dependencies with uv (recommended)
pip install uv
uv pip install -e .

# Install with specific features
uv pip install -e .[all-basic]          # Basic setup with PraisonAI + Mem0
uv pip install -e .[mem0-full]          # Complete Mem0 with all providers
uv pip install -e .[praison-full]       # Complete PraisonAI with all interfaces
uv pip install -e .[all-mem0]           # Development setup with testing
```

### Core Development Workflow

```bash
# Start orchestrator chat CLI with hybrid memory (ChromaDB + SQLite FTS5 + Neo4j)
python -m orchestrator.cli chat --memory-provider hybrid

# Start with Mem0 graph memory
python -m orchestrator.cli chat --memory-provider mem0

# Show memory configuration
python -m orchestrator.cli memory-info --memory-provider hybrid

# Ingest documents into GraphRAG system
python -m orchestrator.ingestion.sample_loader \
  --input docs/document.pdf \
  --metadata manifests/ \
  --memory-provider hybrid
```

### Testing Strategy

The project uses example-driven testing with 100+ test files. Run individual examples rather than formal test runners:

```bash
# Test PraisonAI Agents core functionality
python PraisonAI/src/praisonai-agents/test_agent_simple.py
python PraisonAI/src/praisonai-agents/test_async_sequential.py
python PraisonAI/src/praisonai-agents/test_multi_agents.py

# Test memory systems
python examples/memory/test_mem0_memory.py

# Test specific features
python PraisonAI/src/praisonai-agents/test_guardrails.py      # Guardrails system
python PraisonAI/src/praisonai-agents/test_streaming.py      # Streaming responses
python PraisonAI/src/praisonai-agents/test_self_reflection.py # Self-reflection
```

### Quality Assurance

```bash
# Lint code (when ruff is configured)
ruff check .
ruff format .

# Check syntax in PraisonAI agents
python PraisonAI/src/praisonai-agents/check_syntax.py
```

## High-Level Architecture

### Core Design Patterns

1. **Factory Pattern**: Dynamic agent and task creation via `AgentFactory` and `TaskFactory`
2. **Strategy Pattern**: Pluggable memory providers (`hybrid`, `mem0`, `rag`) and LLM backends
3. **Observer Pattern**: Callback systems for workflow events and progress tracking
4. **Dependency Injection**: Configuration-driven component initialization with validation

### Memory Architecture (Three-Tier System)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Store   │    │ Lexical Index   │    │  Graph Store    │
│   (ChromaDB/    │    │  (SQLite FTS5   │    │  (Neo4j/        │
│   Qdrant)       │    │   + BM25)       │    │   Memgraph)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Memory Manager  │
                    │ (Orchestrator)  │
                    └─────────────────┘
```

**Hybrid Provider Features**:
- **Vector Search**: Semantic similarity via ChromaDB/Qdrant embeddings
- **Lexical Search**: BM25 ranking via SQLite FTS5 for exact term matching
- **Graph Relations**: Neo4j for entity relationships and multi-hop queries
- **Cross-Encoder Reranking**: Optional reranking with sentence-transformers

### Agent Orchestration Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Agent Factory   │  │ Task Factory    │  │ Memory Manager  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Workflow Engine │  │ Router System   │  │ GraphRAG Tool   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌─────────────────┐
                    │ PraisonAI Core  │
                    │   Agents        │
                    └─────────────────┘
```

### Multi-Agent Workflow Patterns

1. **Sequential**: Linear task execution with context passing
2. **Hierarchical**: Manager agent coordinating worker agents
3. **Parallel**: Concurrent execution with result aggregation
4. **Router-Based**: Dynamic routing to specialized agents based on query analysis
5. **Tree-of-Thought**: Advanced planning with branching exploration

## Key Components Deep Dive

### PraisonAI Agents (`PraisonAI/src/praisonai-agents/`)

**Core Classes**:
- `Agent`: LLM-powered agent with self-reflection (1-3 iterations), tool calling, and guardrails
- `Task`: Configurable tasks with Pydantic output, conditional execution, and context dependencies
- `PraisonAIAgents`: Multi-agent orchestrator supporting sequential/hierarchical/parallel execution
- `GuardrailResult`: Validation system for task outputs with automatic retry mechanisms

**Self-Reflection System**: Agents automatically evaluate and improve their outputs through configurable iteration cycles.

**Guardrails System**: Both function-based and LLM-based validation with automatic retry mechanisms.

### Orchestrator (`orchestrator/`)

**Class-Based Architecture**: Modular design replacing 280+ line monolithic functions.

**Key Classes**:
- `Orchestrator`: Main coordination class with workflow management
- `AgentFactory`/`TaskFactory`: Dynamic component creation with template registry
- `MemoryManager`: Unified interface for multiple memory providers
- `WorkflowEngine`: DAG execution with parallel processing and metrics

**Agent Templates**: Router, Researcher, Analyst, Planner, StandardsAgent with specialized instructions.

### Mem0 Integration (`mem0/`)

**Memory Types**:
- **User Memory**: Personal preferences and history
- **Session Memory**: Conversation context
- **Agent Memory**: Behavioral patterns and learning

**Graph Memory Features**:
- Multi-hop reasoning across entity relationships
- Temporal memory with automatic decay
- Cross-session memory persistence with user/agent/run scoping

### Tree-of-Thought System (`tree-of-thought-llm/`)

**Planning Integration**: Automatic assignment generation when router decisions require complex planning.

**Search Strategies**: Breadth-first, depth-first, and greedy search for optimal solution paths.

## Environment Configuration

### Memory Provider Configuration

```bash
# Hybrid Provider (Default - Local ChromaDB + SQLite + Neo4j)
export ORCH_MEMORY_PROVIDER=hybrid
export HYBRID_VECTOR_PATH=.praison/hybrid_chroma
export HYBRID_LEXICAL_DB_PATH=.praison/hybrid_lexical.db
export NEO4J_URL=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Mem0 Provider (Graph + Optional Qdrant)
export ORCH_MEMORY_PROVIDER=mem0
export MEM0_GRAPH_URL=bolt://localhost:7687
export MEM0_VECTOR_PROVIDER=qdrant
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# RAG Provider (Simple ChromaDB)
export ORCH_MEMORY_PROVIDER=rag
export RAG_VECTOR_PATH=.praison/memory/chroma_db
```

### LLM Configuration

```bash
# OpenAI (Default)
export OPENAI_API_KEY=your_key_here

# Alternative providers via PraisonAI
export OPENAI_BASE_URL=http://localhost:11434/v1  # Ollama
export OPENAI_BASE_URL=https://api.groq.com/openai/v1  # Groq
```

## Development Patterns

### Agent Creation with Guardrails

```python
from praisonaiagents import Agent, GuardrailResult
from typing import Tuple, Any

def validate_output(task_output) -> GuardrailResult:
    if "error" in task_output.raw.lower():
        return GuardrailResult(success=False, error="Contains errors")
    return GuardrailResult(success=True, result=task_output)

agent = Agent(
    name="ResearchAgent",
    role="Research Specialist", 
    goal="Gather comprehensive information",
    backstory="Expert researcher with analytical skills",
    llm="gpt-4o-mini",
    self_reflect=True,
    min_reflect=1,
    max_reflect=3,
    guardrail=validate_output,
    max_guardrail_retries=3,
    tools=[duckduckgo_tool, wikipedia_tool]
)
```

### Orchestrator Usage

```python
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig

# Configuration-driven approach
config = OrchestratorConfig(name="MyWorkflow")
config.agents.append(AgentConfig(
    name="Researcher",
    role="Research Specialist",
    goal="Gather information",
    backstory="Expert researcher",
    tools=["duckduckgo"]
))

orchestrator = Orchestrator(config)
result = orchestrator.run_sync()
```

### Memory Integration

```python
from mem0 import Memory

memory = Memory()

# Store conversation memory
memory.add(messages, user_id="user123", agent_id="researcher")

# Search relevant memories
memories = memory.search(query="previous research", user_id="user123", limit=5)

# Graph-based queries (with Mem0 graph provider)
related = memory.search(query="project dependencies", 
                       filters={"memory_type": "fact"})
```

## Project Structure Navigation

```
├── PraisonAI/                 # Core multi-agent framework
│   ├── src/praisonai-agents/  # Agent system implementation
│   └── examples/              # 100+ example notebooks and scripts
├── mem0/                      # Memory layer for AI
│   ├── mem0/                  # Core memory classes
│   └── examples/              # Integration examples
├── orchestrator/              # Scalable orchestration system
│   ├── core/                  # Main orchestrator classes
│   ├── cli/                   # Command-line interface
│   ├── memory/                # Memory management
│   └── ingestion/             # Document processing pipeline
├── tree-of-thought-llm/       # Advanced reasoning system
└── examples/                  # Cross-system integration examples
```

## Key Dependencies

- **Core**: `pydantic`, `rich`, `openai`, `praisonaiagents`
- **Memory**: `chromadb`, `mem0ai[graph]`, `neo4j`, `qdrant-client`
- **LLM**: `litellm` for unified provider access
- **Processing**: `pypdf`, `sentence-transformers`
- **Development**: `uv` for fast dependency management, `ruff` for linting

## Common Development Tasks

### Adding New Agent Templates

Add templates to `orchestrator/cli/main.py` in the `_build_chat_agents()` function following the existing pattern.

### Creating Custom Memory Providers

Implement `MemoryManager` interface in `orchestrator/memory/providers/` directory.

### Extending GraphRAG

Add new document templates in `orchestrator/ingestion/templates/` and update the ingestion pipeline.

### Testing New Features

Create example-driven tests following the pattern in `PraisonAI/src/praisonai-agents/test_*.py` files.

## Performance Optimization

- **Memory Caching**: Automatic embedding caching and connection pooling
- **Parallel Execution**: Workflow engine supports concurrent task execution
- **Token Efficiency**: Mem0 provides 90% token reduction vs full context
- **Hybrid Search**: Combined vector + lexical + graph for optimal retrieval

## Error Handling

The system provides comprehensive error handling with custom exception hierarchies and automatic retry mechanisms for LLM failures, memory provider issues, and workflow execution problems.