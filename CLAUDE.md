# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified AI Agents framework combining production-ready components:

- **Orchestrator**: Scalable class-based architecture for multi-agent coordination with LangGraph execution and workflow engine
- **Mem0**: Intelligent memory layer for AI agents with multi-level memory and graph database support (+26% accuracy, 91% faster, 90% fewer tokens vs full context)
- **Tree-of-Thought-LLM**: Advanced reasoning system for complex problem solving

## Development Commands

### Environment Setup

```bash
# Install core dependencies with uv (recommended)
pip install uv
uv pip install -e .

# Install with specific features
uv pip install -e .[mem0-full]          # Complete Mem0 with all providers
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

Run pytest for orchestrator tests:

```bash
# Run all tests
pytest orchestrator/

# Test specific modules
pytest orchestrator/factories/tests/
pytest orchestrator/memory/tests/
```

### Quality Assurance

```bash
# Lint code
ruff check .
ruff format .
```

## High-Level Architecture

### Core Design Patterns

1. **Factory Pattern**: Dynamic agent and task creation via `AgentFactory` and `TaskFactory`
2. **Strategy Pattern**: Pluggable memory providers (`hybrid`, `mem0`, `rag`) and LangChain backend
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
                    │  LangGraph      │
                    │  StateGraph     │
                    └─────────────────┘
```

### Multi-Agent Workflow Patterns

1. **Sequential**: Linear task execution with context passing
2. **Hierarchical**: Manager agent coordinating worker agents
3. **Parallel**: Concurrent execution with result aggregation
4. **Router-Based**: Dynamic routing to specialized agents based on query analysis
5. **Tree-of-Thought**: Advanced planning with branching exploration

## Key Components Deep Dive

### Orchestrator (`orchestrator/`)

**Class-Based Architecture**: Modular design for scalable multi-agent coordination.

**Key Classes**:
- `Orchestrator`: Main coordination class with workflow management
- `AgentFactory`/`TaskFactory`: Dynamic component creation with template registry
- `MemoryManager`: Unified interface for multiple memory providers
- `WorkflowEngine`: DAG execution with parallel processing and metrics
- `GraphFactory`: LangGraph StateGraph builder for agent orchestration

**Agent Templates**: Router, Researcher, Analyst, Planner, StandardsAgent with specialized instructions.

**LangChain Integration**: Native LangChain/LangGraph backend for agent creation and execution.

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

# Alternative providers via LangChain
export OPENAI_BASE_URL=http://localhost:11434/v1  # Ollama
export OPENAI_BASE_URL=https://api.groq.com/openai/v1  # Groq
```

## Development Patterns

### Agent Creation with LangChain

```python
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig

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
├── orchestrator/              # Core orchestration system
│   ├── core/                  # Main orchestrator classes
│   ├── cli/                   # Command-line interface
│   ├── memory/                # Memory management
│   ├── factories/             # Agent and task factories
│   ├── integrations/          # LangChain integration
│   └── ingestion/             # Document processing pipeline
├── mem0/                      # Memory layer for AI
│   ├── mem0/                  # Core memory classes
│   └── examples/              # Integration examples
├── tree-of-thought-llm/       # Advanced reasoning system
└── examples/                  # Cross-system integration examples
```

## Key Dependencies

- **Core**: `pydantic`, `rich`, `openai`
- **LangChain**: `langchain`, `langgraph`, `langchain-openai`, `langchain-community`
- **Memory**: `chromadb`, `mem0ai[graph]`, `neo4j`, `qdrant-client`
- **Processing**: `pypdf`, `sentence-transformers`
- **Development**: `uv` for fast dependency management, `ruff` for linting

## Common Development Tasks

### Adding New Agent Templates

Add templates to `orchestrator/factories/agent_factory.py` following the existing pattern.

### Creating Custom Memory Providers

Implement `MemoryManager` interface in `orchestrator/memory/providers/` directory.

### Extending GraphRAG

Add new document templates in `orchestrator/ingestion/templates/` and update the ingestion pipeline.

### Testing New Features

Create pytest tests in appropriate `tests/` directories following the pattern in `orchestrator/factories/tests/`.

## Performance Optimization

- **Memory Caching**: Automatic embedding caching and connection pooling
- **Parallel Execution**: Workflow engine supports concurrent task execution
- **Token Efficiency**: Mem0 provides 90% token reduction vs full context
- **Hybrid Search**: Combined vector + lexical + graph for optimal retrieval

## Error Handling

The system provides comprehensive error handling with custom exception hierarchies and automatic retry mechanisms for LLM failures, memory provider issues, and workflow execution problems.
