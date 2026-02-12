# Multi-Agent Orchestration Framework for BIM/CAD Automation

A multi-agent orchestration framework built on LangGraph and LangChain, designed for automating BIM/CAD workflows in the Architecture, Engineering, and Construction (AEC) sector. The system integrates hybrid memory retrieval (vector + lexical + graph), Tree-of-Thought planning, and the Model Context Protocol (MCP) for tool interoperability.

Developed as part of a Bachelor's Thesis (TFG) at the Universidad Pontificia Comillas ICAI

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Memory Providers](#memory-providers)
- [Agent System](#agent-system)
- [Tree-of-Thought Planning](#tree-of-thought-planning)
- [BIM Information Retrieval Agent](#bim-information-retrieval-agent)
- [Testing](#testing)
- [Technical Details](#technical-details)

---

## Architecture Overview

The framework is organized in three layers:

```
+-------------------------------------------------------------+
| CLI Layer                                                    |
|   ChatOrchestrator - DisplayAdapter - EventSystem            |
|   main.py -> ChatOrchestrator -> GraphAgentAdapter           |
+-------------------------------------------------------------+
| Orchestration Layer                                          |
|   GraphFactory - RouteClassifier - ToT Planner               |
|   StateGraph <- AgentFactory <- AgentConfig                  |
|   WorkflowEngine (task DAG) | LangGraph (agent graph)        |
+-------------------------------------------------------------+
| Data Layer                                                   |
|   MemoryManager - SessionManager - MCP Client                |
|   HybridProvider (ChromaDB + SQLite FTS5 + Neo4j)            |
+-------------------------------------------------------------+
```

**Key components:**

- **LangGraph StateGraph** serves as the execution engine, compiling agent workflows into directed graphs with conditional routing.
- **Dual-mode router** (keyword-based + LLM-based) classifies user queries into categories (quick, research, analysis, planning, standards) and activates the appropriate agent pipeline.
- **Hybrid memory** combines vector similarity search (ChromaDB), full-text lexical search (SQLite FTS5), and graph-based retrieval (Neo4j) with cross-encoder re-ranking via Reciprocal Rank Fusion (RRF).
- **Tree-of-Thought (ToT) planner** generates multiple reasoning paths for complex queries and selects the optimal plan before execution.
- **MCP client** provides a standardized protocol layer for tool interoperability across heterogeneous BIM/CAD applications.

---

## Project Structure

```
.
├── orchestrator/                  # Main framework package
│   ├── cli/                       # Command-line interface
│   │   ├── main.py                # CLI entry point (Typer)
│   │   ├── chat_orchestrator.py   # Interactive chat session manager
│   │   ├── graph_adapter.py       # LangGraph execution adapter
│   │   ├── display_adapter.py     # Output rendering
│   │   └── events.py              # Event system
│   ├── core/                      # Core orchestration logic
│   │   ├── orchestrator.py        # Main Orchestrator class
│   │   ├── config.py              # Pydantic configuration models
│   │   ├── exceptions.py          # Custom exception hierarchy
│   │   ├── error_handler.py       # Categorized error handling
│   │   ├── initializer.py         # Component initialization
│   │   ├── executor.py            # Task execution engine
│   │   └── lifecycle.py           # Lifecycle callbacks
│   ├── memory/                    # Memory management
│   │   ├── memory_manager.py      # Provider coordinator
│   │   └── providers/             # Storage backends
│   │       ├── hybrid_provider.py # Vector + Lexical + Graph
│   │       ├── mem0_provider.py   # Mem0AI integration
│   │       └── rag_provider.py    # Simple in-memory provider
│   ├── factories/                 # Factory pattern components
│   │   ├── agent_factory.py       # Agent creation with templates
│   │   ├── task_factory.py        # Task creation
│   │   ├── graph_factory.py       # StateGraph compilation
│   │   └── route_classifier.py    # Query routing logic
│   ├── planning/                  # Planning subsystem
│   │   ├── tot_planner.py         # Tree-of-Thought planner
│   │   ├── tot_graph_planner.py   # ToT with LangGraph integration
│   │   └── graph_compiler.py      # Graph compilation utilities
│   ├── tools/                     # Agent tools
│   │   └── graph_rag_tool.py      # GraphRAG retrieval tool
│   ├── integrations/              # Framework integrations
│   │   └── langchain_integration.py  # LangChain/LangGraph setup
│   ├── ingestion/                 # Document ingestion pipeline
│   ├── session/                   # Session persistence
│   ├── mcp/                       # Model Context Protocol client
│   └── workflow/                  # Workflow engine
├── agents/                        # Specialized agent implementations
│   └── bim_ir/                    # BIM Information Retrieval agent
│       ├── blocks/                # NLU/NLG pipeline blocks
│       ├── models/                # Data models
│       ├── tests/                 # Agent-specific tests
│       └── llm/                   # LLM client abstraction
├── mem0/                          # Mem0AI submodule (long-term memory)
├── tree-of-thought-llm/           # ToT reasoning submodule
└── pyproject.toml                 # Package configuration
```

---

## Prerequisites

- **Python 3.11** (exact version required)
- **OpenAI API key** (for LLM and embedding operations)
- **uv** package manager (recommended) or pip
- **Neo4j** (optional, for graph-enhanced memory)

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CodigoTFG

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install core dependencies
uv pip install -e .

# Install with graph support (Neo4j)
uv pip install -e ".[mem0-graph]"

# Install development and test dependencies
uv pip install -e ".[test,dev]"
```

---

## Configuration

### Environment Variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Required variables:

```
OPENAI_API_KEY=sk-...
```

Optional variables:

```
# Memory configuration
ORCH_MEMORY_PROVIDER=hybrid          # hybrid | mem0 | rag
HYBRID_EMBEDDER_PROVIDER=openai
HYBRID_EMBEDDER_MODEL=text-embedding-3-small

# Neo4j (for graph-enhanced retrieval)
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>

# Session tracking
ORCH_USER_ID=default_user
```

### Programmatic Configuration

```python
from orchestrator.core.config import (
    OrchestratorConfig,
    AgentConfig,
    MemoryConfig,
    MemoryProvider,
)

config = OrchestratorConfig(
    name="MyOrchestrator",
    process="workflow",
    user_id="user_001",
    memory=MemoryConfig(
        provider=MemoryProvider.HYBRID,
        use_embedding=True,
    ),
    agents=[
        AgentConfig(
            name="Researcher",
            role="Research Specialist",
            goal="Gather and synthesize information",
            backstory="Expert researcher with domain knowledge",
            instructions="Use available tools to find relevant data",
        ),
    ],
)
```

---

## Usage

### Interactive Chat

```bash
# Start with default settings (hybrid memory, GPT-4o-mini)
python -m orchestrator.cli chat

# Specify memory provider and model
python -m orchestrator.cli chat --memory-provider hybrid --llm gpt-4o

# Enable verbose logging
python -m orchestrator.cli chat -v

# Specify user identity for session tracking
python -m orchestrator.cli chat --user-id "engineer_001"
```

### Memory Information

```bash
# Check memory provider status and configuration
python -m orchestrator.cli memory-info --memory-provider hybrid
```

### Programmatic API

```python
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.config import OrchestratorConfig

# Create from configuration
config = OrchestratorConfig.from_file("config.yaml")
orchestrator = Orchestrator(config)
orchestrator.initialize()

# Execute a query
result = orchestrator.run_sync()

# Cleanup
orchestrator.cleanup()
```

---

## Memory Providers

The framework supports three memory backends, selected via `MemoryProvider` enum:

### Hybrid Provider (Default)

Combines three retrieval strategies with Reciprocal Rank Fusion (RRF):

| Component     | Technology     | Purpose                                    |
| ------------- | -------------- | ------------------------------------------ |
| Vector store  | ChromaDB       | Semantic similarity search via embeddings  |
| Lexical index | SQLite FTS5    | Full-text keyword search with BM25 scoring |
| Graph store   | Neo4j / KuzuDB | Relationship-aware contextual retrieval    |

Results from all three sources are fused using RRF and optionally re-ranked with a cross-encoder model.

Storage paths:

- Vector: `.orchestrator/hybrid_chroma/`
- Lexical: `.orchestrator/hybrid_lexical.db`

### Mem0AI Provider

Delegates to the Mem0AI framework for long-term memory with graph knowledge representation.

### Simple Provider (RAG)

In-memory provider with substring matching. Useful for testing and lightweight deployments without external storage dependencies.

---

## Agent System

### Factory Pattern

Agents are created through an `AgentFactory` that supports six built-in templates:

| Template     | Role                 | Capabilities                      |
| ------------ | -------------------- | --------------------------------- |
| orchestrator | Coordinator          | Delegates tasks, manages workflow |
| researcher   | Information Gatherer | Web search, document retrieval    |
| planner      | Strategic Planner    | Task decomposition, planning      |
| implementer  | Developer            | Code generation, technical tasks  |
| tester       | Quality Assurance    | Test strategy, validation         |
| writer       | Documentation        | Technical writing, reports        |

Agent type is inferred automatically from the configuration fields (name, role, goal, backstory) or can be specified explicitly.

### Query Routing

The `RouteClassifier` categorizes incoming queries into execution paths:

- **quick**: Factual questions with direct answers
- **research**: Information gathering requiring search tools
- **analysis**: Deep investigation requiring multiple agents
- **planning**: Complex tasks requiring Tree-of-Thought decomposition
- **standards**: Domain-specific compliance and standards checks

### GraphRAG Tool

Agents can use the `graph_rag_lookup` tool (a LangChain `StructuredTool`) to query the hybrid memory system during execution. The tool accepts:

- `query`: Natural language search query
- `tags`: Comma-separated filter tags
- `documents`: Comma-separated document IDs to prioritize
- `sections`: Comma-separated section identifiers
- `top_k`: Maximum number of fragments to return

---

## Tree-of-Thought Planning

For complex or ambiguous queries, the system activates a Tree-of-Thought (ToT) planner that:

1. Decomposes the query into sub-problems
2. Generates multiple candidate reasoning paths (branches)
3. Evaluates each path using LLM-based scoring
4. Selects the optimal plan and compiles it into a LangGraph execution graph

The ToT system is implemented in two variants:

- `tot_planner.py`: Standalone planner with configurable branching
- `tot_graph_planner.py`: LangGraph-integrated planner that compiles plans into executable StateGraphs

---

## BIM Information Retrieval Agent

The `agents/bim_ir/` module implements a specialized NLU/NLG pipeline for BIM data queries, structured as five sequential processing blocks:

| Block                    | Module                 | Function                                                                            |
| ------------------------ | ---------------------- | ----------------------------------------------------------------------------------- |
| 1. Intent Classification | `intent_classifier.py` | Identifies query intent (quantity extraction, property lookup, spatial query, etc.) |
| 2. Parameter Extraction  | `param_extractor.py`   | Extracts structured parameters from natural language                                |
| 3. Value Resolution      | `value_resolver.py`    | Maps extracted values to IFC schema entities                                        |
| 4. Retrieval             | `retriever.py`         | Queries the BIM data source for matching elements                                   |
| 5. Summarization         | `summarizer.py`        | Generates natural language responses from results                                   |

The agent includes 8 test files covering each block individually and as an integrated pipeline.

---

## Testing

```bash
# Run all orchestrator tests
uv run pytest orchestrator/ -v

# Run specific test suites
uv run pytest orchestrator/core/tests/ -v                       # Core tests (117 tests)
uv run pytest orchestrator/factories/tests/ -v                  # Factory tests (39 tests)
uv run pytest orchestrator/factories/tests/test_agent_factory.py -v
uv run pytest orchestrator/factories/tests/test_graphrag_tool_integration.py -v

# Run BIM-IR agent tests
uv run pytest agents/bim_ir/tests/ -v

# Run planning tests
uv run pytest orchestrator/planning/tests/ -v
```

---

## Technical Details

### Core Dependencies

| Package               | Version           | Purpose                             |
| --------------------- | ----------------- | ----------------------------------- |
| langchain             | >=0.3.0           | LLM framework and tool abstractions |
| langgraph             | >=0.6.0           | StateGraph execution engine         |
| langchain-openai      | >=0.2.1           | OpenAI model integration            |
| mcp                   | >=1.0.0, <1.18.0  | Model Context Protocol client       |
| chromadb              | >=1.0.21          | Vector storage                      |
| mem0ai[graph]         | >=0.1.117         | Long-term memory with graph support |
| sentence-transformers | >=5.1.0           | Local embedding models              |
| pydantic              | >=2.7.3, <=2.10.1 | Configuration validation            |
| rich                  | >=13.7            | Terminal UI rendering               |

### System Constants

| Constant                     | Value | Description                    |
| ---------------------------- | ----- | ------------------------------ |
| MAX_EXECUTION_DEPTH          | 100   | Maximum graph traversal depth  |
| DEFAULT_MAX_ITERATIONS       | 25    | Default agent iteration limit  |
| GRAPH_CANDIDATE_LIMIT        | 5     | Default GraphRAG result count  |
| TOT_MAX_BRANCHES             | 4     | Maximum ToT reasoning branches |
| DEFAULT_MAX_CONCURRENT_TASKS | 5     | Parallel task execution limit  |

### Python Version

This project requires Python 3.11 exactly (`requires-python = "==3.11.*"`).

---

## License

This project was developed as an academic work (TFG) at the Universidad Pontificia Comillas ICAI
