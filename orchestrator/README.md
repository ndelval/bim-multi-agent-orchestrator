# Orchestrator - Scalable Agent Orchestration System

A class-based architecture for building and managing multi-agent orchestrator systems with support for parallel execution, conditional routing, and multiple memory providers.

## Overview

The Orchestrator package provides a scalable alternative to function-based orchestrator implementations, offering:

- **Modular Architecture**: Clear separation of concerns with specialized components
- **Configuration-Driven**: YAML/JSON configuration with validation
- **Template System**: Extensible agent and task templates
- **Multiple Memory Providers**: RAG, MongoDB, and Mem0 support
- **Workflow Engine**: DAG execution with parallel processing
- **Comprehensive Monitoring**: Real-time metrics and callbacks
- **Easy Testing**: Each component testable independently

## Quick Start

### Basic Usage

```python
from orchestrator import Orchestrator

# Create with default configuration
orchestrator = Orchestrator.create_default("MyOrchestrator")

# Run the workflow
result = orchestrator.run_sync()
print(f"Workflow completed: {result}")
```

### Configuration-Based Usage

```python
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig

# Create configuration
config = OrchestratorConfig(name="CustomOrchestrator")

# Add agents
config.agents.append(AgentConfig(
    name="Researcher",
    role="Research Specialist",
    goal="Gather comprehensive information",
    backstory="Expert researcher with deep analytical skills",
    instructions="Conduct thorough research with multiple sources",
    tools=["duckduckgo"]
))

# Add tasks  
config.tasks.append(TaskConfig(
    name="research_task",
    description="Research the given topic thoroughly",
    expected_output="Comprehensive research report",
    agent_name="Researcher",
    is_start=True
))

# Create and run orchestrator
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()
```

### File-Based Configuration

```python
# Save configuration
orchestrator.export_config("my_orchestrator.yaml")

# Load from file
orchestrator = Orchestrator.from_file("my_orchestrator.yaml")
result = orchestrator.run_sync()
```

## Architecture

### Core Components

1. **OrchestratorConfig**: Configuration management with validation
2. **AgentFactory**: Dynamic agent creation with registry pattern
3. **TaskFactory**: Task management with dependency resolution
4. **MemoryManager**: Memory provider abstraction
5. **WorkflowEngine**: DAG execution with parallel processing
6. **Orchestrator**: Main coordination class

### Directory Structure

```
orchestrator/
├── __init__.py
├── core/
│   ├── orchestrator.py          # Main Orchestrator class
│   ├── config.py               # Configuration management
│   └── exceptions.py           # Custom exceptions
├── factories/
│   ├── agent_factory.py        # Agent creation and templates
│   └── task_factory.py         # Task creation and dependencies
├── memory/
│   ├── memory_manager.py       # Memory provider abstraction
│   └── providers/              # Memory provider implementations
├── workflow/
│   └── workflow_engine.py      # Workflow execution engine
└── templates/
    ├── agents/                 # Agent templates (future)
    └── tasks/                  # Task templates (future)
```

## Features

### Agent Templates

Built-in agent templates include:

- **OrchestratorAgent**: Project coordination and management
- **ResearcherAgent**: Web research and information gathering
- **PlannerAgent**: Strategic planning and design
- **ImplementerAgent**: Solution implementation
- **TesterAgent**: Quality assurance and validation
- **WriterAgent**: Documentation and reporting

### Task Templates

Built-in task templates include:

- **ResearchTask**: Information gathering and analysis
- **PlanningTask**: Strategic planning and design
- **ImplementationTask**: Solution development
- **TestingTask**: Quality validation
- **ReviewTask**: Decision-making and approval
- **DocumentationTask**: Report generation

### Memory Providers

Supported memory providers:

- **RAG**: Local storage with embedding search
- **MongoDB**: Scalable document storage with vector search
- **Mem0**: Graph-based memory with relationships

### Workflow Execution

- **Sequential**: Tasks execute in order
- **Parallel**: Independent tasks execute concurrently
- **Mixed**: Adaptive execution strategy
- **DAG Support**: Complex dependency graphs
- **Error Handling**: Retries with exponential backoff
- **Progress Tracking**: Real-time metrics and callbacks

## Configuration

### Memory Configuration

```python
from orchestrator.core.config import MemoryConfig, MemoryProvider

# RAG provider
memory_config = MemoryConfig(
    provider=MemoryProvider.RAG,
    use_embedding=True,
    rag_db_path="./memory/chroma_db"
)

# MongoDB provider
memory_config = MemoryConfig(
    provider=MemoryProvider.MONGODB,
    use_embedding=True,
    config={
        "connection_string": "mongodb://localhost:27017",
        "database": "orchestrator_memory"
    }
)

# Mem0 provider
memory_config = MemoryConfig(
    provider=MemoryProvider.MEM0,
    use_embedding=True,
    config={
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": "neo4j://localhost:7687",
                "username": "neo4j",
                "password": "password"
            }
        }
    }
)
```

### Execution Configuration

```python
from orchestrator.core.config import ExecutionConfig, ProcessType

execution_config = ExecutionConfig(
    process=ProcessType.WORKFLOW,
    verbose=1,
    max_iter=10,
    memory=True,
    async_execution=True,
    user_id="user_123"
)
```

## Advanced Usage

### Custom Agent Templates

```python
from orchestrator.factories.agent_factory import BaseAgentTemplate
from orchestrator.core.config import AgentConfig

class CustomAgentTemplate(BaseAgentTemplate):
    @property
    def agent_type(self) -> str:
        return "custom_agent"
    
    def get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="CustomAgent",
            role="Custom Specialist",
            goal="Perform custom operations",
            backstory="Specialized agent for custom tasks",
            instructions="Follow custom protocols"
        )
    
    def create_agent(self, config: AgentConfig, **kwargs):
        # Custom agent creation logic
        pass

# Register template
orchestrator.register_agent_template(CustomAgentTemplate())
```

### Custom Task Templates

```python
from orchestrator.factories.task_factory import BaseTaskTemplate
from orchestrator.core.config import TaskConfig

class CustomTaskTemplate(BaseTaskTemplate):
    @property
    def task_type(self) -> str:
        return "custom_task"
    
    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="custom_task",
            description="Perform custom operations",
            expected_output="Custom task results",
            agent_name=agent_name
        )
    
    def create_task(self, config: TaskConfig, agent, **kwargs):
        # Custom task creation logic
        pass

# Register template
orchestrator.register_task_template(CustomTaskTemplate())
```

### Monitoring and Callbacks

```python
def on_workflow_start():
    print("Workflow started")

def on_task_complete(task_name, execution):
    print(f"Task {task_name} completed in {execution.duration}s")

def on_workflow_complete(metrics):
    print(f"Workflow completed in {metrics.total_duration}s")
    print(f"Parallel efficiency: {metrics.parallel_efficiency:.2f}")

orchestrator.on_workflow_start = on_workflow_start
orchestrator.on_task_complete = on_task_complete
orchestrator.on_workflow_complete = on_workflow_complete

result = orchestrator.run_sync()
```

### Dynamic Agent/Task Management

```python
# Add agent at runtime
custom_agent = AgentConfig(
    name="RuntimeAgent",
    role="Dynamic Agent",
    goal="Handle runtime tasks",
    backstory="Created dynamically",
    instructions="Adapt to runtime requirements"
)
orchestrator.add_agent(custom_agent)

# Add task at runtime
runtime_task = TaskConfig(
    name="runtime_task",
    description="Task created at runtime",
    expected_output="Runtime results",
    agent_name="RuntimeAgent"
)
orchestrator.add_task(runtime_task)
```

## Migration from Function-Based Approach

### Before (Function-Based)

```python
def build_orchestrator_system():
    # 280+ lines of mixed configuration and logic
    orchestrator = Agent(...)
    researcher = Agent(...)
    # ... more agent creation
    
    research_task = Task(...)
    plan_task = Task(...)
    # ... more task creation
    
    system = PraisonAIAgents(
        agents=[...],
        tasks=[...],
        # ... configuration mixed with setup
    )
    return system

system = build_orchestrator_system()
result = asyncio.run(system.astart())
```

### After (Class-Based)

```python
# Simple migration
orchestrator = Orchestrator.create_default("MigratedOrchestrator")
result = orchestrator.run_sync()

# Or configuration-based
config = OrchestratorConfig.from_file("orchestrator.yaml")
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()
```

## Benefits

### Compared to Function-Based Approach

| Aspect | Function-Based | Class-Based |
|--------|---------------|-------------|
| **Organization** | Monolithic (280+ lines) | Modular components |
| **Configuration** | Mixed with logic | Separate validation |
| **Testing** | Hard to test components | Each component testable |
| **Extension** | Modify function code | Template registration |
| **Error Handling** | Basic try/catch | Comprehensive system |
| **Persistence** | Code only | File-based configs |
| **Monitoring** | Manual logging | Built-in metrics |
| **Memory Options** | Single provider | Multiple providers |
| **Workflow Control** | Basic execution | Advanced DAG engine |

### Key Advantages

1. **Maintainability**: Clear separation of concerns
2. **Testability**: Unit test individual components
3. **Configurability**: Environment-specific configurations
4. **Extensibility**: Plugin architecture for custom components
5. **Monitoring**: Built-in metrics and callback system
6. **Scalability**: Support for complex dependency graphs
7. **Flexibility**: Multiple memory and execution options

## Error Handling

The system provides comprehensive error handling with custom exceptions:

```python
from orchestrator.core.exceptions import (
    OrchestratorError,
    ConfigurationError,
    AgentCreationError,
    TaskExecutionError,
    MemoryError,
    WorkflowError
)

try:
    orchestrator = Orchestrator.from_file("config.yaml")
    result = orchestrator.run_sync()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except WorkflowError as e:
    print(f"Workflow error: {e}")
except OrchestratorError as e:
    print(f"General orchestrator error: {e}")
```

## Testing

Each component can be tested independently:

```python
import pytest
from orchestrator.factories.agent_factory import AgentFactory
from orchestrator.core.config import AgentConfig

def test_agent_creation():
    factory = AgentFactory()
    config = AgentConfig(
        name="TestAgent",
        role="Test Role",
        goal="Test Goal", 
        backstory="Test Backstory",
        instructions="Test Instructions"
    )
    
    agent = factory.create_agent(config)
    assert agent.name == "TestAgent"
    assert agent.role == "Test Role"
```

## Performance

The class-based architecture provides several performance benefits:

- **Parallel Execution**: Automatic parallel task execution
- **Memory Efficiency**: Configurable memory providers
- **Resource Management**: Connection pooling and cleanup
- **Caching**: Template and configuration caching
- **Monitoring**: Real-time performance metrics

## Examples

See `orchestrator_migration_example.py` for complete examples showing:

1. Original function-based approach
2. Simple class-based migration
3. Configuration-driven setup
4. File-based configuration
5. Advanced features demonstration
6. Benefits comparison

## Contributing

1. Follow the existing code structure
2. Add tests for new components
3. Update documentation
4. Follow typing conventions
5. Add error handling

## License

This project is part of the PraisonAI ecosystem and follows the same licensing terms.
### Mem0 Conversational Memory (Local Graph)

Requirements:
- pip install mem0ai "mem0ai[graph]"
- A local graph DB (Neo4j or Memgraph). Example env vars in `.env.example`.

Run the demo script:

```
python examples/memory/test_mem0_memory.py
```

It will:
- Initialize Mem0 with your local graph store
- Boot a minimal orchestrator with PraisonAI Agents using Mem0 memory
- Store a few partitioned memories (user/agent/run) and retrieve them

### Optional Vector Store: Qdrant

Why both Neo4j/Memgraph and Qdrant?
- Graph DB (Neo4j/Memgraph): relaciones/topología (impactos, rutas, dependencias). Ideal para preguntas multi-hop.
- Vector DB (Qdrant): recuperación semántica por similitud de texto. Ideal para encontrar pasajes/fragmentos relevantes.
- Juntos (Graph-RAG + Vector): mejor cobertura: primero ubicas entidades/rutas, luego traes evidencia textual precisa.

Run Qdrant locally:
```
docker run -it --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Enable Qdrant in Mem0 config via env (see .env.example):
```
export MEM0_VECTOR_PROVIDER=qdrant
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
# For Qdrant Cloud (optional): QDRANT_URL, QDRANT_API_KEY, QDRANT_USE_HTTPS=true
```

The demo `examples/memory/test_mem0_memory.py` will inject:
```
"vector_store": {
  "provider": "qdrant",
  "config": {"host": "localhost", "port": 6333, "collection_name": "mem0_memory"}
}
```
Mem0 creates/uses the collection automatically; ensure your embedding dims match the model (OpenAI `text-embedding-3-large` = 3072).

### Hybrid RAG (Vector + Léxico + Rerank)

The CLI also expone un proveedor `hybrid` totalmente local que junta:

- **ChromaDB** para búsqueda vectorial (cosine)
- **SQLite FTS5** para ranking BM25 (según [SQLite FTS5 docs](https://www.sqlite.org/fts5.html))
- **Cross-encoder opcional** con modelos de `sentence-transformers` para reranking final

Actívalo mediante `ORCH_MEMORY_PROVIDER=hybrid` o `python -m orchestrator.cli chat --memory-provider hybrid`.

Variables de entorno relevantes:

```
# Embedder (por defecto OpenAI text-embedding-3-small)
export HYBRID_EMBEDDER_PROVIDER=openai
export HYBRID_EMBEDDER_MODEL=text-embedding-3-small

# Vector store (Chroma) persistente
export HYBRID_VECTOR_PATH=.praison/hybrid_chroma
export HYBRID_VECTOR_COLLECTION=hybrid_memory

# Índice léxico (SQLite FTS5)
export HYBRID_LEXICAL_DB_PATH=.praison/hybrid_lexical.db
export HYBRID_LEXICAL_TTL_SECONDS=2592000      # opcional, TTL en segundos
export HYBRID_LEXICAL_MAX_ENTRIES=20000        # opcional, límite de documentos
export HYBRID_LEXICAL_CLEANUP_INTERVAL=200     # cada cuántas escrituras compactar

# Rerank cross-encoder (si sentence-transformers está instalado)
export HYBRID_RERANK_ENABLED=true
export HYBRID_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export HYBRID_RERANK_TOP_K=10
export HYBRID_RERANK_MAX_WORKERS=2
```

El reranker se ejecuta en un `ThreadPoolExecutor` con modelos y puntuaciones cacheadas para no bloquear el CLI. Si `sentence-transformers` (y `torch`) no están disponibles, el sistema continúa sin rerank cruzado.

### GraphRAG Tooling para Agentes

El CLI expone un tool reusable `graph_rag_lookup` respaldado por el `MemoryManager` híbrido:

- Combina filtros de Neo4j (tags, documentos, secciones) con búsqueda vectorial y ranking BM25.
- El router agrega una ruta `standards` atendida por `StandardsAgent` para preguntas normativas.
- Los agentes `Researcher`, `Analyst` y `Planner` reciben instrucciones para invocar `graph_rag_lookup` antes de responder y citar documento / sección / URL.

Ejecución típica:

```bash
uv run python -m orchestrator.cli chat --memory-provider hybrid
```

Establece `HYBRID_GRAPH_ENABLED=false` (o usa `--memory-provider mem0`) si quieres desactivar GraphRAG.

### Pipelines de Ingestión

El paquete `orchestrator/ingestion/` incluye:

- `sample_loader.py`: procesa PDFs, genera metadatos y publica en vector/BM25/Neo4j.
- `ingest_log.py`: registro SQLite que evita reingestas cuando el hash del fichero es idéntico.
- `metadata_loader.py`: carga manifiestos JSON/YAML.
- `templates/`: plantillas para estándares, manuales, cursos, papers, catálogos, tickets, reportes de compliance, etc.
- `run_graph_sync.py`: resincroniza Neo4j a partir del almacenamiento local.

#### Manifiestos y Plantillas

Puedes mantener manifiestos por documento (p. ej. `manifests/ISO_9001.yaml`) reutilizando las plantillas. Campos soportados: `discipline`, `doc_type`, `version`, `effective_date`, `tags`, `references`, `retention_days`, etc.

#### Ingestar PDFs

```bash
uv add pypdf  # instalar una vez
uv run python -m orchestrator.ingestion.sample_loader \
  --input docs/ISO_1234.pdf \
  --metadata manifests/ \
  --discipline ingenieria --version 2024 \
  --user cli-user --run cli-run \
  --memory-provider hybrid
```

- `--metadata` puede apuntar a un directorio (el loader buscará `filename.yaml/.json`) o a un fichero concreto.
- Los flags de CLI sobrescriben los campos del manifiesto (`--title`, `--tags`, etc.).
- `--retention-days` establece `expires_at` y el mantenedor FTS lo respeta junto con el TTL global.
- `--force` reingesta aunque el hash no cambie.

#### Sincronizar el Grafo

```bash
uv run python -m orchestrator.ingestion.run_graph_sync --memory-provider hybrid
```

Reconstruye nodos `Document`, `Chunk`, `Section`, `Tag` y relaciones `BELONGS_TO`, `PART_OF`, `HAS_TAG`, `REFERENCES` a partir del almacenamiento híbrido.

### Acceso Programático

- `MemoryManager.create_graph_tool()` devuelve el callable `graph_rag_lookup` para integrarlo en tus propios agentes.
- `MemoryManager.retrieve_with_graph(...)` permite consultas directas combinadas (grafo + vector/lexical).
- `MemoryManager.sync_graph()` y `run_graph_sync.py` resincronizan Neo4j tras cambios manuales.
- `Orchestrator.create_graph_tool()` expone el mismo tool para workflows basados en clases.
