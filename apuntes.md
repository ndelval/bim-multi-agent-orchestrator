# Analisis Completo del Proyecto: AI Agents Unified Framework

## FASE 1: MAPEO ESTRUCTURAL

### Stacks y Tecnologias

| Capa              | Tecnologias                                              |
| ----------------- | -------------------------------------------------------- |
| Lenguaje          | Python 3.11 (estricto `==3.11.*`)                        |
| Orquestacion LLM  | LangChain >= 0.3.0, LangGraph >= 0.6.0                   |
| LLMs              | OpenAI (GPT-4o, GPT-4o-mini), Ollama, Groq via LangChain |
| Memoria vectorial | ChromaDB >= 1.0.21                                       |
| Memoria lexica    | SQLite FTS5 (BM25)                                       |
| Memoria grafo     | Neo4j (bolt), Memgraph via Mem0                          |
| Reranking         | sentence-transformers >= 5.1.0 (cross-encoder)           |
| MCP               | mcp >= 1.0.0, < 1.18.0                                   |
| CLI/Display       | Rich >= 13.7                                             |
| Config/Validacion | Pydantic >= 2.7.3, PyYAML                                |
| Build             | Hatchling, uv (package manager)                          |
| Linting           | Ruff                                                     |
| Testing           | Pytest + pytest-asyncio + pytest-mock                    |

### Modulos Principales

```
CodigoTFG/
├── orchestrator/              # Sistema principal de orquestacion multi-agente
│   ├── core/                  # Orchestrator, Config, Exceptions, Error handling
│   ├── factories/             # AgentFactory, TaskFactory, GraphFactory, RouteClassifier
│   ├── integrations/          # LangChain/LangGraph wrappers (OrchestratorState, LangChainAgent)
│   ├── memory/                # MemoryManager + 3 providers (hybrid, mem0, rag)
│   ├── workflow/              # WorkflowEngine (DAG-based execution)
│   ├── planning/              # Tree-of-Thought planner (ToT integration)
│   ├── cli/                   # ChatOrchestrator, Rich display, event system
│   ├── session/               # SessionManager, SessionStore, UserContext
│   ├── mcp/                   # MCPClientManager, STDIO transport, tool adapters
│   ├── tools/                 # GraphRAG lookup tool
│   └── ingestion/             # PDF extraction, chunking, graph sync pipeline
├── mem0/                      # Submodulo: Memory layer for AI (vector + graph)
├── tree-of-thought-llm/       # Submodulo: Advanced reasoning system
└── agents/bim_ir/             # Agente domain-specific BIM-IR
```

### Puntos de Entrada

| Punto de entrada                                 | Archivo                                   | Proposito                           |
| ------------------------------------------------ | ----------------------------------------- | ----------------------------------- |
| `python -m orchestrator.cli chat`                | `orchestrator/cli/main.py`                | Chat interactivo multi-agente       |
| `python -m orchestrator.cli memory-info`         | `orchestrator/cli/main.py`                | Info del provider de memoria        |
| `python -m orchestrator.ingestion.sample_loader` | `orchestrator/ingestion/sample_loader.py` | Ingesta de PDFs a memoria           |
| `Orchestrator(config).run_sync()`                | `orchestrator/core/orchestrator.py`       | API programatica de workflows       |
| `Orchestrator.create_default()`                  | `orchestrator/core/orchestrator.py`       | Creacion rapida con agentes default |

---

## FASE 2: MODELO DE DOMINIO Y FLUJOS

### Dominio del Problema

El sistema resuelve la **orquestacion de multiples agentes LLM** con:

- **Routing inteligente**: clasificar queries y dirigirlas al agente apropiado
- **Memoria persistente tri-capa**: busqueda semantica + lexica + grafo
- **Planificacion avanzada**: Tree-of-Thought para descomposicion de tareas complejas
- **Ejecucion como grafo**: LangGraph StateGraph para flujos con dependencias

### Entidades Principales

- `OrchestratorConfig` - Configuracion raiz del sistema
- `AgentConfig` - Definicion de un agente (rol, herramientas, LLM)
- `TaskConfig` - Definicion de una tarea (descripcion, agente, dependencias)
- `LangChainAgent` - Wrapper agente con LLM + tools + system prompt
- `OrchestratorState` - Estado mutable del StateGraph durante ejecucion
- `MemoryManager` / `IMemoryProvider` - Interfaz y providers de memoria
- `SessionManager` / `SessionContext` - Gestion de sesiones multi-turno
- `RouterDecision` - Decision del router (ruta, confianza, razonamiento)

### Flujo 1: Chat Interactivo (flujo principal)

```
1. CLI main.py: parse args (--memory-provider, --backend, --llm)
2. ChatOrchestrator.__init__(args)
3. .run():
   3a. _setup_console()          → Rich Console + welcome panel
   3b. _initialize_memory()      → MemoryManager(MemoryConfig) → HybridRAGMemoryProvider
   3c. _initialize_session()     → SessionManager.create_session(user_id)
   3d. _build_agents()           → _build_chat_agents() → 5 AgentConfigs:
       - Router (clasifica queries)
       - Researcher (busqueda web)
       - Analyst (analisis profundo)
       - Planner (planificacion ToT)
       - StandardsAgent (normativas)
   3e. _setup_backend_adapter()  → GraphAgentAdapter(memory_manager, llm)
4. LOOP: leer input usuario
   4a. _handle_router_phase(query):
       - adapter.run_single_agent(router_config, query)
       - _extract_text() → _parse_router_payload() → RouterDecision
   4b. _handle_execution_phase(decision, query):
       - "quick"     → run_single_agent(researcher)
       - "analysis"  → run_multi_agent_workflow([Researcher, Analyst, Standards])
       - "planning"  → (pendiente ToT) → fallback multi-agent
   4c. _display_result(answer)
   4d. _store_conversation_turn(query, answer, decision):
       - memory.store(user_query + metadata)
       - memory.store(assistant_answer + metadata)
       - memory.store(combined_pair + metadata)
5. EXIT: session.end_session() → cleanup
```

### Flujo 2: Ejecucion Programatica (Orchestrator API)

```
1. Orchestrator(config: OrchestratorConfig)
2. .initialize():
   2a. MemoryManager(config.memory)
   2b. WorkflowEngine(process_type, max_concurrent=5, retries=3)
   2c. _create_agents():  → AgentFactory.create_agent(cfg) para cada AgentConfig
   2d. _create_langgraph_system():
       - GraphFactory(agent_factory)
       - _create_dynamic_tools() → GraphRAG tool si memory disponible
       - register_dynamic_tools({graphrag: tool})
       - _enrich_agent_configs_with_tools()
       - _create_stategraph():
         * Si hay tasks → create_workflow_graph(config) → router + agents + final_output
         * Si solo agents → create_chat_graph(agents, router)
3. .run_sync() → asyncio.run(.run()):
   3a. _build_recall_content() → memory.retrieve_filtered(query)
   3b. _run_langgraph_workflow(recall):
       - OrchestratorState(messages=[HumanMessage], input_prompt, memory_context)
       - compiled_graph.invoke(initial_state)
       - Extraer final_output o ultimo AIMessage
```

### Flujo 3: Construccion del StateGraph (GraphFactory)

```
1. create_workflow_graph(config):
   1a. StateGraph(OrchestratorState)
   1b. Para cada AgentConfig habilitado:
       - _create_agent_from_config(cfg) → LangChainAgent (con dynamic tools)
       - _create_agent_function(agent, cfg) → closure que ejecuta agent.execute()
       - workflow.add_node(node_name, agent_func)
   1c. workflow.add_node("router", _create_router_function())
   1d. workflow.add_node("final_output", _create_completion_function())
   1e. _add_workflow_edges():
       - router → conditional_edges(route_condition) → target agent
       - cada agent → final_output
       - final_output → END
   1f. workflow.set_entry_point("router")
   1g. return workflow  (se compila externamente con .compile())
```

### Flujo 4: Busqueda Hibrida en Memoria

```
1. memory_manager.retrieve(query, limit=10)
   → hybrid_provider.retrieve(query, limit):
2. En paralelo:
   2a. _search_vector(query, limit*2):
       - _generate_embedding(query) → OpenAI embeddings
       - chromadb.collection.query(embedding, n_results)
       - Retorna [{id, content, metadata, score}]
   2b. _search_lexical(query, limit*2):
       - SQLite FTS5: MATCH query con BM25 ranking
       - Retorna [{id, content, metadata, score}]
3. _fuse_results(vector_results, lexical_results, k=60):
   - Reciprocal Rank Fusion: score = sum(1 / (k + rank_i))
4. (Opcional) _rerank(fused_results, query):
   - Cross-encoder reranker (sentence-transformers)
5. Retorna top-N resultados ordenados por score
```

### Flujo 5: Ingesta de Documentos

```
1. sample_loader.py --input doc.pdf --memory-provider hybrid
2. extract_pdf_text(pdf_path) → texto completo (pypdf)
3. chunk_text(text, chunk_size=900, overlap=150) → chunks tokenizados
4. Para cada chunk:
   4a. DocumentMetadata + ChunkMetadata → merge_metadata()
   4b. sanitize_for_chroma(metadata) → limpiar tipos incompatibles
   4c. memory_manager.store(chunk_content, merged_metadata)
       → vector store + lexical index + (opcional) graph
5. ingest_log: registrar hash + timestamp para deduplicacion
```

---

## FASE 3: ANALISIS CRITICO Y CALIDAD

### Hallazgo 1: Planning Path NO IMPLEMENTADO

**Archivo**: `orchestrator/cli/chat_orchestrator.py:362-379`
**Problema**: `_execute_planning_path()` imprime un warning y hace fallback a analysis path. El ToT planner (`tot_planner.py`) existe pero NO esta integrado en el chat flow.
**Impacto**: La ruta "planning" del router es esencialmente identica a "analysis" con un agente extra.
**Recomendacion**: Integrar `generate_plan_with_tot()` en el planning path, o eliminar la ruta "planning" del router si no se va a implementar a corto plazo.

### Hallazgo 2: Duplicacion de RouterDecision

**Archivos**:

- `orchestrator/integrations/langchain_integration.py:111-126` - `RouterDecision` como TypedDict
- `orchestrator/core/value_objects.py` - `RouterDecision` como dataclass
  **Problema**: Dos clases distintas con el mismo nombre, campos diferentes, en modulos diferentes. El TypedDict es para el state de LangGraph, el dataclass para el CLI. Esto crea confusion.
  **Recomendacion**: Unificar en un solo lugar o renombrar para evitar colision de nombres.

### Hallazgo 3: `_create_agent_function` muta estado acumulativamente

**Archivo**: `orchestrator/factories/graph_factory.py:267-330`
**Problema**: Lineas 301-303 construyen `agent_outputs` como `{**state.agent_outputs, config.name: result}` en lugar de retornar solo `{config.name: result}`. Dado que hay un reducer `merge_dicts`, el patron correcto seria retornar solo la contribucion del nodo. Al reconstruir todo el dict, si dos nodos ejecutan en paralelo sobre el mismo snapshot de state, las contribuciones previas de otros nodos paralelos podrian perderse.
**Impacto**: En ejecucion paralela real, estado podria perder outputs de agentes que corrieron concurrentemente.
**Recomendacion**: Cambiar a retornar solo `{"agent_outputs": {config.name: result}}` y dejar que el reducer `merge_dicts` haga el merge.

### Hallazgo 4: `LangChainAgent.execute()` imprime a consola directamente

**Archivo**: `orchestrator/integrations/langchain_integration.py:392-491`
**Problema**: El metodo `execute()` usa `console.print(Panel(...))` directamente con Rich. Esto acopla la logica de ejecucion del agente con la presentacion. Si se usa el agente en un contexto no-CLI (tests, API, batch), se generara output no deseado.
**Recomendacion**: Mover la presentacion al sistema de eventos (`cli/events.py`) o al display adapter. El agente solo deberia retornar datos.

### Hallazgo 5: `create_graph_tool` tiene bug con `run_id`

**Archivo**: `orchestrator/core/orchestrator.py:509-515`

```python
run = run_id or self.config.user_id  # Bug: deberia ser self.config.run_id
```

**Problema**: Cuando `run_id` es None, el fallback usa `user_id` en vez de `config.run_id`.
**Impacto**: El GraphRAG tool podria buscar en memorias del scope incorrecto.
**Recomendacion**: Cambiar a `self.config.run_id`.

### Hallazgo 6: Textos hardcodeados en espanol

**Archivos**: `orchestrator/core/orchestrator.py:792-812`
**Problema**: `_compose_task_description` genera texto en espanol ("Rol del agente", "Prompt actual", "Sigue tus instrucciones base..."). Si el LLM espera instrucciones en ingles, esto puede reducir calidad.
**Impacto medio**: Depende del idioma objetivo del proyecto (parece ser un TFG en espanol, asi que puede ser intencional).
**Recomendacion**: Configurar el idioma de prompts como parametro o constante.

### Hallazgo 7: Modulos refactorizados sin integrar

**Archivos**:

- `orchestrator/core/executor.py` (nuevo, no trackeado)
- `orchestrator/core/initializer.py` (nuevo, no trackeado)
- `orchestrator/core/lifecycle.py` (nuevo, no trackeado)
- `orchestrator/core/orchestrator_refactored.py` (nuevo, no trackeado)
  **Problema**: Estos archivos representan un refactor SRP del Orchestrator en componentes separados, pero el sistema sigue usando `orchestrator.py` original. Los tests en `core/tests/` prueban los modulos refactorizados, pero no estan integrados en el flujo principal.
  **Impacto**: Codigo duplicado y confuso. No queda claro cual es la version "real".
  **Recomendacion**: Decidir si completar la migracion al refactor o eliminarlo.

### Hallazgo 8: Context manager asyncio en cleanup

**Archivo**: `orchestrator/core/orchestrator.py:566-568`

```python
asyncio.run(self.agent_factory.cleanup_mcp())
```

**Problema**: Si `cleanup()` se llama desde un contexto donde ya hay un event loop activo (ej. dentro de `async run()`), `asyncio.run()` lanzara RuntimeError.
**Recomendacion**: Usar `asyncio.get_event_loop().run_until_complete()` o manejar ambos casos.

### Hallazgo 9: RAGMemoryProvider es solo in-memory

**Archivo**: `orchestrator/memory/providers/rag_provider.py`
**Problema**: A pesar de llamarse "RAG", solo hace substring matching en un dict en memoria. No usa embeddings ni ChromaDB (a pesar de lo que dice la documentacion).
**Impacto**: Si alguien selecciona `--memory-provider rag` esperando vector search real, obtendra algo muy inferior.
**Recomendacion**: Renombrar a `SimpleMemoryProvider` o implementar vector search real.

### Hallazgo 10: Tres almacenamientos por turno de conversacion

**Archivo**: `orchestrator/cli/chat_orchestrator.py:407-498`
**Problema**: Cada turno almacena 3 documentos: query, response, y combined. Esto triplica el storage y puede causar resultados duplicados en retrieval.
**Recomendacion**: Almacenar solo el combined pair, o usar metadata para diferenciar sin duplicar contenido.

---

## FASE 4: MAPA DE DECISIONES ARQUITECTONICAS

### Decision 1: LangGraph como motor de ejecucion

- **Razon inferida**: Proporciona grafos de estado compilables con soporte nativo para branching, paralelismo, y checkpointing.
- **Trade-off**: Mayor complejidad de setup vs PraisonAI (eliminado), pero mas control sobre el flujo.

### Decision 2: Memoria hibrida tri-capa (vector + lexical + graph)

- **Razon**: Combina precision semantica (embeddings) con recall exacto (FTS5/BM25) y relaciones (Neo4j).
- **Trade-off**: Complejidad operativa (3 backends), pero el sistema degrada gracefully si alguno falla.

### Decision 3: Factory Pattern para agentes y tareas

- **Razon**: Permite templates reutilizables (Researcher, Planner, etc.) y creacion dinamica.
- **Trade-off**: Indirection vs simplicidad. Funciona bien para el caso de uso actual.

### Decision 4: Router keyword-based por defecto + LLM-based opcional

- **Razon**: Keywords es rapido y predecible; LLM es mas flexible para queries ambiguas.
- **Trade-off**: El router keyword tiene cobertura limitada, pero las rutas son pocas y bien definidas.

### Decision 5: OrchestratorState como dataclass con Annotated reducers

- **Razon**: Permite writes paralelos sin race conditions en campos criticos.
- **Trade-off**: Complejidad en saber que campos son single-writer vs parallel-safe.

---

## FASE 5: EXPLORACION PROFUNDA (Segunda Ronda)

### Flujo 6: GraphAgentAdapter - Puente CLI↔LangGraph

**Archivo**: `orchestrator/cli/graph_adapter.py` (~900 lineas)

El `GraphAgentAdapter` es el componente critico que conecta `ChatOrchestrator` con la ejecucion real de LangGraph.

```
1. run_single_agent(agent_config, user_query):
   1a. SI es RouterAgent:
       - Inyecta instrucciones de formato JSON en el prompt
       - Crea StateGraph minimo: router_node → END
       - Invoca y retorna resultado raw para parsing en ChatOrchestrator
   1b. SI es agente estandar:
       - memory_manager.retrieve(query) → recall_snippets
       - Construye sequential graph: agent_node → END
       - Inyecta memory_context en OrchestratorState
       - compiled_graph.invoke(state) → extrae resultado
   RETORNA: str (texto del agente) o dict (estado completo)

2. run_multi_agent_workflow(agent_sequence, user_query, display_adapter):
   2a. Mapea nombres a AgentConfigs desde _build_chat_agents()
   2b. memory_manager.retrieve(query) → recall_snippets
   2c. execute_route(agent_sequence, query, recall_snippets):
       - tot_planner.generate_graph_with_tot(prompt, agents, assignments)
         → StateGraphSpec con nodos, edges, conditional edges
       - graph_compiler.compile_tot_graph(spec, agents)
         → StateGraph compilado con nodos reales
       - compiled_graph.invoke(OrchestratorState)
   2d. _extract_result_from_state(state):
       - 6 fases de extraccion: final_output → messages → agent_outputs
         → node_outputs → errors → fallback "No output"

3. _extract_result_from_state(state) - Pipeline de 6 fases:
   Fase 1: state.final_output (si no vacio)
   Fase 2: Ultimo AIMessage en state.messages
   Fase 3: Ultimo valor en state.agent_outputs (dict ordenado)
   Fase 4: Ultimo valor en state.node_outputs
   Fase 5: Concatenar errores como fallback
   Fase 6: "No output generated" (fallback final)
```

**Problemas detectados**:

- Retorno mixto (str vs dict) segun el path, dificulta el manejo en ChatOrchestrator
- Sin timeout enforcement - un agente puede bloquear indefinidamente
- LLM model hardcodeado en graph creation, no respeta config global

### Flujo 7: `main.py` - Configuracion de Chat Agents

**Archivo**: `orchestrator/cli/main.py` (498 lineas)

````
_build_chat_agents(memory_manager, use_tools=True):
1. Crea GraphRAG tool desde memory_manager.create_graph_tool()
2. Define 5 agent templates:
   - Router:        gpt-4o-mini, instrucciones JSON output, sin tools
   - Researcher:    gpt-4o-mini, tools=[GraphRAG, DuckDuckGo, Wikipedia]
   - Analyst:       gpt-4o-mini, tools=[GraphRAG]
   - Planner:       gpt-4o-mini, tools=[GraphRAG]
   - StandardsAgent: gpt-4o-mini, tools=[GraphRAG]
3. Retorna tuple de 5 AgentConfig

_parse_router_payload(raw_text) - 4 niveles de fallback:
   Nivel 1: json.loads(text) directo
   Nivel 2: Extraer bloque ```json ... ``` con regex
   Nivel 3: Regex para "decision": "X" o "route": "X"
   Nivel 4: Keyword matching manual ("quick", "research", "analysis", "planning")
````

**Bug critico detectado**: Las herramientas web (`duckduckgo_tool`, `wikipedia_tool`) se importan al inicio del modulo pero las variables siempre son `None` porque la inicializacion esta dentro de un bloque try/except que falla silenciosamente. El Researcher nunca tiene acceso real a busqueda web.

### Componente: RouteClassifier y RoutingConfig

**Archivos**: `orchestrator/factories/route_classifier.py` (381 lineas), `routing_config.py` (276 lineas)

```
RouteClassifier:
- 5 rutas: quick, research, analysis, planning, standards
- RoutingKeywords: diccionario configurable de palabras clave por ruta
- classify_by_keywords(): Prioridad quick > research > planning > analysis > standards
  * Busqueda word-boundary con regex \b{keyword}\b
- classify_from_llm_response(): JSON parse → text extraction → keyword fallback

RoutingStrategy (Strategy Pattern):
- DefaultRoutingStrategy: route → agent name mapping con normalizacion
- ChatRoutingStrategy: directo route → node_name (lowercase)
- RoutingRegistry: singleton con "default" y "chat" pre-registradas
```

**Issue**: La prioridad de `classify_by_keywords()` (quick primero) no coincide con la jerarquia natural de complejidad. Una query con keywords de "quick" Y "planning" siempre ira a quick.

### Componente: WorkflowEngine

**Archivo**: `orchestrator/workflow/workflow_engine.py` (594 lineas)

```
WorkflowEngine:
- DAG-based task execution (complementario a LangGraph, NO legacy)
- Topological sort para determinar orden de ejecucion
- Parallel levels: tareas en el mismo nivel se ejecutan concurrentemente
- Semaphore-controlled concurrency (max_concurrent_tasks=5)
- Retry con exponential backoff: delay * 2^attempt (max 3 retries)
- TaskStatus lifecycle: PENDING → READY → RUNNING → COMPLETED/FAILED/SKIPPED
- Callbacks: on_task_start, on_task_complete, on_task_fail, on_workflow_complete
- WorkflowMetrics: duration, completed/failed/skipped counts
```

**Relacion con LangGraph**: WorkflowEngine gestiona la ejecucion de tareas a nivel de DAG (dependencias entre tareas). LangGraph gestiona la ejecucion de agentes dentro de un StateGraph (routing, estado compartido). Son complementarios: WorkflowEngine para task orchestration, LangGraph para agent orchestration.

### Componente: BIM-IR Agent (Dominio Especifico)

**Directorio**: `agents/bim_ir/` - Pipeline NLU/NLG autonomo

```
Arquitectura 5-Block (basada en paper BIM-GPT):
1. IntentClassifier: Few-shot classification (temperature=0.0)
   → Clasifica queries BIM en categorias predefinidas
2. ParamExtractor: Extrae parametros con auto-correccion
   → Identifica entidades, valores, relaciones en la query
3. ValueResolver: Resuelve valores contra esquema BIM-IR
   → Mapea parametros extraidos a valores validos del dominio
4. Retriever: Busca informacion via MCP
   → Consulta bases de datos BIM con parametros resueltos
5. Summarizer: Genera respuesta natural
   → Sintetiza resultados en lenguaje comprensible
```

**Estado**: AUTONOMO - No integrado con el orchestrator. Tiene su propio flujo de ejecucion independiente. Podria integrarse como un agente custom dentro del orchestrator via AgentConfig, pero actualmente funciona como pipeline standalone.

### Componente: ToT Graph Planner

**Archivo**: `orchestrator/planning/tot_graph_planner.py`

```
Edge Inference Engine (3 estrategias):
1. Conditional Pattern: Detecta nodos "Router"/"Decision" → edge condicional
   - Genera bifurcaciones con labels "true"/"false" o categorias custom
2. Temporal Sequential: Nodos sin edges explicitos → encadenamiento lineal
   - Genera secuencia basada en orden topologico o posicion
3. Parallel Fan-Out/Fan-In: Detecta patron divergencia/convergencia
   - Genera edges paralelos desde nodo padre a N hijos + merge node

Graph Building Pipeline:
1. parse_graph_spec(llm_output) → nodos + edges raw
2. deduplicate_nodes() → eliminar duplicados por nombre
3. ensure_start_end() → garantizar nodo START y END
4. validate_graph() → verificar conectividad y acyclicidad
5. auto_fix() → reparar edges huerfanos o desconectados
6. fuzzy_agent_matching() → 4 estrategias para mapear nodos a AgentConfigs
```

### Componente: SessionManager

**Archivo**: `orchestrator/session/session_manager.py`

```
SessionContext (dataclass):
- session_id: UUID generado automaticamente
- user_id: identificador del usuario
- turn_count: contador de turnos de conversacion
- is_active: flag de sesion activa
- created_at / updated_at: timestamps
- metadata: dict libre para datos extra

SessionStore (SQLite):
- Persistencia en `.praison/sessions.db`
- Tabla: sessions (session_id, user_id, turn_count, is_active, timestamps, metadata)
- Operaciones: create, update, get_by_id, get_active_sessions, end_session

SessionManager (coordinador):
- create_session(user_id, metadata) → SessionContext
- record_turn() → increment turn_count + update timestamp
- end_session() → set is_active=False
- get_session_info() → dict con datos de sesion actual
- close() → cerrar conexion SQLite
```

**Limitacion**: Session solo trackea conteo de turnos, NO almacena historial de mensajes. El historial conversacional se mantiene exclusivamente en memoria (MemoryManager), no en la sesion.

### Componente: MCPClientManager

**Archivo**: `orchestrator/mcp/client_manager.py`

```
MCPClientManager:
- Lazy initialization con double-checked locking (asyncio.Lock)
- Soporta 2 transportes: STDIO y HTTP/SSE
- Protocol version override para compatibilidad con servers antiguos
- Tool discovery: list_tools() → convierte MCP tools a LangChain BaseTool
- Proper resource cleanup via context managers (__aenter__/__aexit__)
- Thread-safe initialization con locks

MCPServerConfig (de config.py):
- name, command, args, env, transport_type
- Se configura por agente en AgentConfig.mcp_servers
```

**Estado real**: Infraestructura completa pero sin MCP servers configurados en el chat flow por defecto. Los agentes del chat usan herramientas LangChain directas (GraphRAG, DuckDuckGo). MCP se usaria para integraciones externas (ej: servidores APS/Autodesk).

---

## FASE 6: ANALISIS DE CALIDAD - SEGUNDA RONDA

### Hallazgo 11: Web tools nunca inicializados

**Archivo**: `orchestrator/cli/main.py` (lineas de import)
**Problema**: `duckduckgo_tool` y `wikipedia_tool` se definen como `None` al inicio del modulo. El bloque de inicializacion real nunca se ejecuta correctamente. El Researcher agent recibe `tools=[]` efectivamente.
**Impacto ALTO**: El agente Researcher, supuestamente el unico con acceso a web, no puede buscar en internet.
**Recomendacion**: Inicializar las herramientas correctamente o mover la inicializacion al momento de uso.

### Hallazgo 12: Retorno mixto en GraphAgentAdapter

**Archivo**: `orchestrator/cli/graph_adapter.py`
**Problema**: `run_single_agent()` retorna `str` para agentes normales pero puede retornar `dict` para el router. `run_multi_agent_workflow()` siempre retorna via `_extract_result_from_state()` que retorna `str`. El caller (`ChatOrchestrator`) asume siempre `str` pero no valida.
**Impacto medio**: Error silencioso si el tipo cambia. `_extract_text()` en main.py maneja ambos casos, pero es fragil.
**Recomendacion**: Unificar el tipo de retorno a siempre `str`.

### Hallazgo 13: Sin timeout en ejecucion de agentes

**Archivo**: `orchestrator/cli/graph_adapter.py`
**Problema**: Ninguna llamada a `compiled_graph.invoke()` tiene timeout. Un agente LLM que cuelga (timeout de API, prompt infinito) bloqueara el chat indefinidamente.
**Impacto ALTO**: Experiencia de usuario degradada sin recovery posible.
**Recomendacion**: Wrappear invocaciones con `asyncio.wait_for()` o usar el mecanismo de timeout de LangGraph.

### Hallazgo 14: error_handler.py sin tests (0% coverage)

**Archivo**: `orchestrator/core/error_handler.py` (390 lineas)
**Problema**: El ErrorHandler es usado extensivamente por ChatOrchestrator para manejar errores, pero tiene 0% de test coverage. Es un componente critico sin validacion.
**Impacto medio**: Regresiones en error handling pasarian desapercibidas.
**Recomendacion**: Agregar test suite para ErrorHandler cubriendo todos los tipos de error definidos en exceptions.py.

### Hallazgo 15: Prioridad de routing inconsistente

**Archivo**: `orchestrator/factories/route_classifier.py`
**Problema**: `classify_by_keywords()` evalua rutas en orden: quick → research → planning → analysis → standards. Una query como "quickly plan my project" matchearia "quick" antes que "planning". La prioridad deberia ser por complejidad: planning → analysis → research → standards → quick.
**Impacto medio**: Queries con keywords mixtos seran clasificadas suboptimamente.
**Recomendacion**: Invertir el orden de prioridad o usar scoring por peso de keywords.

---

## FASE 7: ACTUALIZACION DE PREGUNTAS Y RESPUESTAS

### Preguntas resueltas de la primera ronda:

| Pregunta                           | Respuesta                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| BIM-IR integrado con orchestrator? | **NO** - Pipeline autonomo de 5 bloques, no integrado                                     |
| graph_adapter.py como funciona?    | Puente con 2 paths: single-agent (router/standard) y multi-agent (ToT → compile → invoke) |
| WorkflowEngine vs LangGraph?       | **Complementarios**: WE=task DAG, LG=agent graph. WE NO es legacy                         |
| Test coverage real?                | ~93% promedio en modulos refactorizados. Gap critico: error_handler 0%                    |
| MCP real usage?                    | Infraestructura completa, sin servers configurados por defecto en chat                    |

### Preguntas pendientes:

1. **mem0/mem0/graphs/ y mem0/mem0/vector_stores/**: Backends de storage del submodulo Mem0 a nivel de implementacion. Ultimo paso pendiente de la exploracion.

2. **Performance real del sistema**: No se ha medido latencia end-to-end de un turno de chat completo. Con 3 documentos almacenados por turno + embedding generation, podria ser lento.

3. **Ingestion pipeline en produccion**: El `sample_loader.py` esta documentado, pero necesito verificar si el `graph_sync` (sincronizacion con Neo4j) funciona correctamente con el hybrid provider.

---

## FASE 8: MAPA COMPLETO DE DECISIONES ARQUITECTONICAS (Actualizado)

### Decision 1: LangGraph como motor de ejecucion

- **Razon inferida**: Grafos de estado compilables con branching, paralelismo, y checkpointing nativo
- **Trade-off**: Mayor complejidad vs PraisonAI (eliminado), pero mas control sobre el flujo
- **Validacion**: Confirmada como eleccion activa y central del sistema

### Decision 2: Memoria hibrida tri-capa (vector + lexical + graph)

- **Razon**: Precision semantica (embeddings) + recall exacto (FTS5/BM25) + relaciones (Neo4j)
- **Trade-off**: Complejidad operativa (3 backends), degradacion graceful si alguno falla
- **Validacion**: HybridProvider probado y funcional. Neo4j es opcional

### Decision 3: Factory Pattern para agentes y tareas

- **Razon**: Templates reutilizables y creacion dinamica
- **Trade-off**: Indirection vs simplicidad. Funciona bien para el caso actual
- **Validacion**: Bien implementado, con templates claros en main.py

### Decision 4: Router dual (keyword + LLM)

- **Razon**: Keywords rapido y predecible; LLM flexible para queries ambiguas
- **Trade-off**: Keywords con cobertura limitada y prioridad inconsistente
- **Validacion**: Funcional pero con issue de prioridad (Hallazgo 15)

### Decision 5: OrchestratorState con Annotated reducers

- **Razon**: Writes paralelos sin race conditions en campos criticos
- **Trade-off**: Complejidad en clasificar single-writer vs parallel-safe
- **Validacion**: Parcialmente correcta - bug en agent_function (Hallazgo 3)

### Decision 6: SRP Refactoring del Orchestrator

- **Razon**: Descomponer Orchestrator monolitico en Executor + Initializer + Lifecycle
- **Trade-off**: 2 versiones coexistentes (orchestrator.py + orchestrator_refactored.py)
- **Validacion**: Refactor COMPLETO e integrado, bien testeado (~93% coverage). Pendiente de reemplazar la version original

### Decision 7: GraphAgentAdapter como abstraccion CLI↔Backend

- **Razon**: Separar la logica de chat (ChatOrchestrator) de la ejecucion de agentes (LangGraph)
- **Trade-off**: Capa adicional de indirection pero permite cambiar backend
- **Validacion**: Funciona pero con tipos de retorno inconsistentes (Hallazgo 12)

### Decision 8: ToT como pipeline de planificacion separado

- **Razon**: Razonamiento avanzado para queries complejas antes de ejecutar agentes
- **Trade-off**: Pipeline sofisticado (edge inference, fuzzy matching) pero desconectado del chat
- **Validacion**: Implementacion robusta en tot_graph_planner, conectada via graph_adapter pero con ruta de chat incompleta (Hallazgo 1)

---

## PLAN DE EXPLORACION RESTANTE

### Paso 10 (pendiente): `mem0/mem0/graphs/` y `mem0/mem0/vector_stores/`

**Razon**: Ultimo modulo sin explorar. Entender la implementacion interna de los backends de storage de Mem0.

---

## RESUMEN EJECUTIVO COMPLETO

Este proyecto es un **framework de orquestacion multi-agente** construido sobre LangChain/LangGraph, con un sistema de memoria tri-capa (vector + lexical + graph) y planificacion avanzada con Tree-of-Thought. Desarrollado como TFG (Trabajo Fin de Grado).

### Arquitectura en 3 Capas

```
┌──────────────────────────────────────────────────────────────────┐
│ CAPA CLI: ChatOrchestrator + DisplayAdapter + EventSystem        │
│   main.py → ChatOrchestrator → GraphAgentAdapter                 │
├──────────────────────────────────────────────────────────────────┤
│ CAPA ORQUESTACION: GraphFactory + RouteClassifier + ToT Planner  │
│   StateGraph ← AgentFactory ← AgentConfig                       │
│   WorkflowEngine (task DAG) | LangGraph (agent graph)            │
├──────────────────────────────────────────────────────────────────┤
│ CAPA DATOS: MemoryManager + SessionManager + MCP                 │
│   HybridProvider(ChromaDB + SQLite FTS5 + Neo4j)                 │
│   Mem0Provider(mem0 + Neo4j) | RAGProvider(in-memory)            │
└──────────────────────────────────────────────────────────────────┘
```

### Flujo Principal Completo (Chat)

```
User Input → Router LLM → classify → route decision
  ├─ "quick"    → Researcher agent → response
  ├─ "analysis" → [Researcher → Analyst → Standards] pipeline → response
  └─ "planning" → (fallback a analysis) → response
Response → Display(Rich) → Store(3 docs en memoria) → Session.record_turn()
```

### Estado del Proyecto

| Componente                     | Estado              | Cobertura Tests           |
| ------------------------------ | ------------------- | ------------------------- |
| Core Orchestrator (original)   | Funcional, activo   | N/A (tests en refactored) |
| Core Orchestrator (refactored) | Completo, integrado | ~93%                      |
| GraphFactory + Routing         | Funcional           | tests en factories/       |
| HybridMemoryProvider           | Funcional           | ~95% (Neo4j failure)      |
| Mem0Provider                   | Funcional           | ~90%                      |
| SessionManager                 | Funcional           | Sin tests dedicados       |
| ChatOrchestrator               | Funcional           | Sin tests E2E             |
| GraphAgentAdapter              | Funcional           | Sin tests                 |
| ToT Planner                    | Implementado        | ~100% (prompt wrappers)   |
| BIM-IR Agent                   | Autonomo            | Sin tests                 |
| MCP Infrastructure             | Completo            | Sin tests                 |
| ErrorHandler                   | En uso              | **0% coverage**           |
| Planning Path (ToT→Chat)       | **NO IMPLEMENTADO** | N/A                       |
| Web Tools (DuckDuckGo)         | **ROTOS**           | N/A                       |

### Clasificacion de Hallazgos por Severidad

**CRITICOS (bloqueantes para produccion)**:

- Hallazgo 3: Bug merge paralelo en agent_outputs → perdida de datos
- Hallazgo 11: Web tools nunca inicializados → Researcher sin busqueda web
- Hallazgo 13: Sin timeout en ejecucion de agentes → bloqueo indefinido

**ALTOS (funcionalidad degradada)**:

- Hallazgo 1: Planning path no implementado → ruta "planning" es falsa
- Hallazgo 5: Bug run_id en create_graph_tool → scope de memoria incorrecto
- Hallazgo 4: Presentacion acoplada a LangChainAgent.execute() → ruido en tests/API

**MEDIOS (calidad y mantenibilidad)**:

- Hallazgo 2: RouterDecision duplicado (TypedDict vs dataclass)
- Hallazgo 7: Refactor SRP a medio migrar (2 versiones coexisten)
- Hallazgo 10: Triple storage por turno → duplicados en retrieval
- Hallazgo 12: Retorno mixto en GraphAgentAdapter
- Hallazgo 14: error_handler.py sin tests (390 lineas, 0% coverage)
- Hallazgo 15: Prioridad de routing inconsistente

**BAJOS (mejoras opcionales)**:

- Hallazgo 6: Textos en espanol hardcodeados (intencional para TFG?)
- Hallazgo 8: asyncio.run() en cleanup puede fallar dentro de event loop
- Hallazgo 9: RAGMemoryProvider misleading (solo substring match)
