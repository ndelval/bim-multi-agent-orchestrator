# Multi-Agent Engineering Reasoning Architecture

## 1. Objectives and Context
- Diseñar un sistema de ingeniería asistido por IA con foco en análisis técnico integral (CAD, normativas, optimización, reporting), no sólo en parsing de planos.
- Explotar patrones de razonamiento avanzados (CoT, ToT, GoT, ReAct, Reflexion) descritos en *Advanced AI Reasoning for Multi-Agent Engineering Systems* manteniendo un balance coste/beneficio.
- Apoyarse en el paquete `orchestrator` (CLI + MemoryManager) y en el runtime de PraisonAI para acelerar la construcción y orquestación de agentes especializados.

## 2. Design Principles
1. **Chain of Thought with Self-Consistency as Baseline** – Low token cost and high reliability for dimensional checks, as emphasized in the whitepaper; every agent starts with CoT before escalating.
2. **ReAct for Tool-Mediated Actions** – Ensures tight coupling with AutoCAD APIs, standards databases, and retrieval layers, matching the paper's recommendation for tool integration.
3. **Tree of Thoughts for Exploratory Geometry** – Invoked when single-path CoT stalls; aligns with demonstrated gains (74% success on complex combinatorial reasoning).
4. **Graph of Thoughts for Synthesis** – Used by the orchestrator and reporting agents to fuse geometric, regulatory, and material facts; mirrors the 62% quality lift cited in the source document.
5. **Reflexion for Quality Assurance** – Feedback-driven self-critique to minimize residual defects; suggested in the whitepaper for catching analysis errors.
6. **Budget-Aware Escalation** – Only escalate from CoT → ToT/GoT when complexity warrants it, preserving the cost reductions praised in the research.
7. **Memory-First Workflows** – Leverage `MemoryManager` (Mem0 + hybrid RAG) for recall, graph reasoning, and document citations before hitting external tools.

## 3. Agent Topology

### 3.1 Orchestration Layer
- **Global Orchestrator (existing class in `orchestrator/core/orchestrator.py`)**
  - *Responsibilities*: session setup, memory priming, task graph selection (sequential vs concurrent), monitoring.
  - *Reasoning Loop*:
    1. Retrieve context via `MemoryManager.build_recall` (Mem0 graph + hybrid RAG).
    2. Draft high-level plan using CoT; evaluate complexity heuristics (drawing size, norms touched).
    3. If ambiguity detected, spawn Tree-of-Thoughts planning branch to explore alternative decompositions.
    4. Dispatch tasks to specialized agents (workflow engine handles DAG ordering/parallelism).
    5. Aggregate outputs with Graph-of-Thoughts style merging; run Reflexion checks before finalization.
  - *Justification*: Mirrors hierarchical orchestration advocated in the document (sequential + concurrent + hierarchical blend).

- **Routing Coordinator Agent** (PraisonAI `Agent`)
  - Decides which specialist(s) engage per prompt; uses CoT + short-list scoring.
  - Justified by the document’s emphasis on adaptive routing, implemented here using PraisonAI’s native inter-agent messaging stack.

### 3.2 Specialized Engineering Agents

The following table resumes the core engineering specialists and their reasoning/tooling profiles:

| Agent | Purpose | Reasoning Strategy | Primary Tools | Justification |
| --- | --- | --- | --- | --- |
| Drawing Intake Agent | Normalize DXF/DWG, extract entities, metadata | ReAct + CoT fallback | AutoCAD API wrapper, file parser, Mem0 document store | ReAct recommended for tool-integrated ingestion; CoT sufficient once data is structured |
| Geometry Analyzer | Measurements, tolerances, spatial relationships | CoT with Self-Consistency → ToT fallback | AutoCAD geometry toolkit, NumPy CAD helpers, GraphRAG for prior cases | CoT-SC handles deterministic math; ToT rescues ambiguous geometries |
| Constraint Solver | Evaluate constraint systems, clash detection | ToT + domain heuristics | Constraint solver microservice, simulation APIs | Tree search mirrors multi-path exploration needs |
| Standards Validator | Map geometry to building codes and standards | ReAct + GoT aggregation | GraphRAG (Mem0), Standards API, local regulation DB | Paper stresses ReAct for compliance checks + GoT for synthesis |
| Materials & BOM Analyst | Extract material specs, compute BOM impacts | CoT + GoT summarization | ERP/BOM API, historical pricing RAG | GoT ideal for combining multiple data sources |
| Risk & Safety Analyst | Identify safety, maintainability issues | CoT + Reflexion loop | Incident database, safety checklists, GraphRAG | Reflexion excels at iterative error catching |
| Simulation Orchestrator | Run structural/thermal sims when required | ToT planning + ReAct calls | Simulation queue API, HPC scheduler | Need exploratory reasoning to choose relevant sims |

Additional agents (design optimisation, spatial intelligence, reporting, human interface) extend the workflow beyond the core analysis loop.

#### Drawing Intake Agent (`pythoncad_parser`)
- **Reasoning Mode**: ReAct-first for tool invocation with a low-temperature CoT fallback once raw data is structured.
- **Why**: The whitepaper references Werk24’s 200× productivity gain using simple CoT plus domain knowledge; wrapping ingestion with ReAct ensures deterministic CAD parsing while providing guardrails around AutoCAD API calls.
- **Tooling**: `dxf_parser`, `layer_extractor`, `entity_recognizer`, `coordinate_transformer`, `metadata_extractor`.

#### Geometry Analysis Agent (`pythongeometric_analyzer`)
- **Reasoning Mode**: Chain of Thought with Self-Consistency (five samples, majority vote) and optional ToT escalation when tolerance conflicts remain.
- **Why**: CoT-SC lowers dimensional error rates (14 % → 3 %) with ~300 tokens; ToT only triggers for ambiguous geometries to avoid unnecessary cost.
- **Tooling**: `dimension_calculator`, `area_volume_compute`, `angle_analyzer`, `tolerance_checker`, `scale_validator`, `intersection_finder`.

#### Spatial Intelligence Agent (`pythonspatial_intelligence`)
- **Reasoning Mode**: Tree of Thoughts (beam width = 3, max depth = 4) focused on 3D layout/visibility optimisation.
- **Why**: Benchmarks like SpAItial/AlphaGeometry show ToT jumps success to 74 % vs 4 % for CoT in multi-solution spatial puzzles.
- **Tooling**: `spatial_relationship_detector`, `collision_detector`, `path_optimizer`, `visibility_analyzer`, `constraint_solver`, `pattern_recognizer`.

#### Constraint Solver Agent (`pythonconstraint_solver`)
- **Reasoning Mode**: Guided Tree of Thoughts combined with domain heuristics (e.g., clash severity, constraint priority).
- **Why**: Constraint satisfaction often needs backtracking; ToT enables branching through alternative resolution orders while heuristics keep the search tractable.
- **Tooling**: `constraint_solver_service`, `simulation_api_client`, `tolerance_conflict_resolver`, `load_case_generator`.

#### Standards Validation Agent (`pythonstandards_validator`)
- **Reasoning Mode**: ReAct (max 3 iterations, tools-first policy) with GoT aggregation of evidence before verdicts.
- **Why**: The paper reports hallucination drop (14 % → 6 %) and 57 % faster submittal validation when ReAct orchestrates external knowledge; GoT ensures citations are synthesised coherently.
- **Tooling**: `building_code_api`, `iso_standard_checker`, `local_regulation_db`, `material_specs_validator`, `accessibility_checker`, `fire_safety_validator`.

#### Materials & BOM Analyst (`pythonmaterials_bom`)
- **Reasoning Mode**: CoT baseline with GoT summarisation across cost, availability, and material properties.
- **Why**: GoT’s 62 % quality gain vs ToT helps reconcile multi-source material data, while CoT keeps token budgets low for straightforward lookups.
- **Tooling**: `erp_connector`, `pricing_history_rag`, `material_property_db`, `supply_risk_evaluator`, `sustainability_checker`.

#### Risk & Safety Analyst (`pythonrisk_safety`)
- **Reasoning Mode**: CoT diagnostics augmented by Reflexion feedback loops (memory buffer = 10) to iteratively refine findings.
- **Why**: Reflexion achieves 91 % error detection success and 18 % rework reduction in construction case studies; perfect for safety/maintainability sweeps.
- **Tooling**: `inconsistency_detector`, `duplicate_finder`, `gap_analyzer`, `symmetry_checker`, `statistical_outlier_detector`, `historical_comparison`.

#### Simulation Orchestrator (`pythonsimulation_orchestrator`)
- **Reasoning Mode**: ToT planning to choose simulation portfolios, followed by ReAct tool calls to the HPC/simulation queue.
- **Why**: Selecting the minimal simulation set is an exploratory planning problem; ToT balances cost/coverage before dispatching real workloads.
- **Tooling**: `simulation_queue_api`, `solver_parameterizer`, `results_postprocessor`, `scenario_catalog`, `budget_monitor`.

#### Design Optimization Agent (`pythondesign_optimizer`)
- **Reasoning Mode**: Graph of Thoughts (weighted consensus, two refinement passes) across cost, structural, and efficiency objectives.
- **Why**: GoT excels at multi-constraint synthesis (62 % quality lift, 31 % lower cost vs ToT) making it ideal for optimisation loops.
- **Tooling**: `cost_estimator`, `material_optimizer`, `structural_analyzer`, `energy_efficiency_calc`, `space_utilization_optimizer`, `multi_objective_solver`.

#### Human Interface Agent (`pythonhuman_interface`)
- **Reasoning Mode**: Chain of Thought (temperature = 0.3, explanation mode on) for predictable, auditable dialogue.
- **Why**: Conversational UX prioritises transparency; CoT-based explanations (cf. Microsoft AutoGen) foster operator trust.
- **Tooling**: `natural_language_parser`, `visualization_generator`, `explanation_builder`, `feedback_collector`, `report_formatter`, `alert_system`.

#### Report Synthesis Agent (`pythonreport_synthesizer`)
- **Reasoning Mode**: Graph of Thoughts (hierarchical synthesis, engineering report template output).
- **Why**: The paper notes GoT integrates geometric relationships, constraints, and materials into reports that are 45 % more coherent.
- **Tooling**: `data_aggregator`, `chart_generator`, `table_builder`, `executive_summary_writer`, `technical_writer`, `export_formatter`.

### 3.3 Support & Collaboration Services
- **Memory Services**: `MemoryManager` with Mem0 (graph relationships, conversational history) + Hybrid RAG for document retrieval.
- **Tool Adapters**: AutoCAD CLI/API, simulation orchestrators, regulatory DB connectors, GraphRAG lookup (already implemented in `orchestrator/tools/graph_rag_tool.py`).
- **Communication Fabric**: adopt MCP for tool calls and rely on PraisonAI's native collaboration channels for agent-to-agent coordination.

## 4. Tooling & Integration Matrix
| Tool | Capability | Primary Agents | Notes |
| --- | --- | --- | --- |
| `dxf_parser`, `layer_extractor`, `entity_recognizer` | DXF/DWG ingestion, layer segmentation, entity detection | `pythoncad_parser` | Deterministic parsing; wrap AutoCAD APIs with idempotent sandboxing |
| `dimension_calculator`, `area_volume_compute`, `tolerance_checker` | Precise geometric computations | `pythongeometric_analyzer` | CoT-SC loop votes across multiple samples before committing |
| `spatial_relationship_detector`, `collision_detector`, `path_optimizer` | 3D spatial reasoning and optimization | `pythonspatial_intelligence` | Expose via async tools to support ToT branch evaluation |
| `constraint_solver_service`, `simulation_api_client`, `tolerance_conflict_resolver` | Clash resolution & constraint solving | `pythonconstraint_solver` | Provide deterministic fallbacks for safety-critical constraints |
| `building_code_api`, `iso_standard_checker`, `local_regulation_db` | Regulatory fact retrieval | `pythonstandards_validator` | All tools instrumented for ReAct logging + citation metadata |
| `erp_connector`, `pricing_history_rag`, `material_property_db` | Material & BOM intelligence | `pythonmaterials_bom` | Cache frequent queries to control latency and cost |
| `inconsistency_detector`, `statistical_outlier_detector` | QA anomaly discovery | `pythonrisk_safety` | Feed findings back into Reflexion memory buffer |
| `simulation_queue_api`, `solver_parameterizer`, `results_postprocessor` | Simulation orchestration | `pythonsimulation_orchestrator` | Supports async execution and budgeting heuristics |
| `cost_estimator`, `multi_objective_solver`, `material_optimizer` | Multi-constraint optimisation | `pythondesign_optimizer` | Requires cache of recent solver runs for budget awareness |
| `natural_language_parser`, `visualization_generator`, `explanation_builder` | Human-facing dialogue, visualisation | `pythonhuman_interface` | Must support localization and safety-checked outputs |
| `data_aggregator`, `chart_generator`, `executive_summary_writer` | Report synthesis and formatting | `pythonreport_synthesizer` | Integrate with existing Markdown/HTML templates and export pipeline |
| `graph_rag_lookup` | Standards/document retrieval with citations | Standards + Report agents, QA | Already exposed via `create_graph_tool`; attach provenance metadata |
| Explainability Logger | Persist reasoning traces and tool calls | All agents | Store in Mem0 with run/user metadata for future recall |

## 5. Memory and Knowledge Strategy
1. **Persistent Graph Memory (Mem0)** – stores user/session context, resolved facts, cross-entity relationships for future runs.
2. **Hybrid Retrieval (Chroma + BM25 + reranker)** – hosts standards, manuals, prior reports; invoked before external APIs to save cost.
3. **Short-Term Conversation Capture** – user prompts, agent responses stored using `default_conversation_metadata` to seed future recalls.
4. **Fact Normalization Pipeline** – specialists convert tool outputs into normalized facts before storage, enabling GoT synthesis downstream.

## 6. Orchestrator Reasoning Workflow
1. **Ingestion**: Accept CAD prompt/input, validate file locations, load env.
2. **Context Recall**: Query Mem0 for prior project knowledge; fetch standards from hybrid store based on tags.
3. **Problem Scoping**: CoT-based attempt to classify work; estimate complexity and cost envelope.
4. **Plan Formation**: If straightforward, linear plan; otherwise invoke ToT to branch alternative decompositions (e.g., structural vs. MEP), reserving GoT for later synthesis where graph aggregation outperforms tree search.
5. **Task Graph Assembly**: Map plan onto PraisonAI tasks (workflow engine). Choose sequential vs concurrent execution according to dependency graph.
6. **Execution Monitoring**: Capture task events, re-route on errors. For high-risk outputs, trigger Reflexion checks automatically.
7. **Synthesis**: Use GoT-inspired aggregation to merge specialist outputs; ensure citations attached using GraphRAG metadata.
8. **Report Generation**: Feed aggregated graph to Report Generator; embed decisions, metrics, and QA outcomes.
9. **Post-Run Memory Update**: Store validated findings, resolved conflicts, and performance metrics back into Mem0 + hybrid store.

## 7. Agent Reasoning Templates
- **Standard Agent Loop (CoT First)**
  1. Restate task with constraints (CoT prompt).
  2. Consult memory/tools (ReAct) if data missing.
  3. Draft answer; self-evaluate using checklist (Reflexion-lite).
  4. If confidence < threshold → escalate to ToT/GoT routine.
  5. Return structured output + metadata (confidence, sources).

- **Exploratory Agent Loop (ToT Enabled)**
  1. Enumerate candidate strategies (branching factor configurable).
  2. Score partial solutions using domain heuristics (compliance, tolerances).
  3. Prune failing branches; continue until threshold met or fallback triggered.
  4. Summarize best branch; log discarded alternatives for audit.

- **GoT Synthesis Loop (Reporting/QA)**
  1. Build graph nodes from specialist outputs (facts, constraints, risks).
  2. Apply Aggregation → Refinement → Generation passes to ensure consistency.
  3. Produce narrative + structured data tables with citations.

## 8. End-to-End Process Flow
1. **Ingreso y contextualización**: El orquestador recibe la necesidad de ingeniería (planos, requisitos, KPIs) y recupera conocimiento previo desde Mem0 e índices híbridos.
2. **Planificación razonada**: El orquestador emplea CoT/ToT para dividir el problema en tareas y dependencias (grafo), asignando agentes según dominio.
3. **Preparación de agentes**: Se activan los agentes registrados (actualmente mocks) con la información mínima necesaria y se validan entradas/salidas esperadas.
4. **Ejecución coordinada**: El workflow engine dispara tareas en paralelo cuando el grafo lo permite (ej. análisis geométrico, normativo y materiales) y gestiona reintentos.
5. **Ciclo de aseguramiento**: `pythonrisk_safety` revisa resultados intermedios con Reflexion; cualquier hallazgo vuelve a poner en cola la tarea afectada o escala al orquestador.
6. **Síntesis y comunicaciones**: `pythonreport_synthesizer` aplica GoT para generar entregables coherentes mientras `pythonhuman_interface` gestiona interacción con el equipo humano.
7. **Cierre y aprendizaje**: Se consolidan métricas, decisiones y artefactos en Mem0/RAG para nutrir ejecuciones futuras.

## 9. Implementation Roadmap
1. **Paso 1 – Definir agentes mock**
   - Crear `AgentConfig` para cada especialista con descripciones y herramientas declaradas, devolviendo respuestas dummy controladas.
   - Registrar los agentes en PraisonAI (sin tareas predefinidas) y validar que el CLI del orquestador puede instanciarlos sin ejecutar lógica real.
   - Comando de comprobación rápida: `python -m orchestrator.cli mock-info` imprime el `system_info` del orquestador mock.

2. **Paso 2 – Enriquecer el orquestador con planificación ToT/GoT**
   - Añadir rutinas CoT/ToT que generen un plan detallado (grafo de tareas + agentes asignados) antes de ejecutar.
   - Incorporar una fase GoT ligera para verificar consistencia del plan y documentar dependencias.
   - Exponer el plan en la salida (stdout/log) para inspección humana.

3. **Paso 3 – Validar ejecución secuencial y paralela**
   - Configurar el `WorkflowEngine` para desembocar en ejecución paralela cuando las dependencias lo permitan.
   - Añadir pruebas con mocks que confirmen orden correcto, reintentos y finalización bajo distintos escenarios.

4. **Paso 4 – Instrumentación y memoria**
   - Integrar Mem0/híbrido sólo para logging de contexto (inputs, planes, outputs mock) asegurando que los agentes puedan recuperar esa memoria en futuras iteraciones.
   - Añadir medición básica de tokens/latencias para alimentar decisiones posteriores.

5. **Paso 5 – Sustituir mocks por capacidades reales**
   - Iterar agente por agente (CAD, geometría, normativas, etc.), reemplazando la lógica dummy por implementaciones concretas y ampliando herramientas según sea necesario.
   - Re-evaluar el plan generado y los patrones de paralelización tras cada incorporación real.

6. **Paso 6 – Pruebas integrales y despliegue interno**
   - Construir escenarios representativos (proyectos distintos, normativas variadas) y validar extremo a extremo.
   - Documentar procedimientos de operación, monitorización y escalado para producción.

## 10. Justification Summary
- **CoT-first strategy** keeps compute budgets in check while covering majority cases, aligning with the paper’s recommendation (200–500 tokens typical).
- **ToT/GoT escalation** only when ambiguity or synthesis demands justify extra cost, leveraging the 62–74% quality gains discussed.
- **ReAct** ensures agents interact safely with external systems (AutoCAD, regulatory APIs), reducing hallucinations (6% vs 14% per source data).
- **Reflexion QA** closes the loop on accuracy, aligning with the document’s insistence on quality assurance for engineering deliverables.
- **Hierarchical orchestration** mirrors the recommended phased architecture (sequential → concurrent → magentic/hierarchical), maximizing reuse of the existing orchestrator + PraisonAI frameworks without rewriting core infrastructure.
