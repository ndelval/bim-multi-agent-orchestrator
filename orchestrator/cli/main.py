"""Command-line utilities for the engineering orchestrator."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import asdict
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Sequence

from orchestrator.core.config import (
    AgentConfig,
    EmbedderConfig,
    ExecutionConfig,
    MemoryConfig,
    MemoryProvider,
    OrchestratorConfig,
    ProcessType,
    TaskConfig,
)
from orchestrator.core.exceptions import WorkflowError
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.memory.document_schema import default_conversation_metadata
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.templates.mock_engineering import create_mock_engineering_config


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()  # Load .env so OPENAI_API_KEY and others are available

logger = logging.getLogger(__name__)

ROUTE_OPTIONS = ["quick", "research", "analysis", "planner", "standards"]
DEFAULT_SESSION_TITLE = "CLI Engineering Session"


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose >= 2:
        level = logging.INFO
    if verbose >= 4:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)7s %(name)s:%(lineno)d %(message)s")


def _infer_embedding_dims(model: str) -> int:
    table = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
        "text-embedding-002": 1536,
        "all-mpnet-base-v2": 768,
        "sentence-transformers": 768,
    }
    model_l = (model or "").lower()
    for key, value in table.items():
        if key in model_l:
            return value
    return 1536


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Environment variable %s=%r is not a valid integer", name, raw)
        return default


def build_hybrid_memory_from_env() -> MemoryConfig:
    embedder_provider = os.getenv("HYBRID_EMBEDDER_PROVIDER", "openai")
    embedder_model = os.getenv("HYBRID_EMBEDDER_MODEL", "text-embedding-3-small")
    embedder_dims = _env_int("HYBRID_EMBEDDER_DIMS") or _infer_embedding_dims(embedder_model)
    embedder_conf = {"model": embedder_model, "embedding_dims": embedder_dims}
    embedder = EmbedderConfig(provider=embedder_provider, config=embedder_conf)

    vector_path = os.getenv("HYBRID_VECTOR_PATH", ".praison/hybrid_chroma")
    vector_collection = os.getenv("HYBRID_VECTOR_COLLECTION", "hybrid_memory")
    vector_store = {
        "provider": "chromadb",
        "config": {
            "path": vector_path,
            "collection": vector_collection,
        },
    }

    lexical: Dict[str, Any] = {"db_path": os.getenv("HYBRID_LEXICAL_DB_PATH", ".praison/hybrid_lexical.db")}
    ttl_seconds = _env_int("HYBRID_LEXICAL_TTL_SECONDS")
    if ttl_seconds is not None:
        lexical["ttl_seconds"] = ttl_seconds
    max_entries = _env_int("HYBRID_LEXICAL_MAX_ENTRIES")
    if max_entries is not None:
        lexical["max_entries"] = max_entries
    cleanup_every = _env_int("HYBRID_LEXICAL_CLEANUP_INTERVAL")
    if cleanup_every is not None:
        lexical["cleanup_interval"] = cleanup_every

    rerank_enabled = _env_bool("HYBRID_RERANK_ENABLED", True)
    rerank_top_k = _env_int("HYBRID_RERANK_TOP_K", 10) or 10
    rerank_workers = _env_int("HYBRID_RERANK_MAX_WORKERS", 1) or 1
    rerank = {
        "enabled": rerank_enabled,
        "model": os.getenv("HYBRID_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "top_k": rerank_top_k,
        "max_workers": rerank_workers,
    }

    graph_uri = os.getenv("HYBRID_GRAPH_URI") or os.getenv("NEO4J_URL")
    graph_user = os.getenv("HYBRID_GRAPH_USER") or os.getenv("NEO4J_USER")
    graph_password = os.getenv("HYBRID_GRAPH_PASSWORD") or os.getenv("NEO4J_PASSWORD")
    graph_flag = _env_bool("HYBRID_GRAPH_ENABLED")
    graph_should_enable = graph_flag if graph_flag is not None else bool(graph_uri and graph_user and graph_password)
    graph = {
        "enabled": graph_should_enable,
    }
    if graph_should_enable:
        graph.update({
            "uri": graph_uri,
            "user": graph_user,
            "password": graph_password,
        })

    config = {
        "vector_store": vector_store,
        "lexical": lexical,
        "rerank": rerank,
        "graph": graph,
    }

    return MemoryConfig(
        provider=MemoryProvider.HYBRID,
        use_embedding=True,
        config=config,
        embedder=embedder,
    )


def build_mem0_memory_from_env() -> MemoryConfig:
    graph_provider = os.getenv("MEM0_GRAPH_PROVIDER", "neo4j").lower()
    graph_url = os.getenv("MEM0_GRAPH_URL") or os.getenv("NEO4J_URL", "bolt://localhost:7687")
    graph_user = os.getenv("MEM0_GRAPH_USER") or os.getenv("NEO4J_USER", "neo4j")
    graph_password = os.getenv("MEM0_GRAPH_PASSWORD") or os.getenv("NEO4J_PASSWORD", "password")

    graph_store = {
        "provider": graph_provider,
        "config": {
            "url": graph_url,
            "username": graph_user,
            "password": graph_password,
        },
    }

    vector_store: Optional[Dict[str, Any]] = None
    vector_provider = os.getenv("MEM0_VECTOR_PROVIDER", "").lower()
    if vector_provider == "qdrant":
        q_config: Dict[str, Any] = {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": _env_int("QDRANT_PORT", 6333) or 6333,
            "collection_name": os.getenv("QDRANT_COLLECTION", "mem0_memory"),
        }
        if os.getenv("QDRANT_API_KEY"):
            q_config["api_key"] = os.getenv("QDRANT_API_KEY")
        if os.getenv("QDRANT_URL"):
            q_config["url"] = os.getenv("QDRANT_URL")
        if _env_bool("QDRANT_USE_HTTPS"):
            q_config["https"] = True
        vector_store = {"provider": "qdrant", "config": q_config}

    embedder_provider = os.getenv("MEM0_EMBEDDER_PROVIDER", "openai")
    embedder_model = os.getenv("MEM0_EMBEDDER_MODEL", "text-embedding-3-large")
    embedder_conf = {
        "model": embedder_model,
        "embedding_dims": _infer_embedding_dims(embedder_model),
    }
    embedder = EmbedderConfig(provider=embedder_provider, config=embedder_conf)

    llm_provider = os.getenv("MEM0_LLM_PROVIDER", "openai")
    llm_model = os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini")
    llm = {
        "provider": llm_provider,
        "config": {"model": llm_model},
    }

    config: Dict[str, Any] = {
        "graph_store": graph_store,
        "llm": llm,
        "embedder": {"provider": embedder.provider, "config": embedder.config},
    }
    if vector_store:
        config["vector_store"] = vector_store

    return MemoryConfig(
        provider=MemoryProvider.MEM0,
        use_embedding=True,
        config=config,
        embedder=embedder,
    )


def build_rag_memory_from_env() -> MemoryConfig:
    embedder_provider = os.getenv("RAG_EMBEDDER_PROVIDER", "openai")
    embedder_model = os.getenv("RAG_EMBEDDER_MODEL", "text-embedding-3-small")
    embedder = EmbedderConfig(
        provider=embedder_provider,
        config={"model": embedder_model, "embedding_dims": _infer_embedding_dims(embedder_model)},
    )
    short_db = os.getenv("RAG_SHORT_DB", ".praison/memory/short.db")
    long_db = os.getenv("RAG_LONG_DB", ".praison/memory/long.db")
    rag_db = os.getenv("RAG_VECTOR_PATH", ".praison/memory/chroma_db")
    return MemoryConfig(
        provider=MemoryProvider.RAG,
        use_embedding=True,
        config={},
        short_db=short_db,
        long_db=long_db,
        rag_db_path=rag_db,
        embedder=embedder,
    )


def resolve_memory_config(provider_name: Optional[str]) -> MemoryConfig:
    provider = (provider_name or os.getenv("ORCH_MEMORY_PROVIDER", "hybrid")).lower()
    if provider == MemoryProvider.HYBRID.value:
        return build_hybrid_memory_from_env()
    if provider == MemoryProvider.MEM0.value:
        return build_mem0_memory_from_env()
    if provider == MemoryProvider.RAG.value:
        return build_rag_memory_from_env()
    raise ValueError(f"Unsupported memory provider: {provider}")


def _format_prompt_block(user_prompt: str, recall_items: Sequence[str]) -> str:
    lines = [f"USER PROMPT: {user_prompt.strip()}"]
    if recall_items:
        lines.append("MEMORY RECALL CONTEXT:")
        lines.extend(f"  • {item}" for item in recall_items)
    return "\n".join(lines)


def _conversation_metadata(
    user_id: str,
    run_id: str,
    agent_id: str,
    turn_index: int,
    language: str,
    document_id: str,
    tags: Iterable[str],
    title: str,
) -> Dict[str, Any]:
    metadata = default_conversation_metadata(user_id, run_id, agent_id, content_hash=f"{agent_id}-{turn_index:04d}")
    metadata.update(
        {
            "document_id": document_id,
            "title": title,
            "section": agent_id,
            "order": turn_index,
            "language": language,
            "tags": list(tags),
        }
    )
    return metadata


def _build_recall_snippets(
    manager: MemoryManager,
    query: str,
    *,
    user_id: str,
    run_id: str,
    limit: int,
) -> List[str]:
    try:
        results = manager.retrieve_filtered(
            query,
            limit=limit,
            user_id=user_id,
            run_id=run_id,
        )
    except Exception:
        results = manager.retrieve(query, limit=limit)
    snippets: List[str] = []
    seen: set[str] = set()
    for item in results or []:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        key = content.lower()
        if key in seen:
            continue
        seen.add(key)
        meta = item.get("metadata") or {}
        src = meta.get("document_id") or meta.get("chunk_id") or item.get("id")
        if src:
            snippets.append(f"{content} [src: {src}]")
        else:
            snippets.append(content)
    return snippets


def _build_chat_agents() -> List[AgentConfig]:
    router = AgentConfig(
        name="Orchestrator",
        role="Router & Coordinator",
        goal="Seleccionar la ruta adecuada para cada petición de ingeniería",
        backstory=(
            "Ingeniero senior que decide si una consulta requiere respuesta rápida, investigación documentada, "
            "análisis profundo, planificación táctico-operativa o verificación normativa."
        ),
        instructions=dedent(
            """
            Evalúa el prompt, revisa el contexto proporcionado y decide qué ruta activa:
              - quick: saludos o respuestas triviales que sólo necesitan un recordatorio.
              - research: búsqueda documental con evidencia citada usando graph_rag_lookup.
              - analysis: razonamiento paso a paso combinando memoria + GraphRAG.
              - planner: elaborar un plan de acción con fases y dependencias.
              - standards: dudas sobre normas, guías o cumplimiento regulatorio.
            Devuelve JSON con las claves "response" y "decision" (quick|research|analysis|planner|standards).
            """
        ),
    )

    quick = AgentConfig(
        name="QuickResponder",
        role="Short Response Agent",
        goal="Dar respuestas breves usando sólo el contexto necesario",
        backstory="Especialista en respuestas ágiles para interacciones cotidianas.",
        instructions=dedent(
            """
            Responde en una o dos frases, citando memoria previa cuando aplique.
            Si falta información, pide aclaraciones de forma cordial.
            """
        ),
    )

    researcher = AgentConfig(
        name="Researcher",
        role="Research Specialist",
        goal="Recuperar evidencia documental actualizada",
        backstory="Analista documental con enfoque en ingeniería multidisciplinar.",
        instructions=dedent(
            """
            1. Llama primero a graph_rag_lookup(query=<pregunta>, top_k=5) y revisa los fragmentos.
            2. Resume hallazgos con viñetas, incluyendo citas [document_id §sección].
            3. Si faltan fuentes, explícitalo y sugiere cómo ampliarlas.
            """
        ),
    )

    analyst = AgentConfig(
        name="Analyst",
        role="Analyst Agent",
        goal="Producir análisis técnico argumentado",
        backstory="Ingeniero de sistemas capaz de sintetizar hallazgos y riesgo.",
        instructions=dedent(
            """
            Integra memoria + GraphRAG para elaborar explicación estructurada (contexto, evidencia, riesgos, recomendaciones).
            Cita cada afirmación relevante con [document_id §sección] o marca claramente si es inferencia.
            """
        ),
    )

    planner = AgentConfig(
        name="Planner",
        role="Planning Agent",
        goal="Convertir la petición en un plan accionable",
        backstory="PM técnico acostumbrado a proyectos de ingeniería complejos.",
        instructions=dedent(
            """
            Construye un plan faseado. Cada fase debe tener objetivo, entregables, responsables y referencias a evidencia.
            Usa graph_rag_lookup cuando necesites respaldar decisiones o dependencias.
            """
        ),
    )

    standards = AgentConfig(
        name="StandardsAgent",
        role="Compliance & Standards Advisor",
        goal="Responder sobre normativas y mejores prácticas",
        backstory="Consultor especializado en normativas industriales y de seguridad.",
        instructions=dedent(
            """
            Antes de responder, ejecuta graph_rag_lookup con tags adecuados (p.ej. "norma", "seguridad").
            Resume el requisito, cita documentos y añade advertencias o comprobaciones necesarias.
            """
        ),
    )

    return [router, quick, researcher, analyst, planner, standards]


_AGENT_TEMPLATES = {cfg.name: cfg for cfg in _build_chat_agents()}


def _get_agent_template(name: str) -> AgentConfig:
    try:
        template = _AGENT_TEMPLATES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown agent template: {name}") from exc
    return copy.deepcopy(template)


ROUTE_TASK_SPECS = {
    "research": {
        "task_name": "research_task",
        "agent": "Researcher",
        "description": dedent(
            """
            Genera una nota de investigación sustentada en fuentes técnicas relevantes.
            {prompt_block}
            Usa graph_rag_lookup antes de concluir y entrega viñetas con citas [document_id §sección].
            """
        ),
        "expected": "Bullet list with evidence-backed findings and citations.",
    },
    "analysis": {
        "task_name": "analysis_task",
        "agent": "Analyst",
        "description": dedent(
            """
            Produce un análisis estructurado (contexto, evaluación, riesgos, recomendaciones) apoyándote en memoria y GraphRAG.
            {prompt_block}
            Señala las fuentes con citas [document_id §sección] o marca claramente cualquier hipótesis.
            """
        ),
        "expected": "Structured analysis covering context, evidence, risks, recommendations.",
    },
    "planner": {
        "task_name": "plan_task",
        "agent": "Planner",
        "description": dedent(
            """
            Diseña un plan accionable con fases, responsables, entregables y criterios de aceptación.
            {prompt_block}
            Usa graph_rag_lookup para respaldar dependencias o normas clave.
            """
        ),
        "expected": "Plan with phases, owners, deliverables and acceptance criteria.",
    },
    "standards": {
        "task_name": "standards_task",
        "agent": "StandardsAgent",
        "description": dedent(
            """
            Responde la consulta normativa citando estándares, guías o regulaciones aplicables.
            {prompt_block}
            Añade recomendaciones prácticas y advertencias de cumplimiento.
            """
        ),
        "expected": "Compliance-focused answer with supporting citations.",
    },
}


def _build_router_config(
    *,
    user_prompt: str,
    recall_items: Sequence[str],
    base_memory_config: MemoryConfig,
    user_id: str,
    verbose: int,
    max_iter: int,
) -> OrchestratorConfig:
    memory_config = copy.deepcopy(base_memory_config)
    agent = _get_agent_template("Orchestrator")
    prompt_block = _format_prompt_block(user_prompt, recall_items)
    description = dedent(
        """
        Evalúa el prompt del usuario y decide qué ruta debe manejarlo (quick | research | analysis | planner | standards).
        {prompt_block}
        Devuelve JSON con las claves "response" (respuesta breve inicial) y "decision".
        """
    ).format(prompt_block=prompt_block).strip()

    router_task = TaskConfig(
        name="route_selector",
        description=description,
        expected_output="Decision: one of quick|research|analysis|planner|standards, plus a one-line justification.",
        agent_name="Orchestrator",
        async_execution=False,
        is_start=True,
        task_type="decision",
        condition={route: [] for route in ROUTE_OPTIONS},
    )

    exec_config = ExecutionConfig(
        process=ProcessType.WORKFLOW,
        verbose=2 if verbose >= 2 else 1,
        max_iter=max_iter,
        memory=True,
        user_id=user_id,
        async_execution=False,
    )

    return OrchestratorConfig(
        name="CliRouter",
        memory_config=memory_config,
        execution_config=exec_config,
        agents=[agent],
        tasks=[router_task],
        embedder=memory_config.embedder,
        custom_config={},
    )


def _build_route_config(
    *,
    route: str,
    user_prompt: str,
    recall_items: Sequence[str],
    base_memory_config: MemoryConfig,
    user_id: str,
    verbose: int,
    max_iter: int,
) -> Optional[OrchestratorConfig]:
    spec = ROUTE_TASK_SPECS.get(route)
    if not spec:
        return None

    memory_config = copy.deepcopy(base_memory_config)
    agent = _get_agent_template(spec["agent"])
    prompt_block = _format_prompt_block(user_prompt, recall_items)
    description = spec["description"].format(prompt_block=prompt_block).strip()

    task = TaskConfig(
        name=spec["task_name"],
        description=description,
        expected_output=spec["expected"],
        agent_name=spec["agent"],
        async_execution=False,
        is_start=True,
    )

    exec_config = ExecutionConfig(
        process=ProcessType.WORKFLOW,
        verbose=2 if verbose >= 2 else 1,
        max_iter=max_iter,
        memory=True,
        user_id=user_id,
        async_execution=False,
    )

    return OrchestratorConfig(
        name=f"CliRoute::{route}",
        memory_config=memory_config,
        execution_config=exec_config,
        agents=[agent],
        tasks=[task],
        embedder=memory_config.embedder,
        custom_config={},
    )


def _extract_text(output: Any) -> Optional[str]:
    if output is None:
        return None
    if isinstance(output, str):
        return output.strip() or None
    if isinstance(output, dict):
        for key in ("response", "content", "summary", "text"):
            value = output.get(key)
            if value:
                return str(value).strip()
        return json.dumps(output, ensure_ascii=False)
    raw = getattr(output, "raw", None)
    if raw:
        return str(raw).strip()
    json_dict = getattr(output, "json_dict", None)
    if isinstance(json_dict, dict):
        for key in ("response", "answer", "output"):
            value = json_dict.get(key)
            if value:
                return str(value).strip()
        return json.dumps(json_dict, ensure_ascii=False)
    pydantic_obj = getattr(output, "pydantic", None)
    if pydantic_obj is not None:
        if hasattr(pydantic_obj, "response"):
            value = getattr(pydantic_obj, "response")
            if value:
                return str(value).strip()
        try:
            return pydantic_obj.model_dump_json()
        except Exception:
            return str(pydantic_obj)
    if isinstance(output, Sequence):
        for item in reversed(list(output)):
            text = _extract_text(item)
            if text:
                return text
    return None


def _extract_decision(output: Any) -> Optional[str]:
    if output is None:
        return None
    if isinstance(output, dict):
        decision = output.get("decision")
        return str(decision).strip() if decision else None
    json_dict = getattr(output, "json_dict", None)
    if isinstance(json_dict, dict) and json_dict.get("decision"):
        return str(json_dict["decision"]).strip()
    pydantic_obj = getattr(output, "pydantic", None)
    if pydantic_obj is not None and hasattr(pydantic_obj, "decision"):
        value = getattr(pydantic_obj, "decision")
        return str(value).strip() if value else None
    raw = getattr(output, "raw", None)
    if isinstance(raw, str) and "decision" in raw.lower():
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("decision"):
                return str(data["decision"]).strip()
        except Exception:
            pass
    return None


def _attach_graph_tool(orchestrator: Orchestrator, *, user_id: str, run_id: str) -> None:
    if not orchestrator.memory_manager:
        return
    tool = orchestrator.create_graph_tool(user_id=user_id, run_id=run_id)
    for agent_name in ("Researcher", "Analyst", "Planner", "StandardsAgent"):
        agent = orchestrator.get_agent(agent_name)
        if agent is None:
            continue
        if all(getattr(t, "__name__", None) != getattr(tool, "__name__", None) for t in getattr(agent, "tools", [])):
            agent.tools.append(tool)


def run_chat(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    try:
        base_memory_config = resolve_memory_config(args.memory_provider)
    except Exception as exc:  # pragma: no cover - configuration errors
        print(f"❌ Error construyendo configuración de memoria: {exc}")
        return 1

    memory_manager = MemoryManager(copy.deepcopy(base_memory_config))
    provider_label = base_memory_config.provider.value
    print(f"✅ Memory manager initialized for user: {args.user} (provider={provider_label})")
    document_id = args.session_document or f"cli-session::{args.run}"
    turn_index = 0

    while True:
        try:
            prompt = input("» ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        user_meta = _conversation_metadata(
            args.user,
            args.run,
            agent_id="user",
            turn_index=turn_index,
            language=args.language,
            document_id=document_id,
            tags=["chat", "user"],
            title=DEFAULT_SESSION_TITLE,
        )
        memory_manager.store(prompt, user_meta)
        turn_index += 1

        recall_items = _build_recall_snippets(
            memory_manager,
            prompt,
            user_id=args.user,
            run_id=args.run,
            limit=args.recall_top_k,
        )

        router_config = _build_router_config(
            user_prompt=prompt,
            recall_items=recall_items,
            base_memory_config=base_memory_config,
            user_id=args.user,
            verbose=args.verbose,
            max_iter=args.max_iter,
        )

        router = Orchestrator(router_config)
        try:
            router_result = router.run_sync()
        except WorkflowError as exc:
            logger.warning("Router workflow failed: %s", exc)
            print("⚠️  Modelo no disponible temporalmente; respondiendo de forma básica.")
            final_text = "Estoy teniendo problemas temporales con el LLM. ¿Puedes intentarlo de nuevo en unos segundos?"
            assistant_meta = _conversation_metadata(
                args.user,
                args.run,
                agent_id="assistant",
                turn_index=turn_index,
                language=args.language,
                document_id=document_id,
                tags=["chat", "assistant"],
                title=DEFAULT_SESSION_TITLE,
            )
            assistant_meta["route"] = "quick"
            memory_manager.store(final_text, assistant_meta)
            turn_index += 1
            if router.memory_manager:
                router.memory_manager.cleanup()
            continue

        decision = None
        if router.workflow_engine and "route_selector" in router.workflow_engine.executions:
            router_exec = router.workflow_engine.executions["route_selector"]
            decision = _extract_decision(router_exec.result)
        if not decision:
            decision = _extract_decision(router_result)
        decision = (decision or "quick").lower()

        final_text = _extract_text(router_result)

        if decision != "quick":
            route_config = _build_route_config(
                route=decision,
                user_prompt=prompt,
                recall_items=recall_items,
                base_memory_config=base_memory_config,
                user_id=args.user,
                verbose=args.verbose,
                max_iter=args.max_iter,
            )

            if route_config:
                route_orchestrator = Orchestrator(route_config)
                _attach_graph_tool(route_orchestrator, user_id=args.user, run_id=args.run)
                try:
                    route_result = route_orchestrator.run_sync()
                    final_text = _extract_text(route_result) or final_text
                except WorkflowError as exc:
                    logger.warning("Route %s failed: %s", decision, exc)
                    print("⚠️  El modelo devolvió un error 503; enviando una respuesta corta.")
                    decision = "quick"
                finally:
                    if route_orchestrator.memory_manager:
                        route_orchestrator.memory_manager.cleanup()
            else:
                decision = "quick"

        if router.memory_manager:
            router.memory_manager.cleanup()

        if final_text:
            assistant_meta = _conversation_metadata(
                args.user,
                args.run,
                agent_id="assistant",
                turn_index=turn_index,
                language=args.language,
                document_id=document_id,
                tags=["chat", "assistant"],
                title=DEFAULT_SESSION_TITLE,
            )
            assistant_meta["route"] = decision
            memory_manager.store(final_text, assistant_meta)
            turn_index += 1

    memory_manager.cleanup()
    return 0


def run_mock_info(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    config = create_mock_engineering_config()
    orchestrator = Orchestrator(config)
    info = orchestrator.get_system_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def run_memory_info(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    try:
        memory_config = resolve_memory_config(args.memory_provider)
    except Exception as exc:
        print(f"❌ Error construyendo configuración de memoria: {exc}")
        return 1
    payload = asdict(memory_config)
    payload["provider"] = memory_config.provider.value
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Inicia un chat multiagente desde la terminal")
    chat.add_argument("--memory-provider", choices=[p.value for p in MemoryProvider], default=os.getenv("ORCH_MEMORY_PROVIDER", "hybrid"))
    chat.add_argument("--user", default=os.getenv("ORCH_USER", "cli-user"))
    chat.add_argument("--run", default=os.getenv("ORCH_RUN", "cli-run"))
    chat.add_argument("--language", default=os.getenv("ORCH_LANGUAGE", "es"))
    chat.add_argument("--session-document", default=os.getenv("ORCH_SESSION_DOCUMENT"))
    chat.add_argument("--max-iter", type=int, default=_env_int("ORCH_MAX_ITER", 6) or 6)
    chat.add_argument("--verbose", type=int, default=_env_int("ORCH_VERBOSE", 2) or 2)
    chat.add_argument("--recall-top-k", type=int, default=_env_int("ORCH_RECALL_TOP_K", 10) or 10)
    chat.set_defaults(func=run_chat)

    mock = subparsers.add_parser("mock-info", help="Imprime información del orquestador mock de ingeniería")
    mock.add_argument("--verbose", type=int, default=1)
    mock.set_defaults(func=run_mock_info)

    mem = subparsers.add_parser("memory-info", help="Muestra la configuración de memoria resultante")
    mem.add_argument("--memory-provider", choices=[p.value for p in MemoryProvider], default=os.getenv("ORCH_MEMORY_PROVIDER", "hybrid"))
    mem.add_argument("--verbose", type=int, default=1)
    mem.set_defaults(func=run_memory_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("CLI execution failed")
        print(f"❌ CLI execution failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
