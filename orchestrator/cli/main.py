"""
Orchestrator CLI

Usage examples:
  # Interactive chat with Mem0 conversational memory and router flow
  python -m orchestrator.cli chat --user alice --run demo1

Environment:
  # OpenAI, Mem0 graph (Neo4j/Memgraph). Optional Qdrant for vector store.
  See .env.example for variables.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any, List, Optional

from praisonaiagents import Agent, Task, PraisonAIAgents
from praisonaiagents.tools import duckduckgo
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env from project root if present
except Exception:
    pass

from orchestrator.core.config import (
    OrchestratorConfig,
    MemoryConfig,
    MemoryProvider,
    EmbedderConfig,
    ExecutionConfig,
    ProcessType,
)
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.memory.document_schema import default_conversation_metadata


def build_mem0_memory_from_env() -> MemoryConfig:
    provider = os.getenv("MEM0_GRAPH_PROVIDER", "neo4j").lower()
    graph_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    graph_user = os.getenv("NEO4J_USER", "neo4j")
    graph_password = os.getenv("NEO4J_PASSWORD", "password")

    embedder = EmbedderConfig(
        provider="openai",
        config={"model": "text-embedding-3-large"},
    )

    config: Dict[str, Any] = {
        "graph_store": {
            "provider": provider,
            "config": {"url": graph_url, "username": graph_user, "password": graph_password},
        },
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 1}},
        "embedder": {"provider": embedder.provider, "config": dict(embedder.config)},
    }

    # Optional Qdrant vector store
    if os.getenv("MEM0_VECTOR_PROVIDER", "").lower() == "qdrant":
        qcfg = {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "collection_name": os.getenv("QDRANT_COLLECTION", "mem0_memory"),
        }
        if os.getenv("QDRANT_API_KEY"):
            qcfg["api_key"] = os.getenv("QDRANT_API_KEY")
        if os.getenv("QDRANT_URL"):
            qcfg["url"] = os.getenv("QDRANT_URL")
        if os.getenv("QDRANT_USE_HTTPS", "false").lower() in ("1", "true", "yes"):
            qcfg["https"] = True
        config["vector_store"] = {"provider": "qdrant", "config": qcfg}

    return MemoryConfig(provider=MemoryProvider.MEM0, use_embedding=True, config=config, embedder=embedder)


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_hybrid_memory_from_env() -> MemoryConfig:
    embedder_provider = os.getenv("HYBRID_EMBEDDER_PROVIDER", "openai")
    embedder_model = os.getenv("HYBRID_EMBEDDER_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedder = EmbedderConfig(
        provider=embedder_provider,
        config={"model": embedder_model},
    )

    vector_path = os.getenv("HYBRID_VECTOR_PATH", ".praison/hybrid_chroma")
    vector_collection = os.getenv("HYBRID_VECTOR_COLLECTION", "hybrid_memory")

    lexical_db = os.getenv("HYBRID_LEXICAL_DB_PATH", ".praison/hybrid_lexical.db")
    lexical_ttl = os.getenv("HYBRID_LEXICAL_TTL_SECONDS")
    lexical_max_entries = os.getenv("HYBRID_LEXICAL_MAX_ENTRIES")
    cleanup_interval = os.getenv("HYBRID_LEXICAL_CLEANUP_INTERVAL", "200")

    rerank_enabled = _env_flag("HYBRID_RERANK_ENABLED", True)
    rerank_model = os.getenv("HYBRID_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_top_k = os.getenv("HYBRID_RERANK_TOP_K", "10")
    rerank_workers = os.getenv("HYBRID_RERANK_MAX_WORKERS", "1")

    graph_enabled = _env_flag("HYBRID_GRAPH_ENABLED", False)
    graph_uri = os.getenv("HYBRID_GRAPH_URI", os.getenv("NEO4J_URL"))
    graph_user = os.getenv("HYBRID_GRAPH_USER", os.getenv("NEO4J_USER"))
    graph_password = os.getenv("HYBRID_GRAPH_PASSWORD", os.getenv("NEO4J_PASSWORD"))

    config: Dict[str, Any] = {
        "vector_store": {
            "provider": "chromadb",
            "config": {
                "path": vector_path,
                "collection": vector_collection,
            },
        },
        "lexical": {
            "db_path": lexical_db,
            "ttl_seconds": lexical_ttl,
            "max_entries": lexical_max_entries,
            "cleanup_interval": cleanup_interval,
        },
        "rerank": {
            "enabled": rerank_enabled,
            "model": rerank_model,
            "top_k": rerank_top_k,
            "max_workers": rerank_workers,
        },
    }

    if graph_enabled and graph_uri and graph_user and graph_password:
        config["graph"] = {
            "enabled": True,
            "uri": graph_uri,
            "user": graph_user,
            "password": graph_password,
        }
    else:
        config["graph"] = {"enabled": False}

    return MemoryConfig(provider=MemoryProvider.HYBRID, use_embedding=True, config=config, embedder=embedder)


def resolve_memory_config(provider_name: str) -> MemoryConfig:
    normalized = (provider_name or "mem0").strip().lower()
    if normalized == MemoryProvider.HYBRID.value:
        return build_hybrid_memory_from_env()
    return build_mem0_memory_from_env()


def build_router_flow(prompt: str, recall_context: Optional[str], user_id: str, mem_config: Dict[str, Any]) -> PraisonAIAgents:
    """Build a routing workflow for a single user prompt.

    Orchestrator agent decides route -> next task (quick/research/analysis/planner)
    """
    # Agents
    orchestrator_agent = Agent(
        name="Orchestrator",
        role="Router & Coordinator",
        goal="Decide which specialist should handle the user's request",
        instructions=(
            "You are the orchestrator. Given the prompt and any MEMORY RECALL CONTEXT, choose one route:"
            " 'quick' (short greeting/answer), 'research' (web research), 'analysis' (deeper reasoning),"
            " or 'planner' (break down into steps). Output only the decision and brief rationale."
        ),
    )

    quick_agent = Agent(
        name="QuickResponder",
        role="Short Response Agent",
        goal="Provide concise answers or greetings",
        instructions=(
            "Answer in one or two sentences. If MEMORY RECALL CONTEXT contains facts"
            " like the user's name or preferences, use them to answer."
        )
    )

    research_agent = Agent(
        name="Researcher",
        role="Web Research Specialist",
        goal="Gather up-to-date, sourced information",
        backstory="Expert in web research and summarization.",
        tools=[duckduckgo],
        instructions="Use web search to collect reliable, relevant information with sources.",
    )

    analysis_agent = Agent(
        name="Analyst",
        role="Analysis Specialist",
        goal="Reason about the user's request and produce a deeper analysis",
        instructions=(
            "Provide structured analysis, assumptions, and recommendations."
            " Use MEMORY RECALL CONTEXT if relevant."
        ),
    )

    planner_agent = Agent(
        name="Planner",
        role="Planner Specialist",
        goal="Break down complex tasks into a step-by-step plan",
        instructions=(
            "Produce an actionable plan with steps, owners, and acceptance criteria."
            " Use MEMORY RECALL CONTEXT to personalize if relevant."
        ),
    )

    # Compose shared suffix for task descriptions (instead of putting strings in context)
    desc_suffix = f"\n\nUSER PROMPT:\n{prompt}"
    if recall_context:
        desc_suffix += f"\n\n{recall_context}"

    # Router decision task
    router_task = Task(
        name="router_task",
        description=(
            "Choose the correct route for the user's request."
            " Options: quick | research | analysis | planner." + desc_suffix
        ),
        expected_output=(
            "Decision: one of quick|research|analysis|planner, with a one-line justification"
        ),
        agent=orchestrator_agent,
        task_type="decision",
        condition={
            "quick": ["quick_task"],
            "research": ["research_task"],
            "analysis": ["analysis_task"],
            "planner": ["planner_task"],
        },
        is_start=True,
        context=None,
    )

    quick_task = Task(
        name="quick_task",
        description=(
            "Respond concisely to the user's prompt." + desc_suffix
        ),
        expected_output="A one or two sentence reply",
        agent=quick_agent,
        context=[router_task],
    )

    research_task = Task(
        name="research_task",
        description=(
            "Conduct web research relevant to the user's prompt and summarize findings with sources." + desc_suffix
        ),
        expected_output="Short summary with 2-3 sources",
        agent=research_agent,
        context=[router_task],
    )

    analysis_task = Task(
        name="analysis_task",
        description=(
            "Analyze the user's prompt and produce a deeper reasoning-based answer." + desc_suffix
        ),
        expected_output="Structured analysis with recommendations",
        agent=analysis_agent,
        context=[router_task],
    )

    planner_task = Task(
        name="planner_task",
        description=(
            "Create an actionable plan by splitting the task into steps with acceptance criteria." + desc_suffix
        ),
        expected_output="Plan with steps and acceptance criteria",
        agent=planner_agent,
        context=[router_task],
    )

    # We no longer return a full multi-branch system because the engine
    # can execute unrelated tasks via fallback. Instead, we run routing
    # as a single-task system, then run only the chosen branch separately.
    memory_enabled = bool(mem_config)

    sys_kwargs: Dict[str, Any] = {
        "agents": [orchestrator_agent],
        "tasks": [router_task],
        "process": "workflow",
        "verbose": 1,
        "max_iter": 3,
        "name": "CLIOrchestratorRouterOnly",
        "memory": memory_enabled,
        "user_id": user_id,
    }

    if memory_enabled:
        sys_kwargs["memory_config"] = mem_config
        if "embedder" in mem_config:
            sys_kwargs["embedder"] = mem_config["embedder"]
    return PraisonAIAgents(**sys_kwargs)


def build_single_route_flow(route: str, prompt: str, recall_context: Optional[str], user_id: str, mem_config: Dict[str, Any]) -> PraisonAIAgents:
    """Build a minimal system to execute only the chosen route."""
    desc_suffix = f"\n\nUSER PROMPT:\n{prompt}"
    if recall_context:
        desc_suffix += f"\n\n{recall_context}"

    if route == "quick":
        agent = Agent(
            name="QuickResponder",
            role="Short Response Agent",
            goal="Provide concise answers or greetings",
            instructions="Answer in one or two sentences.",
        )
        task = Task(
            name="quick_task",
            description="Respond concisely to the user's prompt." + desc_suffix,
            expected_output="A one or two sentence reply",
            agent=agent,
        )
    elif route == "research":
        agent = Agent(
            name="Researcher",
            role="Web Research Specialist",
            goal="Gather up-to-date, sourced information",
            backstory="Expert in web research and summarization.",
            tools=[duckduckgo],
            instructions="Use web search to collect reliable, relevant information with sources.",
        )
        task = Task(
            name="research_task",
            description="Conduct web research relevant to the user's prompt and summarize findings with sources." + desc_suffix,
            expected_output="Short summary with 2-3 sources",
            agent=agent,
        )
    elif route == "analysis":
        agent = Agent(
            name="Analyst",
            role="Analysis Specialist",
            goal="Reason about the user's request and produce a deeper analysis",
            instructions="Provide structured analysis, assumptions, and recommendations.",
        )
        task = Task(
            name="analysis_task",
            description="Analyze the user's prompt and produce a deeper reasoning-based answer." + desc_suffix,
            expected_output="Structured analysis with recommendations",
            agent=agent,
        )
    else:  # planner
        agent = Agent(
            name="Planner",
            role="Planner Specialist",
            goal="Break down complex tasks into a step-by-step plan",
            instructions="Produce an actionable plan with steps, owners, and acceptance criteria.",
        )
        task = Task(
            name="planner_task",
            description="Create an actionable plan by splitting the task into steps with acceptance criteria." + desc_suffix,
            expected_output="Plan with steps and acceptance criteria",
            agent=agent,
        )

    memory_enabled = bool(mem_config)

    kwargs: Dict[str, Any] = {
        "agents": [agent],
        "tasks": [task],
        "process": "workflow",
        "verbose": 1,
        "max_iter": 3,
        "name": f"CLIOrchestrator_{route}",
        "memory": memory_enabled,
        "user_id": user_id,
    }

    if memory_enabled:
        kwargs["memory_config"] = mem_config
        if "embedder" in mem_config:
            kwargs["embedder"] = mem_config["embedder"]
    
    return PraisonAIAgents(**kwargs)


def convert_memory_config_for_praison(mc: MemoryConfig, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Convert MemoryConfig dataclass to dict expected by PraisonAIAgents."""
    if mc.provider == MemoryProvider.HYBRID:
        # Hybrid provider is managed directly via Orchestrator's MemoryManager; disable PraisonAI memory.
        return None

    d: Dict[str, Any] = {
        "provider": mc.provider.value,
        "use_embedding": mc.use_embedding,
        "config": dict(mc.config or {}),
    }
    if mc.short_db:
        d["short_db"] = mc.short_db
    if mc.long_db:
        d["long_db"] = mc.long_db
    if mc.rag_db_path:
        d["rag_db_path"] = mc.rag_db_path
    if mc.embedder:
        # Ensure embedding_dims present for Mem0 compatibility
        emb_conf = dict(mc.embedder.config or {})
        if "embedding_dims" not in emb_conf:
            model = str(emb_conf.get("model", "")).lower()
            dims_map = {
                "text-embedding-3-large": 3072,
                "text-embedding-3-small": 1536,
                "text-embedding-ada-002": 1536,
                "text-embedding-002": 1536,
            }
            emb_conf["embedding_dims"] = next((v for k, v in dims_map.items() if k in model), 1536)
        d["embedder"] = {"provider": mc.embedder.provider, "config": emb_conf}

        # Mirror embedder + llm inside Mem0 config; always enforce embedding_dims there
        if d.get("provider") == "mem0":
            mem0_cfg = d.setdefault("config", {})
            existing_embedder = mem0_cfg.get("embedder") or {}
            mem0_embedder_conf = dict(existing_embedder.get("config", {}))
            # Merge with top-level emb_conf, prioritizing explicit values
            merged_emb_conf = {**emb_conf, **mem0_embedder_conf}
            if "embedding_dims" not in merged_emb_conf:
                merged_emb_conf["embedding_dims"] = emb_conf.get("embedding_dims", 1536)
            mem0_cfg["embedder"] = {
                "provider": existing_embedder.get("provider") or mc.embedder.provider,
                "config": merged_emb_conf,
            }
            # Provide a sensible default llm if missing
            mem0_cfg.setdefault("llm", {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 1}})
            
            # Add user_id to Mem0 config if provided
            if user_id:
                mem0_cfg.setdefault("user_id", user_id)
    return d


def build_recall(memory_manager: MemoryManager, query: str, user_id: str, run_id: Optional[str], top_k: int = 5, agent_id: Optional[str] = None) -> Optional[str]:
    """Build a global recall string from conversational memory (Mem0).
    
    Args:
        memory_manager: Memory manager instance
        query: Search query for memory recall
        user_id: User ID for filtering
        run_id: Run/session ID for filtering
        top_k: Maximum number of results to return
        agent_id: Optional agent ID for agent-specific memory filtering
    """
    if not memory_manager:
        return None
    try:
        results = memory_manager.retrieve_filtered(
            query,
            limit=top_k,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,  # Enable agent-specific filtering
        )
    except Exception:
        return None
    if not results:
        return None
    lines = ["MEMORY RECALL CONTEXT:"]
    for r in results:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        md = r.get("metadata", {}) or {}
        src = md.get("filename") or md.get("sheet_id") or r.get("id")
        if src:
            lines.append(f"- {content} [src: {src}]")
        else:
            lines.append(f"- {content}")
    return "\n".join(lines) if len(lines) > 1 else None


def store_conversational_memory(memory_manager: MemoryManager, text: str, *, user_id: str, run_id: str, agent_id: str) -> None:
    if not memory_manager:
        return
    metadata = default_conversation_metadata(user_id, run_id, agent_id)
    try:
        memory_manager.store(text, metadata)
    except Exception:
        pass


def cmd_chat(args: argparse.Namespace) -> int:
    user_id = args.user or os.getenv("ORCH_USER", "cli-user")
    run_id = args.run or os.getenv("ORCH_RUN", "cli-run")

    provider_name = args.memory_provider or os.getenv("ORCH_MEMORY_PROVIDER", "mem0")

    # Build memory manager directly (avoid full orchestrator initialization issues)
    mem_cfg = resolve_memory_config(provider_name)
    
    # Create memory manager
    try:
        memory_manager = MemoryManager(mem_cfg)
        print(f"✅ Memory manager initialized for user: {user_id} (provider={mem_cfg.provider.value})")
    except Exception as e:
        print(f"❌ Failed to initialize memory manager: {e}")
        return 1

    praison_memory_config = convert_memory_config_for_praison(mem_cfg, user_id)

    print("Enter your prompt (type 'exit' to quit):")
    while True:
        try:
            prompt = input("» ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", ":q"}:
            break

        # Store user prompt in conversational memory (Mem0)
        try:
            store_conversational_memory(memory_manager, prompt, user_id=user_id, run_id=run_id, agent_id="user")
            # Synthesize generic facts (no regex) for broad usefulness
            
        except Exception:
            pass

        recall = None if args.no_recall else build_recall(memory_manager, prompt, user_id, run_id, top_k=args.top_k)

        # Phase 1: routing only
        router_system = build_router_flow(prompt, recall, user_id, praison_memory_config)
        try:
            route_result = router_system.start(return_dict=True)
        except Exception as e:
            print("[ERROR] Router failed:", e)
            continue

        # Extract decision safely
        decision = None
        try:
            # Last task result
            task_ids = list(router_system.tasks.keys())
            if task_ids:
                last_id = task_ids[-1]
                last = router_system.get_task_result(last_id)
                if last and last.pydantic and hasattr(last.pydantic, "decision"):
                    decision = last.pydantic.decision.lower()
                elif last and last.raw:
                    import re, json
                    raw = last.raw.strip()
                    # Try JSON first
                    try:
                        obj = json.loads(raw)
                        decision = str(obj.get("decision", "")).lower()
                    except Exception:
                        m = re.search(r"decision\"?\s*[:=]\s*\"?(quick|research|analysis|planner)\"?", raw, re.I)
                        if m:
                            decision = m.group(1).lower()
        except Exception:
            pass

        decision = decision or "quick"
        # Phase 2: run only chosen branch
        branch_system = build_single_route_flow(decision, prompt, recall, user_id, praison_memory_config)
        try:
            result = branch_system.start()
        except Exception as e:
            print("[ERROR] Branch execution failed:", e)
            continue

        print("\n=== Result (", decision, ") ===", sep="")
        print(result if isinstance(result, str) else str(result))
        print("============\n")

        # Store agent response and synthesize facts as well
        try:
            agent_id = decision
            if isinstance(result, str):
                store_conversational_memory(memory_manager, result, user_id=user_id, run_id=run_id, agent_id=agent_id)
                
        except Exception:
            pass

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="orchestrator", description="Orchestrator CLI")
    sub = parser.add_subparsers(dest="command")

    p_chat = sub.add_parser("chat", help="Interactive chat with orchestrator router")
    p_chat.add_argument("--user", help="User ID for memory partitioning")
    p_chat.add_argument("--run", help="Run/session ID for memory partitioning")
    p_chat.add_argument("--no-recall", action="store_true", help="Disable memory recall injection")
    p_chat.add_argument("--top-k", type=int, default=5, help="Recall items to inject")
    p_chat.add_argument(
        "--memory-provider",
        choices=[MemoryProvider.MEM0.value, MemoryProvider.HYBRID.value],
        help="Memory provider to use (defaults to env ORCH_MEMORY_PROVIDER or mem0)",
    )
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
