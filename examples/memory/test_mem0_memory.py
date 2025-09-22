"""
Test script: Conversational memory with Mem0 (local graph) integrated with the Orchestrator.

What it does:
- Builds an Orchestrator configured to use Mem0 as the memory provider
- Initializes Mem0 with a local graph store (Neo4j/Memgraph) via env vars
- Stores a few sample conversational facts (user/agent/run partitioning)
- Retrieves memories to verify filtering and consolidation behavior

Prereqs:
- pip install mem0ai "mem0ai[graph]" praisonaiagents
- A graph DB running locally (choose one):
  - Neo4j (bolt or neo4j+s)
  - Memgraph (bolt://localhost:7687)
- Env vars (examples):
  export OPENAI_API_KEY=...
  export MEM0_GRAPH_PROVIDER=neo4j            # or memgraph
  export NEO4J_URL=bolt://localhost:7687      # or neo4j+s://<host>
  export NEO4J_USER=neo4j
  export NEO4J_PASSWORD=password

Run:
  # Optional: prefer local clones over site-packages
  # export PRAISONAIAGENTS_PATH=/path/to/local/praisonaiagents
  # export MEM0_PATH=/path/to/local/mem0-repo-root
  python examples/memory/test_mem0_memory.py
"""

import os, sys
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file if present

# Prefer local clones if provided
local_paa = "/Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent/PraisonAI"
if local_paa and os.path.isdir(local_paa):
    sys.path.insert(0, os.path.abspath(local_paa))

# If a local mem0 repo exists in workspace or MEM0_PATH provided, prefer it
workspace_mem0 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mem0"))
mem0_path = os.getenv("MEM0_PATH") or (workspace_mem0 if os.path.isdir(workspace_mem0) else None)
if mem0_path and os.path.isdir(mem0_path):
    sys.path.insert(0, mem0_path)
    
from orchestrator.core.config import (
    OrchestratorConfig, MemoryConfig, MemoryProvider, EmbedderConfig,
    AgentConfig, TaskConfig, ExecutionConfig, ProcessType
)
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.cli.main import build_hybrid_memory_from_env


def build_mem0_memory_config() -> MemoryConfig:
    provider = os.getenv("MEM0_GRAPH_PROVIDER", "neo4j").lower()
    if provider not in ("neo4j", "memgraph"):
        provider = "neo4j"

    graph_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    graph_user = os.getenv("NEO4J_USER", "neo4j")
    graph_password = os.getenv("NEO4J_PASSWORD", "password")

    # Embedder: recommended OpenAI embeddings for technical domains
    embedder = EmbedderConfig(
        provider="openai",
        config={"model": "text-embedding-3-large"}
    )

    # Build MemoryConfig for Mem0 with graph store
    mem0_config = MemoryConfig(
        provider=MemoryProvider.MEM0,
        use_embedding=True,
        config={
            "graph_store": {
                "provider": provider,
                "config": {
                    "url": graph_url,
                    "username": graph_user,
                    "password": graph_password,
                },
            },
            # Optional vector store (Qdrant) will be injected below if enabled via env
            # Provide llm for Mem0 update/merge logic
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini"}
            },
            # Also pass embedder here (Orchestrator will mirror top-level too)
            "embedder": {
                "provider": embedder.provider,
                "config": embedder.config,
            },
        },
        embedder=embedder,
    )

    # Extensible vector store: enable Qdrant via env
    vector_provider = os.getenv("MEM0_VECTOR_PROVIDER", "").lower()
    if vector_provider == "qdrant":
        q_host = os.getenv("QDRANT_HOST", "localhost")
        q_port = int(os.getenv("QDRANT_PORT", "6333"))
        collection = os.getenv("QDRANT_COLLECTION", "mem0_memory")
        cfg = {
            "host": q_host,
            "port": q_port,
            "collection_name": collection,
        }
        # Optional for Qdrant Cloud
        if os.getenv("QDRANT_API_KEY"):
            cfg["api_key"] = os.getenv("QDRANT_API_KEY")
        if os.getenv("QDRANT_URL"):
            cfg["url"] = os.getenv("QDRANT_URL")
        if os.getenv("QDRANT_USE_HTTPS", "false").lower() in ("1", "true", "yes"):
            cfg["https"] = True
        mem0_config.config["vector_store"] = {
            "provider": "qdrant",
            "config": cfg,
        }

    return mem0_config


def resolve_memory_config() -> MemoryConfig:
    provider = os.getenv("MEMORY_PROVIDER", "mem0").lower()
    if provider == MemoryProvider.HYBRID.value:
        return build_hybrid_memory_from_env()
    return build_mem0_memory_config()


def build_minimal_config() -> OrchestratorConfig:
    mem_cfg = resolve_memory_config()

    # Simple execution config
    exec_cfg = ExecutionConfig(
        process=ProcessType.WORKFLOW,
        verbose=1,
        max_iter=4,
        memory=True,
        user_id=os.getenv("MEM0_TEST_USER", "engineer_user"),
        async_execution=False,  # keep it simple for the demo
    )

    cfg = OrchestratorConfig(
        name="Mem0MemoryDemo",
        memory_config=mem_cfg,
        execution_config=exec_cfg,
        embedder=mem_cfg.embedder,
    )

    # Minimal agent and task so PraisonAIAgents initializes properly
    cfg.agents.append(AgentConfig(
        name="Writer",
        role="Technical Writer",
        goal="Summarize notes",
        backstory="Synthesizes outputs into concise notes",
        instructions="Write a one-line note consolidating the key point.",
    ))

    cfg.tasks.append(TaskConfig(
        name="doc_task",
        description="Create a short note for testing memory",
        expected_output="One-line note",
        agent_name="Writer",
        async_execution=False,
        is_start=True,
    ))

    return cfg


def smoke_test_mem0(orchestrator: Orchestrator) -> None:
    print("\n== Provider Info ==")
    print(orchestrator.get_system_info().get("memory", {}))

    # Direct store/retrieve via MemoryManager
    # Use metadata partitioning as per PDF best practices
    mm = orchestrator.memory_manager
    assert mm is not None

    meta_common = {
        "user_id": orchestrator.config.execution_config.user_id,
        "run_id": "mem0_demo_run",
    }

    ref1 = mm.store("El usuario prefiere unidades métricas.", {**meta_common, "agent_id": "mecanico"})
    ref2 = mm.store("Suministro monofásico de 220V.", {**meta_common, "agent_id": "electrico"})
    print(f"Stored refs: {ref1}, {ref2}")

    # Mem0 requires at least one of user_id/agent_id/run_id for search
    results_all = mm.retrieve_filtered(
        "preferencias",
        user_id=meta_common["user_id"],
        run_id=meta_common["run_id"],
        limit=5,
    )
    print(f"Retrieved {len(results_all)} results for 'preferencias' (filtered by user/run)")
    for r in results_all[:3]:
        print("-", r.get("content"), r.get("metadata"))

    # Filtered by agent/discipline (e.g., mecánico)
    results_mech = mm.retrieve_filtered(
        "preferencias",
        user_id=meta_common["user_id"],
        agent_id="mecanico",
        run_id=meta_common["run_id"],
        limit=5,
    )
    print(f"Filtered (mecanico) results: {len(results_mech)}")
    for r in results_mech:
        print("  ·", r.get("content"), r.get("metadata"))


def main():
    cfg = build_minimal_config()
    orch = Orchestrator(cfg)
    try:
        import praisonaiagents as paa
        print("Using praisonaiagents from:", getattr(paa, "__file__", paa.__package__))
        import mem0 as mem0_mod
        print("Using mem0 from:", getattr(mem0_mod, "__file__", mem0_mod.__package__))
    except Exception:
        pass
    # Run a tiny workflow to ensure PraisonAIAgents boots with Mem0 memory
    try:
        result = orch.run_sync()
        print("Workflow result (truncated):", str(result)[:200])
    except Exception as e:
        print("Workflow run skipped due to LLM config or other error:", e)

    smoke_test_mem0(orch)


if __name__ == "__main__":
    main()
