"""Tree-of-Thought based planner integration for the orchestrator."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from orchestrator.core.config import AgentConfig, MemoryConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import of tree-of-thought-llm package
# ---------------------------------------------------------------------------
_TOT_AVAILABLE = False


_TOT_SRC = Path(__file__).resolve().parents[2] / "tree-of-thought-llm" / "src"
logger.debug("Configured Tree-of-Thought source directory: %s", _TOT_SRC)
if _TOT_SRC.exists():
    if str(_TOT_SRC) not in sys.path:
        sys.path.insert(0, str(_TOT_SRC))
        logger.debug("Added Tree-of-Thought path to sys.path")
else:
    logger.debug("Tree-of-Thought source directory not found at %s", _TOT_SRC)

solve = None  # type: ignore
Task = object  # type: ignore
gpt_usage = lambda backend: {}  # type: ignore
_TOT_OPENAI_SHIM: Optional[ModuleType] = None


def _ensure_openai_stub() -> Optional[ModuleType]:
    """Load the Tree-of-Thought OpenAI shim without permanently overriding imports."""

    global _TOT_OPENAI_SHIM

    if _TOT_OPENAI_SHIM is not None:
        return _TOT_OPENAI_SHIM

    shim_path = _TOT_SRC / "openai" / "__init__.py"
    if not shim_path.exists():
        logger.warning(
            "Tree-of-Thought OpenAI shim missing at %s; cannot initialize planner",
            shim_path,
        )
        return None

    try:
        spec = importlib.util.spec_from_file_location("tot_openai_shim", shim_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not create spec for openai shim")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        _TOT_OPENAI_SHIM = module
        logger.info("Tree-of-Thought openai shim loaded from %s", shim_path)
        return module
    except Exception:
        logger.exception(
            "Failed to load Tree-of-Thought openai shim from %s", shim_path
        )
        _TOT_OPENAI_SHIM = None
        return None


def _try_import_tot() -> bool:
    """Attempt to import Tree-of-Thought dependencies lazily."""

    global solve, Task, gpt_usage, _TOT_AVAILABLE

    if _TOT_AVAILABLE:
        return True

    if not _TOT_SRC.exists():
        logger.warning(
            "Tree-of-Thought source directory missing at %s; planning disabled",
            _TOT_SRC,
        )
        return False

    if not _ensure_openai_stub():
        return False

    shim_module = _ensure_openai_stub()
    if shim_module is None:
        return False

    original_openai = sys.modules.get("openai")

    # Ensure previous partial imports do not interfere
    for name in list(sys.modules.keys()):
        if name == "tot" or name.startswith("tot."):
            sys.modules.pop(name, None)

    sys.modules["openai"] = shim_module

    try:
        from tot.methods.bfs import solve as _solve
        from tot.tasks.base import Task as _Task
        from tot.models import gpt_usage as _gpt_usage
    except Exception:
        logger.exception(
            "Tree-of-thought package import failed from %s", _TOT_SRC
        )
        return False
    finally:
        if original_openai is not None:
            sys.modules["openai"] = original_openai
        else:
            sys.modules.pop("openai", None)

    solve = _solve
    Task = _Task
    gpt_usage = _gpt_usage
    _TOT_AVAILABLE = True
    logger.info("Tree-of-thought-llm package imported from %s", _TOT_SRC)
    return True


# Try eager import so issues surface early, but continue even if unavailable.
_try_import_tot()


@dataclass
class PlanningSettings:
    """Configuration options for the ToT planner."""

    backend: str = "gpt-4"
    temperature: float = 0.7
    max_steps: int = 3
    n_generate_sample: int = 3
    n_evaluate_sample: int = 2
    n_select_sample: int = 2
    prompt_style: str = "cot"  # 'standard' or 'cot'


class OrchestratorPlanningTask(Task):
    """Custom ToT task for orchestrator planning."""

    def __init__(
        self,
        problem_statement: str,
        agent_catalog: Sequence[AgentConfig],
        max_steps: int,
        prompt_style: str = "cot",
    ) -> None:
        super().__init__()
        self.instances = [problem_statement]
        self.agent_catalog = list(agent_catalog)
        self.steps = max_steps
        self.prompt_style = (
            prompt_style if prompt_style in {"cot", "standard"} else "cot"
        )
        # Use newline termination for each step to keep outputs concise
        self.stops: List[Optional[str]] = ["\n"] * max_steps
        self.value_cache: Dict[str, float] = {}

    # --------------- Task protocol implementations ---------------
    def __len__(self) -> int:
        return len(self.instances)

    def get_input(self, idx: int) -> str:
        return self.instances[idx]

    def test_output(
        self, idx: int, output: str
    ) -> Dict[str, Any]:  # pragma: no cover - diagnostics
        """Simple structural validation for diagnostic purposes."""
        assignments = parse_plan_to_assignments(output)
        return {"num_tasks": len(assignments)}

    # --------------- Prompt helpers ---------------
    def _base_prompt(self, x: str, y: str = "") -> str:
        agent_lines = [
            f"- {agent.name}: {agent.role}. Objetivo base: {agent.goal}"
            for agent in self.agent_catalog
        ]
        current_plan = y.strip() if y.strip() else "<Sin pasos definidos>"
        return (
            "Eres un orquestador de agentes de ingeniería."
            " Tu objetivo es descomponer el problema en tareas claras asignadas a agentes especializados.\n"
            f"Problema: {x.strip()}\n\n"
            "Agentes disponibles:\n"
            f"{chr(10).join(agent_lines)}\n\n"
            f"Plan actual:\n{current_plan}\n\n"
            "Propón el siguiente paso del plan en una única línea con el formato exacto:\n"
            "Agent: <nombre> | Objective: <acción concreta> | Deliverable: <resultado> | Tags: <etiquetas separadas por coma>\n"
            "No añadas explicaciones adicionales."
        )

    @staticmethod
    def _ensure_plan_suffix(y: str, addition: str) -> str:
        if not addition.endswith("\n"):
            addition = addition + "\n"
        return y + addition

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        raise NotImplementedError  # dynamically assigned per instance

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        raise NotImplementedError  # dynamically assigned per instance

    def _wrap_prompt(self, x: str, y: str = "") -> str:
        return self._base_prompt(x, y)

    # The ToT solver looks up attributes on the class, so we define wrappers dynamically.
    def standard_prompt_wrap(self, x: str, y: str = "") -> str:  # type: ignore[override]
        return self._wrap_prompt(x, y)

    def cot_prompt_wrap(self, x: str, y: str = "") -> str:  # type: ignore[override]
        prompt = (
            self._wrap_prompt(x, y)
            + "\nPiensa paso a paso antes de decidir la línea final. Al final devuelve solo la línea solicitada."
        )
        return prompt

    def propose_prompt_wrap(
        self, x: str, y: str = ""
    ) -> str:  # pragma: no cover - fallback
        return self._wrap_prompt(x, y)

    def value_prompt_wrap(self, x: str, y: str) -> str:  # type: ignore[override]
        plan = y.strip()
        if not plan:
            plan = "<Sin pasos>"
        return (
            "Evalúa la calidad del siguiente plan parcial para resolver la solicitud."
            ' Responde en JSON estricto con el formato {"score": <0-10>, "reason": "..."}.'
            f"\nSolicitud: {x.strip()}\nPlan parcial:\n{plan}\n"
            "Ten en cuenta cobertura del problema, orden lógico, claridad y uso adecuado de agentes."
        )

    def value_outputs_unwrap(self, x: str, y: str, value_outputs: List[str]) -> float:  # type: ignore[override]
        best_score = 0.0
        for raw in value_outputs:
            try:
                data = json.loads(raw)
                score = float(data.get("score", 0))
            except Exception:
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
                score = float(match.group(1)) if match else 0.0
            best_score = max(best_score, score)
        return best_score

    def vote_prompt_wrap(
        self, x: str, ys: List[str]
    ) -> str:  # pragma: no cover - unused
        enumerated = "\n".join(f"Plan {i+1}:\n{y.strip()}" for i, y in enumerate(ys))
        return (
            "Selecciona el plan parcial más prometedor para resolver la solicitud."
            " Devuelve el número correspondiente.\n"
            f"Solicitud: {x.strip()}\n{enumerated}\n"
        )

    def vote_outputs_unwrap(
        self, vote_outputs: List[str], n_candidates: int
    ) -> List[int]:  # pragma: no cover - unused
        votes = [0] * n_candidates
        for raw in vote_outputs:
            match = re.search(r"(\d+)", raw)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < n_candidates:
                    votes[idx] += 1
        return votes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_memory_config(memory_config: MemoryConfig) -> str:
    parts: List[str] = []
    parts.append(f"Proveedor: {memory_config.provider.value}")
    if memory_config.use_embedding:
        embed_model = None
        if memory_config.embedder and isinstance(memory_config.embedder.config, dict):
            embed_model = memory_config.embedder.config.get("model")
        if embed_model:
            parts.append(f"Embeddings: {embed_model}")
        else:
            parts.append("Embeddings: habilitados")
    else:
        parts.append("Embeddings: deshabilitados")

    config = memory_config.config or {}
    vector_store = config.get("vector_store")
    if isinstance(vector_store, dict):
        provider = vector_store.get("provider") or vector_store.get("type")
        if provider:
            parts.append(f"Vector store: {provider}")
        collection = vector_store.get("config", {}).get("collection")
        if collection:
            parts.append(f"Colección vectorial: {collection}")

    lexical = config.get("lexical")
    if isinstance(lexical, dict):
        db_path = lexical.get("db_path")
        if db_path:
            parts.append(f"Lexical DB: {db_path}")

    graph = config.get("graph")
    if isinstance(graph, dict):
        enabled = graph.get("enabled", False)
        parts.append("Grafo: habilitado" if enabled else "Grafo: deshabilitado")
        if enabled and graph.get("uri"):
            parts.append(f"Graph URI: {graph['uri']}")

    rerank = config.get("rerank")
    if isinstance(rerank, dict) and rerank.get("enabled"):
        model = rerank.get("model")
        if model:
            parts.append(f"Rerank model: {model}")

    return "\n".join(f"- {line}" for line in parts if line)


def build_problem_statement(
    prompt: str,
    recall_snippets: Sequence[str],
    agent_catalog: Sequence[AgentConfig],
    memory_config: Optional[MemoryConfig] = None,
) -> str:
    context_lines = [f"- {snippet}" for snippet in recall_snippets if snippet]
    agents_lines = [
        f"- {agent.name}: {agent.role}. Objetivo base: {agent.goal}"
        for agent in agent_catalog
    ]
    memory_block = ""
    if memory_config:
        summary = summarize_memory_config(memory_config)
        if summary:
            memory_block = f"Memoria activa:\n{summary}\n"
    return (
        f"Solicitud del usuario: {prompt.strip()}\n"
        "Contexto recuperado:\n"
        f"{chr(10).join(context_lines) if context_lines else '- (sin contexto relevante)'}\n"
        "Agentes disponibles:\n"
        f"{chr(10).join(agents_lines)}\n"
        f"{memory_block}"
        "Diseña un plan con tareas independientes, cada una a cargo de un agente."
    )


def parse_plan_to_assignments(plan_text: str) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    if not plan_text:
        return assignments
    pattern = re.compile(
        r"Agent:\s*(?P<agent>[^|]+)\|\s*Objective:\s*(?P<objective>[^|]+)\|\s*Deliverable:\s*(?P<deliverable>[^|]+)\|\s*Tags:\s*(?P<tags>.+)",
        re.IGNORECASE,
    )
    for line in plan_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        agent = match.group("agent").strip()
        objective = match.group("objective").strip()
        deliverable = match.group("deliverable").strip()
        tags_raw = match.group("tags").strip()
        tags = [t.strip() for t in re.split(r",|;", tags_raw) if t.strip()]
        assignments.append(
            {
                "agent": agent,
                "objective": objective,
                "expected_output": deliverable,
                "tags": tags,
            }
        )
    return assignments


def generate_plan_with_tot(
    prompt: str,
    recall_snippets: Sequence[str],
    agent_catalog: Sequence[AgentConfig],
    settings: Optional[PlanningSettings] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> Dict[str, Any]:
    """Run Tree-of-Thought planning and return assignments alongside metadata."""
    if not _TOT_AVAILABLE and not _try_import_tot():
        raise RuntimeError(
            "tree-of-thought-llm package is not available in the environment"
        )

    settings = settings or PlanningSettings()
    if not agent_catalog:
        raise ValueError("agent_catalog must not be empty")

    problem = build_problem_statement(
        prompt, recall_snippets, agent_catalog, memory_config
    )
    logger.info(
        "Invoking Tree-of-Thought planner (backend=%s, steps=%d, agents=%d)",
        settings.backend,
        settings.max_steps,
        len(agent_catalog),
    )
    logger.debug("ToT planning problem statement:\n%s", problem)
    task = OrchestratorPlanningTask(
        problem,
        agent_catalog,
        max_steps=settings.max_steps,
        prompt_style=settings.prompt_style,
    )

    args = SimpleNamespace(
        backend=settings.backend,
        temperature=settings.temperature,
        method_generate="sample",
        method_evaluate="value",
        method_select="greedy",
        n_generate_sample=settings.n_generate_sample,
        n_evaluate_sample=settings.n_evaluate_sample,
        n_select_sample=settings.n_select_sample,
        prompt_sample=settings.prompt_style,
    )

    try:
        plans, info = solve(args, task, idx=0, to_print=False)
    except Exception as exc:
        logger.exception("ToT planner failed during solve()")
        raise

    best_plan = plans[0] if plans else ""
    assignments = parse_plan_to_assignments(best_plan)
    logger.info(
        "ToT planner completed search with %d plan(s); best assigns %d task(s)",
        len(plans),
        len(assignments),
    )
    if best_plan:
        logger.debug("ToT planner best plan:\n%s", best_plan)

    metadata = {
        "raw_plan": best_plan,
        "tot_usage": gpt_usage(settings.backend),
        "steps": info.get("steps") if isinstance(info, dict) else info,
    }
    logger.debug("ToT planner metadata: %s", metadata)
    return {"assignments": assignments, "metadata": metadata}


__all__ = [
    "generate_plan_with_tot",
    "PlanningSettings",
    "parse_plan_to_assignments",
    "summarize_memory_config",
]
