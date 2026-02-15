"""
Per-agent token cost tracking and budget enforcement.

Instruments LLM calls via LangChain callbacks to log input/output/reasoning
tokens per agent per workflow. Enforces configurable max_tokens_per_task and
max_cost_per_run budgets with alert thresholds.

Addresses: BP-COST-01
"""

import logging
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult
    except ImportError:
        # Provide minimal stubs so the module can still be imported
        BaseCallbackHandler = object  # type: ignore[misc,assignment]
        LLMResult = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 000 000 tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "reasoning": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00, "reasoning": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "reasoning": 30.00},
    "o1": {"input": 15.00, "output": 60.00, "reasoning": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40, "reasoning": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40, "reasoning": 4.40},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00, "reasoning": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "reasoning": 4.00},
    # Cohere
    "command-r-plus": {"input": 2.50, "output": 10.00, "reasoning": 10.00},
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    agent_name: str = ""
    timestamp: float = 0.0
    estimated_cost: float = 0.0


@dataclass
class AgentTokenSummary:
    """Aggregated token usage for a single agent across all its calls."""

    agent_name: str = ""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


@dataclass
class CostConfig:
    """Budget and tracking configuration."""

    max_tokens_per_task: Optional[int] = None
    max_cost_per_run: Optional[float] = None
    alert_threshold_pct: float = 0.8
    pricing: Optional[Dict[str, Dict[str, float]]] = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# LangChain callback handler
# ---------------------------------------------------------------------------


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Captures token usage from LLM responses (provider-agnostic).

    Works with OpenAI, Anthropic and Cohere via LangChain's normalised
    ``LLMResult`` interface.  Falls back to ``usage_metadata`` on the
    response message when ``llm_output`` is absent.
    """

    def __init__(self, agent_name: str, model: str = ""):
        super().__init__()
        self.agent_name = agent_name
        self.model = model
        self.usage = TokenUsage(
            agent_name=agent_name, model=model, timestamp=time.time()
        )

    # -- LangChain callback --------------------------------------------------

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # noqa: ARG002
        """Extract token counts from *response*.

        Priority order:
        1. ``response.llm_output["token_usage"]`` (dict with provider keys)
        2. ``generation.message.usage_metadata`` (LangChain ≥ 0.3)
        3. Log a warning and leave counts at zero.
        """
        token_usage = self._extract_from_llm_output(response)
        if token_usage is None:
            token_usage = self._extract_from_usage_metadata(response)
        if token_usage is None:
            logger.warning(
                "No token usage data for agent '%s' (model=%s). "
                "Counts will be zero for this call.",
                self.agent_name,
                self.model,
            )
            return

        inp = token_usage.get("input_tokens") or token_usage.get("prompt_tokens", 0)
        out = token_usage.get("output_tokens") or token_usage.get(
            "completion_tokens", 0
        )
        reasoning = token_usage.get("reasoning_tokens", 0)
        total = token_usage.get("total_tokens", 0) or (inp + out + reasoning)

        self.usage.input_tokens += inp
        self.usage.output_tokens += out
        self.usage.reasoning_tokens += reasoning
        self.usage.total_tokens += total

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _extract_from_llm_output(response: LLMResult) -> Optional[Dict[str, int]]:
        if response.llm_output and isinstance(response.llm_output, dict):
            tu = response.llm_output.get("token_usage")
            if tu and isinstance(tu, dict):
                return tu
        return None

    @staticmethod
    def _extract_from_usage_metadata(response: LLMResult) -> Optional[Dict[str, int]]:
        try:
            generations = response.generations
            if generations and generations[0]:
                gen = generations[0][0]
                msg = getattr(gen, "message", None)
                if msg is not None:
                    metadata = getattr(msg, "usage_metadata", None)
                    if metadata and isinstance(metadata, dict):
                        return metadata
        except (IndexError, AttributeError):
            pass
        return None


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """Per-agent token cost tracker with optional budget enforcement.

    Thread-safe: a ``threading.Lock`` protects mutable state so that
    parallel LangGraph nodes can record usage concurrently.
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
        self.enabled = self.config.enabled
        self._lock = threading.Lock()
        self._usage_log: List[TokenUsage] = []
        self._agent_summaries: Dict[str, AgentTokenSummary] = {}
        self._total_cost: float = 0.0

    # -- public API -----------------------------------------------------------

    def create_callback(
        self, agent_name: str, model: str = ""
    ) -> TokenUsageCallbackHandler:
        """Create a fresh callback handler for a single LLM invocation."""
        return TokenUsageCallbackHandler(agent_name=agent_name, model=model)

    def record(self, agent_name: str, usage: TokenUsage) -> None:
        """Record token usage from a completed LLM call."""
        if not self.enabled:
            return

        usage.estimated_cost = self._estimate_cost(usage)

        with self._lock:
            self._usage_log.append(usage)
            summary = self._agent_summaries.setdefault(
                agent_name, AgentTokenSummary(agent_name=agent_name)
            )
            summary.total_calls += 1
            summary.total_input_tokens += usage.input_tokens
            summary.total_output_tokens += usage.output_tokens
            summary.total_reasoning_tokens += usage.reasoning_tokens
            summary.total_tokens += usage.total_tokens
            summary.total_cost += usage.estimated_cost
            self._total_cost += usage.estimated_cost

        logger.debug(
            "Recorded %d tokens ($%.6f) for agent '%s' (model=%s)",
            usage.total_tokens,
            usage.estimated_cost,
            agent_name,
            usage.model,
        )

    def check_budgets(self, agent_name: str) -> None:
        """Check budget limits and raise ``BudgetExceededError`` if exceeded.

        Also emits a warning event when the alert threshold is crossed.
        """
        if not self.enabled:
            return

        with self._lock:
            summary = self._agent_summaries.get(agent_name)

        # Per-task token limit
        if self.config.max_tokens_per_task is not None and summary is not None:
            self._check_token_limit(agent_name, summary)

        # Per-run cost limit
        if self.config.max_cost_per_run is not None:
            self._check_cost_limit()

    def get_agent_summary(self, agent_name: str) -> AgentTokenSummary:
        """Return aggregated summary for *agent_name*."""
        with self._lock:
            return self._agent_summaries.get(
                agent_name, AgentTokenSummary(agent_name=agent_name)
            )

    def get_run_summary(self) -> Dict[str, Any]:
        """Return full run summary with per-agent breakdown."""
        with self._lock:
            agents = []
            total_tokens = 0
            for s in self._agent_summaries.values():
                agents.append(
                    {
                        "agent_name": s.agent_name,
                        "total_calls": s.total_calls,
                        "total_input_tokens": s.total_input_tokens,
                        "total_output_tokens": s.total_output_tokens,
                        "total_reasoning_tokens": s.total_reasoning_tokens,
                        "total_tokens": s.total_tokens,
                        "total_cost": s.total_cost,
                    }
                )
                total_tokens += s.total_tokens

            return {
                "total_tokens": total_tokens,
                "total_cost": self._total_cost,
                "total_calls": len(self._usage_log),
                "agents": agents,
            }

    def reset(self) -> None:
        """Clear all recorded usage data."""
        with self._lock:
            self._usage_log.clear()
            self._agent_summaries.clear()
            self._total_cost = 0.0

    # -- internals ------------------------------------------------------------

    def _estimate_cost(self, usage: TokenUsage) -> float:
        pricing = self._resolve_pricing(usage.model)
        if pricing is None:
            return 0.0
        cost = (
            usage.input_tokens * pricing.get("input", 0.0)
            + usage.output_tokens * pricing.get("output", 0.0)
            + usage.reasoning_tokens * pricing.get("reasoning", 0.0)
        ) / 1_000_000
        return cost

    def _resolve_pricing(self, model: str) -> Optional[Dict[str, float]]:
        if self.config.pricing and model in self.config.pricing:
            return self.config.pricing[model]
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]
        logger.warning("No pricing data for model '%s'; cost will be $0.00", model)
        return None

    def _check_token_limit(self, agent_name: str, summary: AgentTokenSummary) -> None:
        from .exceptions import BudgetExceededError

        limit = self.config.max_tokens_per_task
        if limit is None:
            return

        pct = summary.total_tokens / limit if limit else 0.0

        if pct >= self.config.alert_threshold_pct and pct < 1.0:
            self._emit_budget_warning(
                agent_name, pct, "tokens", float(summary.total_tokens), float(limit)
            )

        if summary.total_tokens > limit:
            raise BudgetExceededError(
                f"Agent '{agent_name}' exceeded token budget: "
                f"{summary.total_tokens} > {limit}",
                agent_name=agent_name,
                current_usage=float(summary.total_tokens),
                budget_limit=float(limit),
                budget_type="tokens",
            )

    def _check_cost_limit(self) -> None:
        from .exceptions import BudgetExceededError

        limit = self.config.max_cost_per_run
        if limit is None:
            return

        with self._lock:
            current = self._total_cost

        pct = current / limit if limit else 0.0

        if pct >= self.config.alert_threshold_pct and pct < 1.0:
            self._emit_budget_warning("run", pct, "cost", current, limit)

        if current > limit:
            raise BudgetExceededError(
                f"Run cost budget exceeded: ${current:.6f} > ${limit:.6f}",
                agent_name="run",
                current_usage=current,
                budget_limit=limit,
                budget_type="cost",
            )

    @staticmethod
    def _emit_budget_warning(
        agent_name: str,
        usage_pct: float,
        budget_type: str,
        current: float,
        limit: float,
    ) -> None:
        try:
            from ..cli.events import emit_budget_warning

            emit_budget_warning(agent_name, usage_pct, budget_type, current, limit)
        except ImportError:
            pass
        logger.warning(
            "Budget warning for '%s': %.0f%% of %s limit (%.2f / %.2f)",
            agent_name,
            usage_pct * 100,
            budget_type,
            current,
            limit,
        )
