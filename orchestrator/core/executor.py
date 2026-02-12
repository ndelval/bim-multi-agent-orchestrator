"""
Workflow execution logic for the Orchestrator.

This module handles the execution of workflows using LangGraph,
following the Single Responsibility Principle by separating
execution concerns from initialization and lifecycle management.
"""

import asyncio
import logging
from typing import Any, Optional, Sequence, Dict, List
from textwrap import dedent

from ..integrations.langchain_integration import (
    OrchestratorState,
    HumanMessage,
)
from ..memory.memory_manager import MemoryManager
from .config import OrchestratorConfig, AgentConfig, TaskConfig
from .exceptions import WorkflowError, AgentCreationError

logger = logging.getLogger(__name__)


class OrchestratorExecutor:
    """
    Handles execution of orchestrator workflows.

    This class is responsible for running workflows using LangGraph,
    managing memory recall, and coordinating workflow execution.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        compiled_graph: Any,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize the orchestrator executor.

        Args:
            config: Orchestrator configuration
            compiled_graph: Compiled LangGraph StateGraph
            memory_manager: Optional memory manager for recall
        """
        self.config = config
        self.compiled_graph = compiled_graph
        self.memory_manager = memory_manager

    async def run_langgraph_workflow(self, recall_content: Optional[str] = None) -> Any:
        """
        Run the LangGraph StateGraph workflow.

        Args:
            recall_content: Optional memory recall content to include in execution

        Returns:
            The result of the StateGraph execution

        Raises:
            WorkflowError: If workflow execution fails
        """
        try:
            # Determine the user prompt
            user_prompt = recall_content or "Execute orchestrator workflow"

            # Build initial state
            initial_state = OrchestratorState(
                messages=[HumanMessage(content=user_prompt)],
                input_prompt=user_prompt,
                memory_context=recall_content,
                max_iterations=self.config.max_iterations,
                recall_items=recall_content.split("\n") if recall_content else [],
            )

            # Execute the StateGraph
            logger.info("Executing LangGraph StateGraph workflow")
            result = await asyncio.to_thread(self.compiled_graph.invoke, initial_state)

            # Extract final output from state
            return self._extract_final_output(result)

        except Exception as e:
            logger.error(f"LangGraph workflow execution failed: {str(e)}")
            raise WorkflowError(f"LangGraph execution failed: {str(e)}")

    def _extract_final_output(self, result: Any) -> Any:
        """
        Extract final output from workflow result.

        Args:
            result: Workflow execution result

        Returns:
            Final output content
        """
        # Check for final_output attribute
        if hasattr(result, "final_output") and result.final_output:
            return result.final_output

        # Check messages
        if hasattr(result, "messages") and result.messages:
            # Return the last AI message
            for message in reversed(result.messages):
                if hasattr(message, "content") and "AI" in str(type(message)):
                    return message.content

        # Fallback - return the entire state as string
        return str(result)

    def build_recall_content(self) -> Optional[str]:
        """
        Build a global recall context string from memory based on custom_config.

        Expects custom_config.recall like:
          {
            "query": "preferencias|contexto...",
            "limit": 5,
            "agent_id": "mecanico",
            "run_id": "proyecto_A",
            "user_id": "override_user",  # defaults to config.user_id
            "rerank": true
          }

        Returns:
            Formatted recall context string or None
        """
        if not self.memory_manager:
            return None

        recall_cfg = (self.config.custom_config or {}).get("recall")
        if not recall_cfg:
            return None

        query = recall_cfg.get("query")
        if not query:
            return None

        limit = recall_cfg.get("limit", recall_cfg.get("top_k", 5))
        user_id = recall_cfg.get("user_id", self.config.user_id)
        agent_id = recall_cfg.get("agent_id")
        run_id = recall_cfg.get("run_id")
        rerank = recall_cfg.get("rerank")

        try:
            # Try provider's filtered retrieval if available
            results = self.memory_manager.retrieve_filtered(
                query,
                limit=limit,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                rerank=rerank,
            )
        except Exception as e:
            # Fallback to simple retrieval if filtered retrieval fails
            logger.warning(
                f"Filtered memory retrieval failed, falling back to simple retrieval: {e}"
            )
            results = self.memory_manager.retrieve(query, limit=limit)

        if not results:
            return None

        # Format a compact context string with citations
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

    def plan_from_prompt(
        self,
        prompt: str,
        agent_sequence: Sequence[str],
        enabled_agents: Dict[str, AgentConfig],
        *,
        recall_snippets: Optional[Sequence[str]] = None,
        assignments: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[TaskConfig]:
        """
        Generate task plan dynamically from prompt and selected agents.

        Args:
            prompt: User prompt
            agent_sequence: Sequence of agent names to use
            enabled_agents: Dictionary of available agents
            recall_snippets: Optional memory recall snippets
            assignments: Optional task assignments

        Returns:
            List of TaskConfig instances

        Raises:
            AgentCreationError: If agents are not available
        """
        if not agent_sequence:
            raise ValueError("agent_sequence must contain at least one agent")

        missing_agents = [name for name in agent_sequence if name not in enabled_agents]
        if missing_agents:
            raise AgentCreationError(
                f"Agents not available in orchestrator configuration: {missing_agents}"
            )

        # Build new task configurations
        dynamic_tasks: List[TaskConfig] = []
        previous_task_name: Optional[str] = None
        assignment_iter = list(assignments or [])

        for index, agent_name in enumerate(agent_sequence):
            agent_cfg = enabled_agents[agent_name]
            task_name = self._generate_task_name(agent_name, index)
            assignment_payload = (
                assignment_iter[index] if index < len(assignment_iter) else {}
            )

            objective = assignment_payload.get("objective") or assignment_payload.get(
                "description"
            )
            deliverable = assignment_payload.get(
                "expected_output"
            ) or assignment_payload.get("deliverable")
            tags = assignment_payload.get("tags")

            description = self._compose_task_description(
                agent_cfg,
                prompt,
                recall_snippets,
                task_hint=self._task_type_hint(agent_name),
                assignment_objective=objective,
                assignment_tags=tags,
            )
            expected_output = self._compose_expected_output(
                agent_cfg, prompt, deliverable=deliverable
            )

            dynamic_tasks.append(
                TaskConfig(
                    name=task_name,
                    description=description,
                    expected_output=expected_output,
                    agent=agent_name,
                    async_execution=False,
                    is_start=index == 0,
                    context=[previous_task_name] if previous_task_name else [],
                )
            )
            previous_task_name = task_name

        return dynamic_tasks

    @staticmethod
    def _generate_task_name(agent_name: str, index: int) -> str:
        """Generate task name from agent name and index."""
        slug = agent_name.lower().replace(" ", "_")
        return f"{slug}_task_{index + 1}"

    @staticmethod
    def _compose_task_description(
        agent_cfg: AgentConfig,
        prompt: str,
        recall_snippets: Optional[Sequence[str]] = None,
        *,
        task_hint: Optional[str] = None,
        assignment_objective: Optional[str] = None,
        assignment_tags: Optional[Sequence[str]] = None,
    ) -> str:
        """Compose task description from agent config and prompt."""
        recall_block = ""
        if recall_snippets:
            formatted = "\n".join(
                f"  - {snippet}" for snippet in recall_snippets if snippet
            )
            if formatted:
                recall_block = f"\nRecalled context:\n{formatted}"

        hint_block = f"\nSuggested task type: {task_hint}" if task_hint else ""
        objective_block = (
            f"\nSpecific objective: {assignment_objective}"
            if assignment_objective
            else ""
        )
        tags_block = ""
        if assignment_tags:
            tag_str = ", ".join(str(tag) for tag in assignment_tags if tag)
            if tag_str:
                tags_block = f"\nTags: {tag_str}"

        return (
            dedent(
                f"""
                Agent role: {agent_cfg.role}
                Base goal: {agent_cfg.goal}{hint_block}{objective_block}{tags_block}

                Current prompt:
                {prompt}
                """
            ).strip()
            + recall_block
            + "\n\nFollow your base instructions and deliver a concrete, actionable result."
        )

    @staticmethod
    def _compose_expected_output(
        agent_cfg: AgentConfig, prompt: str, *, deliverable: Optional[str] = None
    ) -> str:
        """Compose expected output from agent config and prompt."""
        goal = agent_cfg.goal or "Produce a deliverable"
        if deliverable:
            return f"{deliverable} (base goal: {goal}). Context: {prompt}"
        return f"{goal}. Respond to the prompt: {prompt}"

    @staticmethod
    def _task_type_hint(agent_name: str) -> Optional[str]:
        """Get task type hint for agent name."""
        hints = {
            "Researcher": "research",
            "Analyst": "analysis",
            "Planner": "planning",
            "StandardsAgent": "review",
            "QuickResponder": "documentation",
        }
        return hints.get(agent_name)
