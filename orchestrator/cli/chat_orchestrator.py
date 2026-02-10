"""
Chat orchestration for interactive multi-agent sessions.

This module provides the ChatOrchestrator class that manages interactive
chat sessions with router-based agent orchestration.
"""

import argparse
import logging
import os
from typing import Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..core.config import (
    AgentConfig,
    EmbedderConfig,
    MemoryConfig,
    MemoryProvider,
)
from ..core.value_objects import RouterDecision
from ..core.error_handler import ErrorHandler
from ..memory.memory_manager import MemoryManager
from ..session.session_manager import SessionManager
from .display_adapter import DisplayAdapter, create_display_adapter
from .graph_adapter import GraphAgentAdapter

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Manages interactive chat sessions with multi-agent orchestration.

    This class handles the complete lifecycle of a chat session including:
    - Memory and agent initialization
    - Router-based decision making
    - Multi-path execution (quick, analysis, planning)
    - Result display and error handling
    """

    def __init__(self, args: argparse.Namespace, display_adapter: Optional[DisplayAdapter] = None):
        """
        Initialize chat orchestrator with configuration.

        Args:
            args: Command-line arguments with memory_provider, backend, llm, verbose, user_id
            display_adapter: Optional display adapter (defaults to RichDisplayAdapter)
        """
        self.args = args
        self.console: Optional[Console] = None
        self.display: DisplayAdapter = display_adapter or create_display_adapter("rich")
        self.error_handler = ErrorHandler(logger)
        self.memory_manager: Optional[MemoryManager] = None
        self.session_manager: Optional[SessionManager] = None
        self.adapter: Optional[GraphAgentAdapter] = None

        # Agent configurations
        self.router_config: Optional[AgentConfig] = None
        self.researcher_config: Optional[AgentConfig] = None
        self.analyst_config: Optional[AgentConfig] = None
        self.planner_config: Optional[AgentConfig] = None
        self.standards_config: Optional[AgentConfig] = None

    def _setup_console(self) -> None:
        """Setup console and display components."""
        self.console = Console()

        # Welcome message
        self.console.print(
            Panel(
                "[bold cyan]Multi-Agent Orchestrator[/bold cyan]\n"
                f"Memory Provider: {self.args.memory_provider}\n"
                f"Backend: {self.args.backend}\n"
                "Type 'exit' or 'quit' to end the session.",
                title="Welcome",
                border_style="green",
            )
        )

    def _initialize_memory(self) -> bool:
        """
        Initialize memory manager from configuration.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create memory configuration with proper config object pattern
            embedder_provider = os.getenv("HYBRID_EMBEDDER_PROVIDER", "openai")
            embedder_model = os.getenv("HYBRID_EMBEDDER_MODEL", "text-embedding-3-small")
            memory_config = MemoryConfig(
                provider=MemoryProvider(self.args.memory_provider.lower()),
                use_embedding=True,
                embedder=EmbedderConfig(
                    provider=embedder_provider,
                    config={"model": embedder_model}
                ),
                config={},
            )

            self.memory_manager = MemoryManager(config=memory_config)
            self.console.print("[green]✓ Memory provider initialized[/green]")
            return True

        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="memory_initialization",
                component="memory_manager"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return False

    def _initialize_session(self) -> bool:
        """
        Initialize session manager and create new session.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize session manager
            self.session_manager = SessionManager()

            # Get user_id from args or use default
            user_id = getattr(self.args, 'user_id', 'default_user')

            # Create new session with metadata
            session = self.session_manager.create_session(
                user_id=user_id,
                metadata={
                    "source": "cli",
                    "backend": self.args.backend,
                    "memory_provider": self.args.memory_provider,
                }
            )

            self.console.print(
                f"[green]✓ Session created: {session.session_id[:8]}... "
                f"(user: {user_id}, session #{session.turn_count + 1})[/green]"
            )
            return True

        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="session_initialization",
                component="session_manager"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return False

    def _build_agents(self) -> bool:
        """
        Build agent configurations for chat session.

        Returns:
            True if successful, False otherwise
        """
        try:
            from .main import _build_chat_agents

            (
                self.router_config,
                self.researcher_config,
                self.analyst_config,
                self.planner_config,
                self.standards_config,
            ) = _build_chat_agents(self.memory_manager, use_tools=True)

            self.console.print("[green]✓ Agent configurations ready[/green]\n")
            return True

        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="agent_building",
                component="agent_factory"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return False

    def _setup_backend_adapter(self) -> bool:
        """
        Setup backend adapter for agent execution.

        Returns:
            True if successful, False otherwise
        """
        if self.args.backend == "langgraph":
            self.adapter = GraphAgentAdapter(
                memory_manager=self.memory_manager,
                llm=self.args.llm,
                enable_parallel=True,
            )
            self.console.print("[cyan]Using LangGraph backend with parallel execution[/cyan]\n")
            return True
        else:
            self.console.print(f"[red]Unknown backend: {self.args.backend}[/red]")
            return False

    def _handle_router_phase(self, user_query: str) -> Optional[RouterDecision]:
        """
        Execute router and extract decision.

        Args:
            user_query: User input query

        Returns:
            RouterDecision object with route, confidence, reasoning, and metadata,
            or None if router execution failed
        """
        from .main import _extract_decision, _extract_text, _parse_router_payload

        try:
            router_result = self.adapter.run_single_agent(
                agent_config=self.router_config,
                user_query=user_query,
            )
        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="router_execution",
                component="router"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return None

        # Extract decision from router result
        router_raw_text = _extract_text(router_result)
        payload = _parse_router_payload(router_raw_text)

        decision = payload.get("decision")
        if not decision:
            decision = _extract_decision(router_result)

        # Replace silent fallback with logged fallback
        if not decision:
            logger.error(
                "Router decision extraction failed - no decision found in result"
            )
            logger.debug(
                f"Router result type: {type(router_result)}, content: {router_result}"
            )
            decision = "quick"  # Explicit fallback with logging
        else:
            decision = decision.lower()

        # Extract metadata from payload
        confidence = payload.get("confidence", "Medium")
        reasoning = payload.get("rationale", "Router analysis completed")
        latency = payload.get("latency", 0.0)
        tokens = payload.get("tokens", 0)
        assigned_agents = payload.get("assigned_agents", [])

        # Create RouterDecision value object
        router_decision = RouterDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            latency=latency,
            assigned_agents=assigned_agents,
            tokens=tokens,
            payload=payload
        )

        # Show router decision via display adapter
        self.display.show_router_decision(
            decision=router_decision.decision,
            confidence=router_decision.confidence,
            rationale=router_decision.reasoning,
            latency=router_decision.latency,
            tokens=router_decision.tokens
        )

        return router_decision

    def _handle_execution_phase(self, decision: str, user_query: str) -> Optional[str]:
        """
        Execute appropriate workflow based on router decision.

        Args:
            decision: Router decision (quick, analysis, research, planning)
            user_query: User input query

        Returns:
            Final answer text or None if execution failed
        """
        from .main import _extract_text

        final_answer = None

        if decision == "quick":
            # Quick path: Single researcher agent
            final_answer = self._execute_quick_path(user_query)

        elif decision in ("analysis", "research"):
            # Analysis path: Multi-agent workflow
            final_answer = self._execute_analysis_path(user_query)

        elif decision == "planning":
            # Planning path: ToT planner + multi-agent
            final_answer = self._execute_planning_path(user_query)

        else:
            self.console.print(f"[red]✗ Unknown decision: {decision}[/red]")

        return final_answer

    def _execute_quick_path(self, user_query: str) -> Optional[str]:
        """Execute quick path with single researcher agent."""
        from .main import _extract_text

        self.display.show_agent_start("Researcher", "Gathering information")

        try:
            researcher_result = self.adapter.run_single_agent(
                agent_config=self.researcher_config,
                user_query=user_query,
            )
            final_answer = _extract_text(researcher_result)

            self.display.show_agent_complete("Researcher", "Research completed")

            return final_answer

        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="researcher_execution",
                component="researcher_agent"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return None

    def _execute_analysis_path(self, user_query: str) -> Optional[str]:
        """Execute analysis path with multi-agent workflow."""
        from .main import _extract_text

        agent_sequence = ["Researcher", "Analyst", "StandardsAgent"]

        try:
            workflow_result = self.adapter.run_multi_agent_workflow(
                agent_sequence=agent_sequence,
                user_query=user_query,
                display_adapter=self.display,
            )
            return _extract_text(workflow_result)

        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="analysis_workflow",
                component="multi_agent"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return None

    def _execute_planning_path(self, user_query: str) -> Optional[str]:
        """Execute planning path with ToT planner and multi-agent workflow."""
        from .main import _extract_text

        self.console.print(
            "[yellow]⚠ Planning path with ToT is not yet implemented[/yellow]"
        )

        # Fallback to analysis path
        agent_sequence = ["Researcher", "Analyst", "Planner", "StandardsAgent"]

        try:
            workflow_result = self.adapter.run_multi_agent_workflow(
                agent_sequence=agent_sequence,
                user_query=user_query,
                display_adapter=self.display,
            )
            return _extract_text(workflow_result)

        except Exception as e:
            resolution = self.error_handler.handle_error(
                exception=e,
                operation="planning_workflow",
                component="multi_agent"
            )
            self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
            return None

    def _display_result(self, final_answer: Optional[str], router_decision: RouterDecision) -> None:
        """
        Display final answer to user.

        Args:
            final_answer: Final answer text to display
            router_decision: Router decision object for context
        """
        if final_answer:
            self.display.show_final_answer(final_answer)
        else:
            # Log debug info when no answer is generated
            logger.warning("No final answer generated - workflow may have failed")
            logger.debug(f"Router decision was: {router_decision.decision}")
            logger.debug(f"Router reasoning: {router_decision.reasoning}")
            self.console.print("[yellow]⚠ No answer generated - check logs for details[/yellow]")

    def _store_conversation_turn(
        self,
        user_query: str,
        final_answer: str,
        router_decision: RouterDecision
    ) -> None:
        """
        Store conversation turn in memory for future recall.

        This method persists both the user query and assistant response to the
        memory system (hybrid provider with vector, lexical, and graph storage).
        Uses proper session tracking from SessionManager.

        Args:
            user_query: The user's input query
            final_answer: The assistant's response
            router_decision: Router decision object with metadata
        """
        if not self.memory_manager:
            logger.debug("Memory manager not available - skipping conversation storage")
            return

        if not final_answer:
            logger.debug("No final answer to store - skipping conversation storage")
            return

        # Record conversation turn in session
        if self.session_manager and self.session_manager.current_session:
            try:
                self.session_manager.record_turn()
            except Exception as e:
                logger.warning(f"Failed to record session turn: {e}")

        try:
            from ..memory.document_schema import current_timestamp

            # Get session context for proper tracking
            user_id = "default_user"
            session_id = "chat_session"

            if self.session_manager and self.session_manager.current_session:
                user_id = self.session_manager.current_session.user_id
                session_id = self.session_manager.current_session.session_id

            # Create base metadata for this conversation turn
            base_metadata = {
                "content_type": "conversation",
                "user_id": user_id,
                "agent_id": "chat_orchestrator",
                "run_id": session_id,
                "timestamp": current_timestamp(),
                "decision": router_decision.decision,
                "confidence": router_decision.confidence,
                "reasoning": router_decision.reasoning,
            }

            # Store user query
            user_metadata = {**base_metadata, "speaker": "user"}
            user_doc_id = self.memory_manager.store(
                content=f"User: {user_query}",
                metadata=user_metadata
            )
            logger.debug(f"Stored user query in memory: {user_doc_id}")

            # Store assistant response
            assistant_metadata = {**base_metadata, "speaker": "assistant"}
            assistant_doc_id = self.memory_manager.store(
                content=f"Assistant: {final_answer}",
                metadata=assistant_metadata
            )
            logger.debug(f"Stored assistant response in memory: {assistant_doc_id}")

            # Store combined conversation context for better retrieval
            combined_metadata = {**base_metadata, "speaker": "conversation_pair"}
            combined_content = f"""Conversation Turn:
User Query: {user_query}
Assistant Response: {final_answer}
Routing Decision: {router_decision.decision}
Confidence: {router_decision.confidence}"""

            combined_doc_id = self.memory_manager.store(
                content=combined_content,
                metadata=combined_metadata
            )
            logger.info(f"Conversation turn stored in memory: {combined_doc_id}")

        except Exception as e:
            # Don't crash the chat loop on storage failures
            logger.error(f"Failed to store conversation in memory: {e}", exc_info=True)
            # Optionally inform user of storage failure
            if self.args.verbose:
                self.console.print(f"[yellow]⚠ Warning: Conversation not saved to memory[/yellow]")

    def run(self) -> int:
        """
        Run interactive chat loop.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        from .main import _setup_logging

        # Setup logging
        _setup_logging(self.args.verbose)

        # Initialize components
        self._setup_console()

        if not self._initialize_memory():
            return 1

        if not self._initialize_session():
            return 1

        if not self._build_agents():
            return 1

        if not self._setup_backend_adapter():
            return 1

        # Chat loop
        while True:
            try:
                # Get user input
                user_query = self.console.input("\n[bold blue]You:[/bold blue] ").strip()

                if not user_query:
                    continue

                if user_query.lower() in ["exit", "quit"]:
                    # End session properly
                    if self.session_manager:
                        session_info = self.session_manager.get_session_info()
                        self.session_manager.end_session()
                        self.console.print(
                            f"[yellow]Goodbye! Session ended "
                            f"({session_info.get('turn_count', 0)} turns)[/yellow]"
                        )
                    else:
                        self.console.print("[yellow]Goodbye![/yellow]")
                    break

                # Clear previous workflow display
                self.display.clear()

                # Router phase
                router_decision = self._handle_router_phase(user_query)
                if not router_decision:
                    continue

                # Execution phase
                final_answer = self._handle_execution_phase(router_decision.decision, user_query)

                # Display result
                self._display_result(final_answer, router_decision)

                # Store conversation in memory for future recall
                self._store_conversation_turn(user_query, final_answer, router_decision)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted by user[/yellow]")
                # End session on interrupt
                if self.session_manager:
                    self.session_manager.end_session()
                break
            except Exception as e:
                resolution = self.error_handler.handle_error(
                    exception=e,
                    operation="chat_loop",
                    component="orchestrator"
                )
                self.console.print(f"[red]✗ {resolution.recovery_hint}[/red]")
                continue

        # Cleanup on exit
        if self.session_manager:
            self.session_manager.close()

        return 0


# Export main class
__all__ = ["ChatOrchestrator"]
