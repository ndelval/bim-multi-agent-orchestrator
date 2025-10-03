"""
Rich-based CLI display for beautiful orchestrator UI.

This module provides a comprehensive UI system using Rich library
to display agent execution, reasoning, and timeline visualization.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.text import Text
    from rich.columns import Columns
    from rich.align import Align

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .events import EventEmitter, EventType, ExecutionEvent, get_event_emitter

import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Track execution metrics for display."""

    # Timing
    start_time: float = field(default_factory=time.time)
    router_latency: float = 0.0
    planning_latency: float = 0.0
    execution_latency: float = 0.0
    total_duration: float = 0.0

    # Tokens
    router_tokens: int = 0
    planning_tokens: int = 0
    execution_tokens: int = 0
    total_tokens: int = 0

    # Quality
    router_confidence: str = "Unknown"
    plan_quality_score: float = 0.0
    memory_relevance: float = 0.0

    # Progress
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0

    # Status
    current_phase: str = "Initializing"
    current_agent: str = ""


class RichWorkflowDisplay:
    """
    Rich-based display for orchestrator execution.

    Provides beautiful, real-time visualization of:
    - Router decisions
    - ToT planning process
    - Agent execution timeline
    - Performance metrics
    - Error recovery
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize Rich display.

        Args:
            console: Optional Rich Console instance
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library not available. Install with: pip install rich"
            )

        self.console = console or Console()
        self.metrics = ExecutionMetrics()
        self.event_emitter = get_event_emitter()

        # Timeline tracking
        self.timeline_events: List[Dict[str, Any]] = []
        self.reasoning_tree: Optional[Tree] = None

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )

        # Task IDs for progress
        self.router_task: Optional[int] = None
        self.planning_task: Optional[int] = None
        self.execution_task: Optional[int] = None

        # Register event callback
        self.event_emitter.register_callback(self._handle_event)

    def _handle_event(self, event: ExecutionEvent) -> None:
        """
        Handle incoming execution events.

        Args:
            event: Execution event
        """
        # Add to timeline
        self.timeline_events.append(
            {
                "timestamp": event.timestamp,
                "type": event.event_type.value,
                "data": event.data,
            }
        )

        # Update metrics based on event type
        if event.event_type == EventType.WORKFLOW_START:
            self.metrics.start_time = event.timestamp
            self.metrics.current_phase = "Router Decision"

        elif event.event_type == EventType.ROUTER_DECISION:
            self.metrics.router_latency = event.data.get("latency", 0)
            self.metrics.router_tokens = event.data.get("tokens", 0)
            self.metrics.router_confidence = event.data.get("confidence", "Unknown")
            self.metrics.current_phase = "Planning"

        elif event.event_type == EventType.PLANNING_COMPLETE:
            self.metrics.planning_latency = event.data.get("duration", 0)
            self.metrics.plan_quality_score = event.data.get("quality_score", 0)
            self.metrics.total_nodes = event.data.get("assignments", 0)
            self.metrics.current_phase = "Agent Execution"

        elif event.event_type == EventType.NODE_START:
            self.metrics.current_agent = event.data.get("agent_name", "")

        elif event.event_type == EventType.NODE_COMPLETE:
            self.metrics.completed_nodes += 1
            self.metrics.execution_latency += event.data.get("duration", 0)

        elif event.event_type == EventType.WORKFLOW_COMPLETE:
            self.metrics.total_duration = event.data.get("duration", 0)
            self.metrics.current_phase = "Complete"

    def clear(self) -> None:
        """
        Clear display state for a new workflow execution.

        Resets all metrics, timeline events, and progress tracking
        to prepare for a new query in the chat loop.
        """
        # Reset metrics
        self.metrics = ExecutionMetrics()

        # Clear timeline
        self.timeline_events.clear()
        self.reasoning_tree = None

        # Reset progress task IDs
        self.router_task = None
        self.planning_task = None
        self.execution_task = None

    def show_header(self, user_prompt: str, backend: str = "LangGraph") -> None:
        """
        Show workflow header with user prompt.

        Args:
            user_prompt: User's input prompt
            backend: Backend type (LangGraph/PraisonAI)
        """
        header_text = Text()
        header_text.append("🤖 Multi-Agent Orchestrator CLI", style="bold cyan")
        header_text.append(f"  Backend: {backend} ✓", style="green")

        header_panel = Panel(
            Group(
                Align.center(header_text), "", Text(f"» {user_prompt}", style="yellow")
            ),
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(header_panel)
        self.console.print()

    def show_router_decision(
        self,
        decision: str,
        confidence: str,
        rationale: str,
        latency: float,
        tokens: int,
    ) -> None:
        """
        Show router decision panel.

        Args:
            decision: Decision type (quick/research/analysis/standards)
            confidence: Confidence level
            rationale: Decision rationale
            latency: Decision latency in seconds
            tokens: Tokens used
        """
        decision_table = Table(show_header=False, box=None, padding=(0, 1))
        decision_table.add_column("Attribute", style="cyan")
        decision_table.add_column("Value", style="white")

        decision_table.add_row("Decision", f"[bold green]{decision}[/bold green]")
        decision_table.add_row("Confidence", confidence)
        decision_table.add_row("Latency", f"{latency:.2f}s")
        decision_table.add_row("Tokens", str(tokens))
        decision_table.add_row(
            "Rationale", rationale[:80] + "..." if len(rationale) > 80 else rationale
        )

        panel = Panel(
            decision_table,
            title="🧠 [bold blue]Router Decision[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def show_tot_planning(
        self,
        method: str,
        thoughts_explored: int,
        quality_score: float,
        assignments: int,
        duration: float,
        tokens: int,
    ) -> None:
        """
        Show ToT planning panel.

        Args:
            method: Planning method (tree-of-thought/assignments)
            thoughts_explored: Number of thoughts explored
            quality_score: Plan quality score (0-10)
            assignments: Number of agent assignments
            duration: Planning duration
            tokens: Tokens used
        """
        planning_table = Table(show_header=False, box=None, padding=(0, 1))
        planning_table.add_column("Attribute", style="cyan")
        planning_table.add_column("Value", style="white")

        planning_table.add_row("Method", f"[bold]{method}[/bold]")
        planning_table.add_row("Thoughts Explored", str(thoughts_explored))
        planning_table.add_row(
            "Best Plan Score",
            (
                f"{quality_score:.1f}/10 ⭐"
                if quality_score >= 8
                else f"{quality_score:.1f}/10"
            ),
        )
        planning_table.add_row("Assignments", f"{assignments} agents")
        planning_table.add_row("Duration", f"{duration:.2f}s")
        planning_table.add_row("Tokens", str(tokens))

        panel = Panel(
            planning_table,
            title="🎯 [bold magenta]ToT Planning[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def create_reasoning_tree(self) -> Tree:
        """
        Create reasoning tree from timeline events.

        Returns:
            Rich Tree with reasoning hierarchy
        """
        tree = Tree(
            "🧠 [bold blue]Reasoning Chain[/bold blue]", guide_style="bold bright_blue"
        )

        # Group events by phase
        router_events = [e for e in self.timeline_events if "router" in e["type"]]
        planning_events = [
            e
            for e in self.timeline_events
            if "planning" in e["type"] or "tot" in e["type"]
        ]
        execution_events = [
            e
            for e in self.timeline_events
            if "node" in e["type"] or "tool" in e["type"]
        ]

        # Router branch
        if router_events:
            router_branch = tree.add(
                f"[cyan]Router Decision[/cyan] ({self.metrics.router_latency:.1f}s)"
            )
            for event in router_events[:3]:  # Show first 3
                data = event["data"]
                if "decision" in data:
                    router_branch.add(f"Decision: [green]{data['decision']}[/green]")
                if "confidence" in data:
                    router_branch.add(f"Confidence: {data['confidence']}")

        # Planning branch
        if planning_events:
            planning_branch = tree.add(
                f"[magenta]ToT Planning[/magenta] ({self.metrics.planning_latency:.1f}s)"
            )
            for event in planning_events:
                data = event["data"]
                if event["type"] == "tot_generation":
                    planning_branch.add(
                        f"Step {data.get('step', 0)}: Generated {data.get('candidates', 0)} thoughts"
                    )
                elif event["type"] == "tot_evaluation":
                    scores = data.get("scores", [])
                    if scores:
                        planning_branch.add(
                            f"Step {data.get('step', 0)}: Best score {max(scores):.1f}/10"
                        )
                elif event["type"] == "tot_selection":
                    planning_branch.add(
                        f"Step {data.get('step', 0)}: Kept {data.get('kept', 0)}, pruned {data.get('pruned', 0)}"
                    )

        # Execution branch
        if execution_events:
            exec_branch = tree.add(
                f"[yellow]Agent Execution[/yellow] ({self.metrics.execution_latency:.1f}s)"
            )

            # Group by node
            nodes: Dict[str, List[Dict]] = {}
            for event in execution_events:
                data = event["data"]
                node_name = data.get("node_name", data.get("agent_name", "unknown"))
                if node_name not in nodes:
                    nodes[node_name] = []
                nodes[node_name].append(event)

            # Show each node
            for node_name, node_events in nodes.items():
                node_start = next(
                    (e for e in node_events if e["type"] == "node_start"), None
                )
                node_complete = next(
                    (e for e in node_events if e["type"] == "node_complete"), None
                )

                if node_start and node_complete:
                    duration = node_complete["data"].get("duration", 0)
                    node_branch = exec_branch.add(
                        f"[bold]{node_name}[/bold] ({duration:.2f}s)"
                    )

                    # Show tool invocations
                    tool_events = [e for e in node_events if "tool" in e["type"]]
                    for tool_event in tool_events:
                        tool_data = tool_event["data"]
                        if tool_event["type"] == "tool_complete":
                            tool_branch = node_branch.add(
                                f"🔧 {tool_data.get('tool_name', 'tool')}"
                            )
                            tool_branch.add(
                                f"Results: {tool_data.get('results_count', 0)} items"
                            )

        return tree

    def show_timeline(self) -> None:
        """Show execution timeline."""
        if not self.timeline_events:
            return

        timeline_table = Table(
            title="📊 Execution Timeline",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
        )

        timeline_table.add_column("Time", style="cyan", width=8)
        timeline_table.add_column("Event", style="white", width=30)
        timeline_table.add_column("Details", style="dim", width=40)

        # Show last 10 events
        for event in self.timeline_events[-10:]:
            timestamp = event["timestamp"]
            elapsed = timestamp - self.metrics.start_time
            event_type = event["type"]
            data = event["data"]

            # Format event name
            event_name = event_type.replace("_", " ").title()

            # Format details
            details = []
            if "decision" in data:
                details.append(f"decision={data['decision']}")
            if "agent_name" in data:
                details.append(f"agent={data['agent_name']}")
            if "tool_name" in data:
                details.append(f"tool={data['tool_name']}")
            if "duration" in data:
                details.append(f"duration={data['duration']:.2f}s")

            details_str = ", ".join(details[:2]) if details else ""

            timeline_table.add_row(f"{elapsed:.1f}s", event_name, details_str)

        self.console.print(timeline_table)
        self.console.print()

    def show_metrics_dashboard(self) -> None:
        """Show performance and quality metrics dashboard."""
        # Performance metrics
        perf_table = Table(
            title="Performance", show_header=False, box=None, padding=(0, 1)
        )
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="white")

        perf_table.add_row("Total Duration", f"{self.metrics.total_duration:.2f}s")
        perf_table.add_row("Router Latency", f"{self.metrics.router_latency:.2f}s")
        perf_table.add_row("Planning Latency", f"{self.metrics.planning_latency:.2f}s")
        perf_table.add_row(
            "Execution Latency", f"{self.metrics.execution_latency:.2f}s"
        )
        perf_table.add_row("Total Tokens", str(self.metrics.total_tokens))

        # Quality metrics
        quality_table = Table(
            title="Quality", show_header=False, box=None, padding=(0, 1)
        )
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="white")

        quality_table.add_row(
            "Plan Quality", f"{self.metrics.plan_quality_score:.1f}/10"
        )
        quality_table.add_row("Router Confidence", self.metrics.router_confidence)
        quality_table.add_row(
            "Success Rate",
            f"{self.metrics.completed_nodes}/{self.metrics.total_nodes} (100%)",
        )
        quality_table.add_row("Failed Nodes", str(self.metrics.failed_nodes))

        # Create layout
        columns = Columns(
            [
                Panel(
                    perf_table,
                    border_style="green",
                    title="⚡ [bold]Performance[/bold]",
                ),
                Panel(
                    quality_table, border_style="blue", title="✨ [bold]Quality[/bold]"
                ),
            ],
            equal=True,
            expand=True,
        )

        self.console.print(
            Panel(
                columns,
                title="📊 [bold cyan]Metrics Dashboard[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print()

    def show_final_output(self, output: str) -> None:
        """
        Show final assistant response.

        Args:
            output: Final output text
        """
        self.console.print(
            Panel(
                Text(output, style="white"),
                title="💬 [bold green]Assistant Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        self.console.print()

    def show_error(
        self, error_message: str, recovery_path: Optional[str] = None
    ) -> None:
        """
        Show error with recovery information.

        Args:
            error_message: Error message
            recovery_path: Optional recovery path description
        """
        error_panel = Panel(
            Group(
                Text(f"❌ Error: {error_message}", style="bold red"),
                "",
                (
                    Text(f"🔄 Recovery: {recovery_path}", style="yellow")
                    if recovery_path
                    else Text("")
                ),
            ),
            title="⚠️  [bold red]Error[/bold red]",
            border_style="red",
            padding=(1, 2),
        )

        self.console.print(error_panel)
        self.console.print()

    def show_complete_execution(self) -> None:
        """Show complete execution summary with all components."""
        # Show reasoning tree
        reasoning_tree = self.create_reasoning_tree()
        self.console.print(
            Panel(
                reasoning_tree,
                title="🧠 [bold blue]Reasoning Chain[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        self.console.print()

        # Show timeline
        self.show_timeline()

        # Show metrics
        self.show_metrics_dashboard()


def create_rich_display(
    console: Optional[Console] = None,
) -> Optional[RichWorkflowDisplay]:
    """
    Create Rich display if available.

    Args:
        console: Optional Rich Console instance

    Returns:
        RichWorkflowDisplay instance or None if Rich not available
    """
    if not RICH_AVAILABLE:
        logger.warning("Rich library not available. Using plain text display.")
        return None

    try:
        return RichWorkflowDisplay(console=console)
    except Exception as e:
        logger.error(f"Failed to create Rich display: {e}")
        return None
