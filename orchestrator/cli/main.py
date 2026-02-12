#!/usr/bin/env python3
"""
Enhanced CLI for StateGraph-based orchestrator with real-time Rich display.
Provides chat and info commands with comprehensive memory integration.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.config import OrchestratorConfig, AgentConfig, TaskConfig

# Updated import path for new memory manager module structure
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.memory.providers.registry import MemoryProviderRegistry
from orchestrator.factories.agent_factory import AgentFactory
from orchestrator.factories.task_factory import TaskFactory

# Web search tools - initialized lazily on first use
_web_tools_initialized = False
_duckduckgo_tool = None
_wikipedia_tool = None


def _initialize_web_tools():
    """Initialize web search tools from langchain_community. Called once on first use."""
    global _web_tools_initialized, _duckduckgo_tool, _wikipedia_tool
    if _web_tools_initialized:
        return
    _web_tools_initialized = True

    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        _duckduckgo_tool = DuckDuckGoSearchResults(max_results=5)
        logger.info("DuckDuckGo search tool initialized")
    except ImportError:
        logger.warning(
            "langchain-community not installed or DuckDuckGoSearchResults unavailable; "
            "Researcher agent will run without web search"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize DuckDuckGo tool: {e}")

    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        _wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2)
        )
        logger.info("Wikipedia tool initialized")
    except ImportError:
        logger.warning(
            "wikipedia package not installed; Researcher agent will run without Wikipedia"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Wikipedia tool: {e}")


# Import Rich display event system
from orchestrator.cli.events import (
    emit_router_decision,
    emit_agent_start,
    emit_agent_progress,
    emit_agent_complete,
    emit_final_answer,
)
from orchestrator.cli.rich_display import RichWorkflowDisplay


# Import GraphAdapter for both backends
from orchestrator.cli.graph_adapter import GraphAgentAdapter


logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def _show_memory_info(args: argparse.Namespace) -> int:
    """Display memory configuration information."""
    console = Console()
    provider_name = args.memory_provider.lower()

    # Get registered provider class
    provider_class = MemoryProviderRegistry.get_provider(provider_name)
    if not provider_class:
        console.print(f"[red]Unknown memory provider: {provider_name}[/red]")
        return 1

    # Display configuration details
    config_panel = Panel(
        f"[bold cyan]Memory Provider:[/bold cyan] {provider_name}\n"
        f"[bold cyan]Provider Class:[/bold cyan] {provider_class.__name__}\n"
        f"[bold cyan]Module:[/bold cyan] {provider_class.__module__}",
        title="Memory Configuration",
        border_style="green",
    )
    console.print(config_panel)

    # Initialize provider to show runtime status
    try:
        from orchestrator.core.config import (
            MemoryConfig,
            MemoryProvider,
            EmbedderConfig,
        )

        # Create proper config for provider
        memory_config = MemoryConfig(
            provider=MemoryProvider(provider_name), embedder=EmbedderConfig()
        )

        # Create manager and check health
        manager = MemoryManager(config=memory_config)

        if manager.health_check():
            status_text = "[green]✓ Provider initialized successfully[/green]"

            # Get provider info
            provider_info = manager.get_provider_info()
            status_text += (
                f"\n[cyan]Provider:[/cyan] {provider_info.get('provider', 'Unknown')}"
            )
            status_text += f"\n[cyan]Status:[/cyan] {'Healthy' if provider_info.get('initialized') else 'Unhealthy'}"
        else:
            status_text = (
                "[yellow]⚠ Provider initialized but health check failed[/yellow]"
            )

    except Exception as e:
        status_text = f"[red]✗ Provider initialization failed:[/red] {str(e)}"
        logger.exception("Provider initialization failed in memory-info")

    console.print(Panel(status_text, title="Provider Status", border_style="blue"))
    return 0


def _get_agent_template(agent_name: str) -> AgentConfig:
    """Get predefined agent configuration template.

    Delegates to orchestrator.factories.agent_templates.
    """
    from orchestrator.factories.agent_templates import get_agent_template

    return get_agent_template(agent_name)


def _build_chat_agents(memory_manager: MemoryManager, use_tools: bool = True) -> tuple:
    """Build standard chat agents with GraphRAG tool."""
    from orchestrator.tools.graph_rag_tool import create_graph_rag_tool

    # Create GraphRAG tool
    graphrag_tool = create_graph_rag_tool(memory_manager)

    # Web search tools (optional, lazy-initialized)
    web_tools = []
    if use_tools:
        _initialize_web_tools()
        if _duckduckgo_tool:
            web_tools.append(_duckduckgo_tool)
        if _wikipedia_tool:
            web_tools.append(_wikipedia_tool)

    # Get agent templates
    router_config = _get_agent_template("router")
    researcher_config = _get_agent_template("researcher")
    analyst_config = _get_agent_template("analyst")
    planner_config = _get_agent_template("planner")
    standards_config = _get_agent_template("standards")

    # Attach tools to agents
    router_config.tools = [graphrag_tool]
    researcher_config.tools = [graphrag_tool] + web_tools
    analyst_config.tools = [graphrag_tool]
    planner_config.tools = [graphrag_tool]
    standards_config.tools = [graphrag_tool]

    return (
        router_config,
        researcher_config,
        analyst_config,
        planner_config,
        standards_config,
    )


def _parse_router_payload(text: str) -> Dict[str, Any]:
    """Extract decision payload from router output.

    Delegates to orchestrator.utils.output_extraction.
    """
    from orchestrator.utils.output_extraction import parse_router_payload

    return parse_router_payload(text)


def _extract_text(output: Any) -> str:
    """Extract text content from various output formats.

    Delegates to orchestrator.utils.output_extraction.
    """
    from orchestrator.utils.output_extraction import extract_text

    return extract_text(output)


def _extract_decision(output: Any) -> Optional[str]:
    """Extract routing decision from various output formats.

    Delegates to orchestrator.utils.output_extraction.
    """
    from orchestrator.utils.output_extraction import extract_decision

    return extract_decision(output)


# Graph tool attachment now handled by CLI adapter


def run_chat(args: argparse.Namespace) -> int:
    """Run interactive chat with the orchestrator."""
    from .chat_orchestrator import ChatOrchestrator

    orchestrator = ChatOrchestrator(args)
    return orchestrator.run()


def main() -> int:
    """Main entry point for CLI."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--memory-provider",
        type=str,
        default=os.getenv("ORCH_MEMORY_PROVIDER", "hybrid"),
        choices=["hybrid", "mem0", "rag"],
        help="Memory provider to use (default: hybrid)",
    )
    chat_parser.add_argument(
        "--backend",
        type=str,
        default="langgraph",
        choices=["langgraph"],
        help="Agent backend to use (default: langgraph)",
    )
    chat_parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    chat_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    chat_parser.add_argument(
        "--user-id",
        type=str,
        default=os.getenv("ORCH_USER_ID", "default_user"),
        help="User identifier for session tracking (default: default_user)",
    )

    # Memory info command
    info_parser = subparsers.add_parser("memory-info", help="Show memory configuration")
    info_parser.add_argument(
        "--memory-provider",
        type=str,
        default=os.getenv("ORCH_MEMORY_PROVIDER", "hybrid"),
        choices=["hybrid", "mem0", "rag"],
        help="Memory provider to inspect (default: hybrid)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "chat":
        return run_chat(args)
    elif args.command == "memory-info":
        return _show_memory_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
