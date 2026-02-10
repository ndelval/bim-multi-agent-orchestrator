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

# Tools are now provided by LangChain integration
duckduckgo_tool = None
wikipedia_tool = None

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

    Args:
        agent_name: Agent template name (case-insensitive)

    Returns:
        AgentConfig for the requested template

    Raises:
        ValueError: If agent_name is not a valid template
    """
    # Normalize to lowercase for case-insensitive lookup
    agent_name_lower = agent_name.lower()

    # Handle special case: StandardsAgent → standards
    if agent_name_lower == "standardsagent":
        agent_name_lower = "standards"

    templates = {
        "router": AgentConfig(
            name="Router",
            role="Query Analyzer & Decision Router",
            goal="Analyze user queries and route to appropriate execution path",
            backstory=(
                "You are an intelligent routing agent that analyzes user queries and determines "
                "the best execution strategy. You classify queries into: 'quick' for simple factual "
                "questions, 'analysis' for complex research requiring multi-agent collaboration, "
                "or 'planning' for strategic planning tasks requiring Tree-of-Thought reasoning."
            ),
            instructions=(
                "1. Analyze the user query complexity, intent, and required capabilities\n"
                "2. Use GraphRAG tool to search relevant context from memory\n"
                "3. Determine optimal routing decision:\n"
                "   - 'quick': Simple factual questions answerable with web search\n"
                "   - 'analysis': Complex research requiring multi-agent workflows\n"
                "   - 'planning': Strategic planning requiring ToT reasoning\n"
                "4. Provide confidence score (High/Medium/Low) and clear rationale\n"
                '5. Return decision as JSON: {"decision": "quick|analysis|planning", "confidence": "High|Medium|Low", "rationale": "explanation"}'
            ),
            tools=["graphrag"],
            llm="gpt-4o-mini",
        ),
        "researcher": AgentConfig(
            name="Researcher",
            role="Information Research Specialist",
            goal="Gather comprehensive information from web sources and memory",
            backstory=(
                "You are a meticulous research specialist skilled at gathering information from "
                "multiple sources. You combine web search with memory retrieval to provide "
                "comprehensive, well-sourced answers."
            ),
            instructions=(
                "1. Search GraphRAG memory for relevant historical context\n"
                "2. Use web search (DuckDuckGo/Wikipedia) for current information\n"
                "3. Synthesize findings into coherent response with sources\n"
                "4. Highlight key insights and data points"
            ),
            tools=["graphrag", "duckduckgo", "wikipedia"],
            llm="gpt-4o-mini",
        ),
        "analyst": AgentConfig(
            name="Analyst",
            role="Data Analysis & Insight Specialist",
            goal="Analyze research findings and extract actionable insights",
            backstory=(
                "You are an analytical expert who excels at processing information and "
                "identifying patterns, trends, and actionable insights."
            ),
            instructions=(
                "1. Review research findings from previous agents\n"
                "2. Query GraphRAG for relevant analytical context\n"
                "3. Identify patterns, trends, and key insights\n"
                "4. Provide structured analysis with recommendations"
            ),
            tools=["graphrag"],
            llm="gpt-4o-mini",
        ),
        "planner": AgentConfig(
            name="Planner",
            role="Strategic Planning Specialist",
            goal="Create actionable plans and strategies",
            backstory=(
                "You are a strategic planning expert who creates comprehensive, actionable "
                "plans based on research and analysis."
            ),
            instructions=(
                "1. Review all previous agent outputs\n"
                "2. Query GraphRAG for relevant planning context\n"
                "3. Create structured plan with clear steps\n"
                "4. Include timeline, resources, and success metrics"
            ),
            tools=["graphrag"],
            llm="gpt-4o-mini",
        ),
        "standards": AgentConfig(
            name="StandardsAgent",
            role="Quality Assurance Specialist",
            goal="Ensure response quality and completeness",
            backstory=(
                "You are a quality assurance expert who reviews outputs for accuracy, "
                "completeness, and adherence to best practices."
            ),
            instructions=(
                "1. Review final output for accuracy and completeness\n"
                "2. Check against GraphRAG memory for consistency\n"
                "3. Verify all claims are properly sourced\n"
                "4. Ensure response meets quality standards"
            ),
            tools=["graphrag"],
            llm="gpt-4o-mini",
        ),
    }

    if agent_name_lower not in templates:
        available = ", ".join(sorted(templates.keys()))
        raise ValueError(
            f"Unknown agent template: '{agent_name}'. "
            f"Available templates (case-insensitive): {available}"
        )

    return templates[agent_name_lower]


def _build_chat_agents(memory_manager: MemoryManager, use_tools: bool = True) -> tuple:
    """Build standard chat agents with GraphRAG tool."""
    from orchestrator.tools.graph_rag_tool import create_graph_rag_tool

    # Create GraphRAG tool
    graphrag_tool = create_graph_rag_tool(memory_manager)

    # Web search tools (optional)
    web_tools = []
    if use_tools and duckduckgo_tool:
        web_tools.append(duckduckgo_tool)
    if use_tools and wikipedia_tool:
        web_tools.append(wikipedia_tool)

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
    """
    Extract decision payload from router output.
    Supports JSON objects and markdown code blocks.
    """
    if not text:
        return {}

    # Try direct JSON parsing
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    import re

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding JSON object in text
    json_obj_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', text)
    if json_obj_match:
        try:
            data = json.loads(json_obj_match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract decision keyword
    decision_match = re.search(r'decision["\s:]+(\w+)', text, re.IGNORECASE)
    if decision_match:
        return {"decision": decision_match.group(1)}

    return {}


def _extract_text(output: Any) -> str:
    """Extract text content from various output formats."""
    if output is None:
        return ""

    # PRIORITY 1 FIX: Handle OrchestratorState with final_output field
    if hasattr(output, "final_output") and output.final_output:
        return str(output.final_output)

    # Handle string output
    if isinstance(output, str):
        return output

    # Handle dict with final_output
    if isinstance(output, dict):
        if "final_output" in output:
            return str(output["final_output"])
        # Try common text fields
        for key in ["output", "text", "content", "response", "result"]:
            if key in output:
                return str(output[key])
        # Fallback to messages
        if "messages" in output:
            messages = output["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    return last_msg.get("content", str(output))
                elif hasattr(last_msg, "content"):
                    return str(last_msg.content)
        return str(output)

    # Handle TaskOutput objects
    if hasattr(output, "raw"):
        return str(output.raw)
    if hasattr(output, "content"):
        return str(output.content)

    return str(output)


def _extract_decision(output: Any) -> Optional[str]:
    """Extract routing decision from various output formats including StateGraph state."""
    if output is None:
        return None

    # PHASE 1 FIX: Check StateGraph state object attributes first
    if hasattr(output, "current_route"):
        route = getattr(output, "current_route")
        return str(route).strip() if route else None

    # Check for router_decision attribute (alternative state field)
    if hasattr(output, "router_decision"):
        route = getattr(output, "router_decision")
        return str(route).strip() if route else None

    # PHASE 1 FIX: Check dict state (StateGraph can return dict or object)
    if isinstance(output, dict):
        # StateGraph state as dict
        if "current_route" in output:
            return str(output["current_route"]).strip()
        if "router_decision" in output:
            return str(output["router_decision"]).strip()
        # Legacy router payload
        if "decision" in output:
            return str(output["decision"]).strip()

    # EXISTING CODE: Check other formats (json_dict, pydantic, raw JSON)
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
