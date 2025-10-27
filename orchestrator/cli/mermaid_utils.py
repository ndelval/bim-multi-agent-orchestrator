"""
Mermaid diagram generation utilities for LangGraph workflows.

This module provides functionality to generate Mermaid diagrams from compiled
LangGraph StateGraph objects and save them to files for visualization.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


def save_mermaid_diagram(
    compiled_graph: Any,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    include_timestamp: bool = True
) -> Optional[Path]:
    """
    Generate and save a Mermaid diagram from a compiled LangGraph StateGraph.

    Args:
        compiled_graph: Compiled StateGraph object from LangGraph
        output_dir: Directory to save the diagram (default: claudedocs/graphs/)
        filename: Base filename (default: workflow)
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Path to the saved Mermaid file, or None if generation failed

    Example:
        >>> from langgraph.graph import StateGraph
        >>> graph = StateGraph(...)
        >>> compiled = graph.compile()
        >>> path = save_mermaid_diagram(compiled, filename="my_workflow")
        >>> print(f"Diagram saved to: {path}")
    """
    try:
        # Get the graph object from compiled StateGraph
        if not hasattr(compiled_graph, 'get_graph'):
            logger.error("Compiled graph does not have get_graph() method")
            return None

        graph_obj = compiled_graph.get_graph()

        # Generate Mermaid diagram string
        if not hasattr(graph_obj, 'draw_mermaid'):
            logger.error("Graph object does not have draw_mermaid() method")
            return None

        mermaid_str = graph_obj.draw_mermaid()

        if not mermaid_str:
            logger.warning("Mermaid diagram generation returned empty string")
            return None

        # Determine output path
        if output_dir is None:
            # Default to claudedocs/graphs/ relative to project root
            output_dir = Path(__file__).parent.parent.parent / "claudedocs" / "graphs"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if filename is None:
            filename = "workflow"

        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

        output_path = output_dir / f"{filename}.mmd"

        # Write Mermaid diagram to file
        output_path.write_text(mermaid_str, encoding="utf-8")

        logger.info(f"✅ Mermaid diagram saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate Mermaid diagram: {e}")
        return None


def print_ascii_diagram(compiled_graph: Any) -> None:
    """
    Print ASCII representation of the graph to console.

    Args:
        compiled_graph: Compiled StateGraph object from LangGraph

    Example:
        >>> from langgraph.graph import StateGraph
        >>> graph = StateGraph(...)
        >>> compiled = graph.compile()
        >>> print_ascii_diagram(compiled)
    """
    try:
        if not hasattr(compiled_graph, 'get_graph'):
            logger.error("Compiled graph does not have get_graph() method")
            return

        graph_obj = compiled_graph.get_graph()

        if hasattr(graph_obj, 'print_ascii'):
            graph_obj.print_ascii()
        elif hasattr(graph_obj, 'draw_ascii'):
            print(graph_obj.draw_ascii())
        else:
            logger.warning("Graph object does not have ASCII rendering methods")

    except Exception as e:
        logger.error(f"Failed to print ASCII diagram: {e}")


def save_mermaid_png(
    compiled_graph: Any,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    include_timestamp: bool = True
) -> Optional[Path]:
    """
    Generate and save a Mermaid diagram as PNG image.

    Note: This requires the Mermaid CLI (mmdc) to be installed.
    Install with: npm install -g @mermaid-js/mermaid-cli

    Args:
        compiled_graph: Compiled StateGraph object from LangGraph
        output_dir: Directory to save the PNG (default: claudedocs/graphs/)
        filename: Base filename (default: workflow)
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Path to the saved PNG file, or None if generation failed

    Example:
        >>> from langgraph.graph import StateGraph
        >>> graph = StateGraph(...)
        >>> compiled = graph.compile()
        >>> path = save_mermaid_png(compiled, filename="my_workflow")
        >>> print(f"PNG saved to: {path}")
    """
    try:
        # Get the graph object from compiled StateGraph
        if not hasattr(compiled_graph, 'get_graph'):
            logger.error("Compiled graph does not have get_graph() method")
            return None

        graph_obj = compiled_graph.get_graph()

        # Check if draw_mermaid_png is available
        if not hasattr(graph_obj, 'draw_mermaid_png'):
            logger.warning("Graph object does not have draw_mermaid_png() method")
            logger.info("PNG generation requires LangGraph >=0.2.0 and Mermaid CLI")
            return None

        # Determine output path
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "claudedocs" / "graphs"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if filename is None:
            filename = "workflow"

        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

        output_path = output_dir / f"{filename}.png"

        # Generate PNG
        png_bytes = graph_obj.draw_mermaid_png()

        if not png_bytes:
            logger.warning("PNG generation returned empty bytes")
            return None

        # Write PNG to file
        output_path.write_bytes(png_bytes)

        logger.info(f"✅ Mermaid PNG saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate Mermaid PNG: {e}")
        logger.info("Make sure Mermaid CLI is installed: npm install -g @mermaid-js/mermaid-cli")
        return None


def get_graph_info(compiled_graph: Any) -> dict:
    """
    Extract basic information about the graph structure.

    Args:
        compiled_graph: Compiled StateGraph object from LangGraph

    Returns:
        Dictionary with graph metadata (nodes, edges, etc.)

    Example:
        >>> info = get_graph_info(compiled_graph)
        >>> print(f"Nodes: {info['node_count']}")
        >>> print(f"Edges: {info['edge_count']}")
    """
    try:
        if not hasattr(compiled_graph, 'get_graph'):
            return {"error": "Graph does not have get_graph() method"}

        graph_obj = compiled_graph.get_graph()

        info = {
            "node_count": 0,
            "edge_count": 0,
            "nodes": [],
            "edges": [],
        }

        # Extract nodes
        if hasattr(graph_obj, 'nodes'):
            nodes = graph_obj.nodes  # Access as property, not method
            info["nodes"] = list(nodes) if nodes else []
            info["node_count"] = len(info["nodes"])

        # Extract edges
        if hasattr(graph_obj, 'edges'):
            edges = graph_obj.edges  # Access as property, not method
            info["edges"] = list(edges) if edges else []
            info["edge_count"] = len(info["edges"])

        return info

    except Exception as e:
        logger.error(f"Failed to extract graph info: {e}")
        return {"error": str(e)}
