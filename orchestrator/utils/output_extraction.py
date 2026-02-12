"""Output extraction utilities for parsing agent and router results.

Extracted from cli/main.py to make these functions available to all layers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def parse_router_payload(text: str) -> Dict[str, Any]:
    """Extract decision payload from router output.

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


def extract_text(output: Any) -> str:
    """Extract text content from various output formats."""
    if output is None:
        return ""

    # Handle OrchestratorState with final_output field
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


def extract_decision(output: Any) -> Optional[str]:
    """Extract routing decision from various output formats including StateGraph state."""
    if output is None:
        return None

    # Check StateGraph state object attributes first
    if hasattr(output, "current_route"):
        route = getattr(output, "current_route")
        return str(route).strip() if route else None

    if hasattr(output, "router_decision"):
        route = getattr(output, "router_decision")
        return str(route).strip() if route else None

    # Check dict state (StateGraph can return dict or object)
    if isinstance(output, dict):
        if "current_route" in output:
            return str(output["current_route"]).strip()
        if "router_decision" in output:
            return str(output["router_decision"]).strip()
        if "decision" in output:
            return str(output["decision"]).strip()

    # Check other formats (json_dict, pydantic, raw JSON)
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
