"""Routing prompt template (BP-MCP-05, BP-PROMPT-08).

Template: ``routing.classify``
Placeholder: ``{user_request}``
"""

from .registry import PromptRegistry

_ROUTING_CLASSIFY = (
    "<instructions>\n"
    "Analyze the user request and select the best route.\n"
    "</instructions>\n\n"
    "<context>\n"
    "USER REQUEST: {user_request}\n"
    "</context>\n\n"
    "<output_format>\n"
    'Respond with JSON: {{"route": "route_name", "confidence": float, "reasoning": "why this route"}}\n\n'
    "AVAILABLE ROUTES:\n"
    "- quick: Simple responses, greetings, trivial questions\n"
    "- research: Information gathering, web search, fact-finding\n"
    "- analysis: Deep analysis, reasoning, detailed examination\n"
    "- planning: Complex tasks requiring multi-step planning + multi-agent execution\n"
    "- standards: Compliance, regulations, normative references\n"
    "</output_format>\n\n"
    "<examples>\n"
    'User: "Hello, how are you?"\n'
    '{{"route": "quick", "confidence": 0.95, "reasoning": "Simple greeting"}}\n\n'
    'User: "Find the latest research on BIM interoperability standards"\n'
    '{{"route": "research", "confidence": 0.9, "reasoning": "Information gathering request requiring web search"}}\n\n'
    'User: "Design a multi-agent pipeline to automate IFC model validation"\n'
    '{{"route": "planning", "confidence": 0.92, "reasoning": "Complex task requiring multi-step planning and agent coordination"}}\n'
    "</examples>"
)

PromptRegistry.register("routing.classify", _ROUTING_CLASSIFY)
