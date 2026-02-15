"""System prompt template for LangChain agents (BP-MCP-05, BP-PROMPT-08).

Template: ``system.agent``
Placeholders: ``{name}``, ``{role}``, ``{goal}``, ``{backstory}``, ``{instructions}``
"""

from .registry import PromptRegistry

_SYSTEM_AGENT = (
    "<role>You are {name}, a {role}.</role>\n\n"
    "<goal>{goal}</goal>\n\n"
    "<backstory>{backstory}</backstory>\n\n"
    "<instructions>\n{instructions}\n</instructions>\n\n"
    "Provide clear, actionable responses based on your role and expertise."
)

PromptRegistry.register("system.agent", _SYSTEM_AGENT)
