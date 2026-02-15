"""Lightweight prompt template registry.

Templates use ``str.format()`` placeholders.  Literal JSON braces
in templates must be escaped as ``{{`` / ``}}``.
"""

from typing import Dict, List


class PromptRegistry:
    """Global registry of named prompt templates."""

    _templates: Dict[str, str] = {}

    @classmethod
    def get(cls, template_name: str, **kwargs: object) -> str:
        """Retrieve and optionally render a prompt template.

        Args:
            template_name: Dot-separated template name
                (e.g. ``"routing.classify"``).
            **kwargs: Values for ``str.format()`` placeholders.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If *template_name* is not registered.
        """
        if template_name not in cls._templates:
            raise KeyError(
                f"Unknown prompt template: '{template_name}'. "
                f"Available: {cls.list_templates()}"
            )
        template = cls._templates[template_name]
        return template.format(**kwargs) if kwargs else template

    @classmethod
    def register(cls, template_name: str, template: str) -> None:
        """Register a new template (or overwrite an existing one)."""
        cls._templates[template_name] = template

    @classmethod
    def list_templates(cls) -> List[str]:
        """Return sorted list of registered template names."""
        return sorted(cls._templates.keys())


def get_prompt(template_name: str, **kwargs: object) -> str:
    """Convenience shortcut for ``PromptRegistry.get()``."""
    return PromptRegistry.get(template_name, **kwargs)
