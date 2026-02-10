"""
Viewer highlighting utility for visual BIM feedback.

Generates Autodesk Viewer highlighting instructions from BIM query results.
"""

import re
from typing import List, Tuple, Optional
from agents.bim_ir.models.highlight_result import (
    HighlightConfig,
    HighlightResult,
    HighlightMode,
    ViewerCommand,
    InvalidColorError,
    InvalidElementIDError,
    ViewerCommandError
)
from agents.bim_ir.models.retriever_result import RetrieverResult


class ViewerHighlighter:
    """
    Generate Autodesk Viewer highlighting instructions.

    Converts BIM query results into viewer commands for visual feedback
    in the Autodesk Forge/APS Viewer.

    Usage:
        highlighter = ViewerHighlighter()

        # From RetrieverResult
        result = highlighter.highlight_from_result(
            retriever_result=result,
            mode=HighlightMode.COLOR,
            color="#FF6600"
        )

        # From element IDs
        result = highlighter.highlight(
            config=HighlightConfig(
                element_ids=[1234, 5678],
                model_urn="urn:...",
                mode=HighlightMode.SELECT
            )
        )

        # Use JavaScript output
        print(result.javascript)
    """

    def highlight(self, config: HighlightConfig) -> HighlightResult:
        """
        Generate viewer highlighting instructions.

        Args:
            config: Highlighting configuration

        Returns:
            HighlightResult with commands and JavaScript

        Raises:
            InvalidElementIDError: If element IDs are invalid
            InvalidColorError: If color format is invalid
            ViewerCommandError: If command generation fails
        """
        # Validate inputs
        config.validate()

        # Generate commands based on mode
        if config.mode == HighlightMode.SELECT:
            commands = self._generate_select_commands(config)
        elif config.mode == HighlightMode.COLOR:
            commands = self._generate_color_commands(config)
        elif config.mode == HighlightMode.ISOLATE:
            commands = self._generate_isolate_commands(config)
        else:
            raise ViewerCommandError(f"Unknown highlight mode: {config.mode}")

        # Generate complete JavaScript
        javascript = self._to_javascript(commands, config)

        # Build metadata
        metadata = {
            "model_urn": config.model_urn,
            "color": config.color if config.mode == HighlightMode.COLOR else None,
            "fit_to_view": config.fit_to_view,
            "clear_previous": config.clear_previous,
            "truncated": len(config.element_ids) >= config.max_elements
        }

        return HighlightResult(
            commands=commands,
            javascript=javascript,
            element_count=len(config.element_ids),
            mode=config.mode,
            metadata=metadata
        )

    def highlight_from_result(
        self,
        retriever_result: RetrieverResult,
        mode: HighlightMode = HighlightMode.SELECT,
        color: str = "#FF6600",
        fit_to_view: bool = True,
        max_elements: int = 500
    ) -> HighlightResult:
        """
        Generate highlighting from RetrieverResult (Block 4 output).

        Convenience method that extracts element IDs from retrieval results.

        Args:
            retriever_result: Result from Retriever (Block 4)
            mode: Highlighting mode
            color: Hex color for COLOR mode
            fit_to_view: Whether to zoom to highlighted elements
            max_elements: Maximum elements to highlight

        Returns:
            HighlightResult with commands and JavaScript
        """
        # Extract element IDs from retriever result
        element_ids = [elem.element_id for elem in retriever_result.elements]

        # Get model URN from metadata
        model_urn = retriever_result.query_metadata.model_urn

        # Create config and highlight
        config = HighlightConfig(
            element_ids=element_ids,
            model_urn=model_urn,
            color=color,
            mode=mode,
            fit_to_view=fit_to_view,
            max_elements=max_elements
        )

        return self.highlight(config)

    def _generate_select_commands(self, config: HighlightConfig) -> List[ViewerCommand]:
        """
        Generate SELECT mode commands (blue outline highlight).

        Uses viewer.select() to highlight elements with blue outline.
        """
        commands = []

        # Clear previous selection if requested
        if config.clear_previous:
            commands.append(ViewerCommand(
                command_type="clearSelection",
                parameters={},
                javascript="viewer.clearSelection();"
            ))

        # Select elements
        element_ids_str = ", ".join(map(str, config.element_ids))
        commands.append(ViewerCommand(
            command_type="select",
            parameters={"dbIds": config.element_ids},
            javascript=f"viewer.select([{element_ids_str}]);"
        ))

        # Fit to view if requested
        if config.fit_to_view:
            commands.append(ViewerCommand(
                command_type="fitToView",
                parameters={"dbIds": config.element_ids},
                javascript=f"viewer.fitToView([{element_ids_str}]);"
            ))

        return commands

    def _generate_color_commands(self, config: HighlightConfig) -> List[ViewerCommand]:
        """
        Generate COLOR mode commands (custom color theming).

        Uses viewer.setThemingColor() to apply custom colors to elements.
        """
        commands = []

        # Clear previous theming if requested
        if config.clear_previous:
            commands.append(ViewerCommand(
                command_type="clearThemingColors",
                parameters={},
                javascript="viewer.clearThemingColors();"
            ))

        # Convert hex color to Vector4
        r, g, b = self._hex_to_rgb_normalized(config.color)

        # Apply theming color to each element
        # Note: Batching for performance - generate single command with loop
        element_ids_str = ", ".join(map(str, config.element_ids))
        color_js = f"new THREE.Vector4({r}, {g}, {b}, 1.0)"

        # Generate batched JavaScript for performance
        js_code = f"""
// Apply color theming to {len(config.element_ids)} elements
(function() {{
    const color = {color_js};
    const dbIds = [{element_ids_str}];
    dbIds.forEach(dbId => viewer.setThemingColor(dbId, color));
}})();
""".strip()

        commands.append(ViewerCommand(
            command_type="setThemingColor",
            parameters={
                "dbIds": config.element_ids,
                "color": {"r": r, "g": g, "b": b, "a": 1.0}
            },
            javascript=js_code
        ))

        # Fit to view if requested
        if config.fit_to_view:
            commands.append(ViewerCommand(
                command_type="fitToView",
                parameters={"dbIds": config.element_ids},
                javascript=f"viewer.fitToView([{element_ids_str}]);"
            ))

        return commands

    def _generate_isolate_commands(self, config: HighlightConfig) -> List[ViewerCommand]:
        """
        Generate ISOLATE mode commands (show only selected, hide others).

        Uses viewer.isolate() to show only highlighted elements.
        """
        commands = []

        # Isolate elements (hides everything else)
        element_ids_str = ", ".join(map(str, config.element_ids))
        commands.append(ViewerCommand(
            command_type="isolate",
            parameters={"dbIds": config.element_ids},
            javascript=f"viewer.isolate([{element_ids_str}]);"
        ))

        # Fit to view if requested
        if config.fit_to_view:
            commands.append(ViewerCommand(
                command_type="fitToView",
                parameters={"dbIds": config.element_ids},
                javascript=f"viewer.fitToView([{element_ids_str}]);"
            ))

        return commands

    def _hex_to_rgb_normalized(self, hex_color: str) -> Tuple[float, float, float]:
        """
        Convert hex color to normalized RGB (0.0-1.0 range).

        Args:
            hex_color: Hex color string (e.g., "#FF6600" or "#F60")

        Returns:
            Tuple of (r, g, b) normalized to 0.0-1.0

        Raises:
            InvalidColorError: If color format is invalid
        """
        # Remove # prefix
        hex_color = hex_color.lstrip("#")

        # Validate format
        if not re.match(r'^[0-9A-Fa-f]{3}$|^[0-9A-Fa-f]{6}$', hex_color):
            raise InvalidColorError(
                f"Invalid hex color: #{hex_color}. "
                "Expected format: #RRGGBB or #RGB"
            )

        # Expand 3-digit format to 6-digit
        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])

        # Convert to RGB (0-255)
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except ValueError as e:
            raise InvalidColorError(f"Failed to parse hex color: {e}")

        # Normalize to 0.0-1.0 range
        return (
            round(r / 255.0, 3),
            round(g / 255.0, 3),
            round(b / 255.0, 3)
        )

    def _to_javascript(
        self,
        commands: List[ViewerCommand],
        config: HighlightConfig
    ) -> str:
        """
        Generate complete executable JavaScript from commands.

        Args:
            commands: List of viewer commands
            config: Highlighting configuration

        Returns:
            Complete JavaScript code ready for execution
        """
        js_lines = [
            "// Autodesk Viewer Highlighting Commands",
            f"// Mode: {config.mode.value}",
            f"// Elements: {len(config.element_ids)}",
            ""
        ]

        # Add each command's JavaScript
        for cmd in commands:
            js_lines.append(cmd.javascript)

        return "\n".join(js_lines)
