"""
Prompt builder utility for BIM-IR agent.

Constructs dynamic prompts following the BIM-GPT paper structure (Figure 4):
1. System Role
2. Relevant Database Info
3. Task Instruction
4. Few-Shot Examples
5. User Query
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Utility class for building prompts for Intent Classification and Parameter Extraction."""

    @staticmethod
    def build_intent_classification_prompt(
        examples: List[Dict[str, Any]],
        schema: Dict[str, Any],
        query: str
    ) -> str:
        """
        Build complete prompt for intent classification.

        Args:
            examples: Few-shot examples
            schema: BIM property schema
            query: User query to classify

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        # Component 1: System Role
        prompt_parts.append(PromptBuilder._build_system_role())

        # Component 2: Relevant Database Info
        prompt_parts.append(PromptBuilder._build_database_info(schema))

        # Component 3: Task Instruction
        prompt_parts.append(PromptBuilder._build_task_instruction())

        # Component 4: Few-Shot Examples
        prompt_parts.append(PromptBuilder._build_few_shot_examples(examples))

        # Component 5: User Query
        prompt_parts.append(PromptBuilder._build_user_query(query))

        # Join all parts
        full_prompt = "\n\n".join(prompt_parts)

        logger.debug(f"Built prompt with {len(full_prompt)} characters")

        return full_prompt

    @staticmethod
    def _build_system_role() -> str:
        """Build Component 1: System Role."""
        return """You are an expert BIM (Building Information Modeling) information retrieval specialist. Your task is to classify natural language queries about building models into intent categories.

You must determine:
1. Whether the query is BIM-related (is_bim: true/false)
2. The intent category if BIM-related (location/quantity/material/detail/area)

Non-BIM queries are general questions unrelated to building models (e.g., weather, people, dates, general knowledge)."""

    @staticmethod
    def _build_database_info(schema: Dict[str, Any]) -> str:
        """Build Component 2: Relevant Database Info."""
        usage_patterns = schema.get("usage_patterns", {})

        info = "Available BIM Intent Categories:\n\n"

        # Location
        location_info = usage_patterns.get("location_queries", {})
        info += f"- location: {location_info.get('description', 'Queries asking WHERE elements are located')}\n"
        info += f"  Examples: {', '.join(location_info.get('examples', []))}\n\n"

        # Quantity
        quantity_info = usage_patterns.get("quantity_queries", {})
        info += f"- quantity: {quantity_info.get('description', 'Queries asking HOW MANY elements exist')}\n"
        info += f"  Examples: {', '.join(quantity_info.get('examples', []))}\n\n"

        # Material
        material_info = usage_patterns.get("material_queries", {})
        info += f"- material: {material_info.get('description', 'Queries asking WHAT MATERIAL elements are made of')}\n"
        info += f"  Examples: {', '.join(material_info.get('examples', []))}\n\n"

        # Detail
        detail_info = usage_patterns.get("detail_queries", {})
        info += f"- detail: {detail_info.get('description', 'Queries asking about SPECIFIC PROPERTIES')}\n"
        info += f"  Examples: {', '.join(detail_info.get('examples', []))}\n\n"

        # Area
        area_info = usage_patterns.get("area_queries", {})
        info += f"- area: {area_info.get('description', 'Queries asking about AREA or VOLUME measurements')}\n"
        info += f"  Examples: {', '.join(area_info.get('examples', []))}"

        return info

    @staticmethod
    def _build_task_instruction() -> str:
        """Build Component 3: Task Instruction."""
        return """Analyze the user query and respond with JSON in this exact format:
{
  "is_bim": true or false,
  "intent": "category_name" or null,
  "category": "category_name" or null,
  "confidence": 0.0-1.0
}

Rules:
- If is_bim is false, both intent and category must be null
- If is_bim is true, intent and category must be one of: location, quantity, material, detail, area
- intent and category should always have the same value
- confidence must be a float between 0.0 and 1.0 indicating classification certainty
- Only output the JSON, no additional text"""

    @staticmethod
    def _build_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
        """Build Component 4: Few-Shot Examples."""
        examples_text = "# Examples:\n\n"

        # Group examples by category
        categories = {
            "location": [],
            "quantity": [],
            "material": [],
            "detail": [],
            "area": [],
            "non_bim": []
        }

        for ex in examples:
            intent = ex.get("intent", {})
            if not intent.get("is_bim"):
                categories["non_bim"].append(ex)
            else:
                cat = intent.get("category")
                if cat in categories:
                    categories[cat].append(ex)

        # Add examples for each category
        for category_name, category_examples in categories.items():
            if not category_examples:
                continue

            # Header
            if category_name == "non_bim":
                examples_text += "## Non-BIM queries (is_bim: false)\n"
            else:
                examples_text += f"## {category_name.capitalize()} queries (is_bim: true, category: {category_name})\n"

            # Add examples (limit to 2-3 per category to keep prompt manageable)
            for ex in category_examples[:3]:
                query = ex.get("query", "")
                intent = ex.get("intent", {})

                examples_text += f'Query: "{query}"\n'
                examples_text += f'Output: {{"is_bim": {str(intent.get("is_bim")).lower()}, '
                examples_text += f'"intent": {_format_json_value(intent.get("intent"))}, '
                examples_text += f'"category": {_format_json_value(intent.get("category"))}}}\n\n'

            examples_text += "\n"

        return examples_text.strip()

    @staticmethod
    def _build_user_query(query: str) -> str:
        """Build Component 5: User Query."""
        return f'# Now classify this query:\n\nQuery: "{query}"\nOutput:'

    # ============================================================
    # Parameter Extraction Prompt Building (Block 2)
    # ============================================================

    @staticmethod
    def build_parameter_extraction_prompt(
        examples: List[Dict[str, Any]],
        schema: Dict[str, Any],
        query: str,
        intent_category: str
    ) -> str:
        """
        Build complete prompt for parameter extraction (Block 2).

        Args:
            examples: Few-shot examples with parameters
            schema: BIM property schema
            query: User query to extract parameters from
            intent_category: Intent category from Block 1 (location, quantity, etc.)

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        # Component 1: System Role
        prompt_parts.append(PromptBuilder._build_param_system_role())

        # Component 2: Database Info
        prompt_parts.append(PromptBuilder._build_param_database_info(schema))

        # Component 3: Task Instruction
        prompt_parts.append(PromptBuilder._build_param_task_instruction(intent_category))

        # Component 4: Few-Shot Examples
        prompt_parts.append(PromptBuilder._build_param_few_shot_examples(examples))

        # Component 5: User Query
        prompt_parts.append(PromptBuilder._build_param_user_query(query, intent_category))

        # Join all parts
        full_prompt = "\n\n".join(prompt_parts)

        logger.debug(f"Built parameter extraction prompt with {len(full_prompt)} characters")

        return full_prompt

    @staticmethod
    def _build_param_system_role() -> str:
        """Build Component 1: System Role for parameter extraction."""
        return """You are an expert BIM parameter extraction specialist. Your task is to extract structured filter and projection parameters from natural language queries about building models.

You must extract:
1. filter_para: Properties to filter by (WHERE clause) - e.g., Category=Walls, Level=Level 1
2. proj_para: Properties to return (SELECT clause) - e.g., Volume, Area, Name

All parameters must use canonical property names from the available properties list."""

    @staticmethod
    def _build_param_database_info(schema: Dict[str, Any]) -> str:
        """Build Component 2: Database Info for parameter extraction."""
        properties = schema.get("properties", {})

        info = "Available BIM Properties:\n\n"

        info += "## Filter Properties (WHERE clause):\n"
        filter_props = ["Category", "Level", "Type", "Material", "Family", "Name"]
        for prop in filter_props:
            prop_data = properties.get(prop, {})
            desc = prop_data.get("description", "")
            synonyms = prop_data.get("synonyms", [])
            info += f"- {prop}: {desc}\n"
            if synonyms:
                info += f"  Synonyms: {', '.join(synonyms)}\n"

        info += "\n## Projection Properties (SELECT clause):\n"
        proj_props = ["Volume", "Area", "Height", "Width", "Length", "Thickness",
                      "Count", "Location", "Name", "Material", "Total_Floor_Area"]
        for prop in proj_props:
            prop_data = properties.get(prop, {})
            desc = prop_data.get("description", "")
            info += f"- {prop}: {desc}\n"

        return info

    @staticmethod
    def _build_param_task_instruction(intent_category: str) -> str:
        """Build Component 3: Task Instruction for parameter extraction."""
        # Category-specific guidance
        category_guidance = {
            "location": "Must include Location or Name in proj_para for identifying element positions",
            "quantity": "Must include Count in proj_para for counting elements",
            "material": "Must include Material in proj_para for material information",
            "detail": "Include specific properties mentioned in query (Volume, Area, Height, etc.)",
            "area": "Must include Area or Total_Floor_Area in proj_para for area measurements"
        }

        guidance = category_guidance.get(intent_category, "Extract parameters based on query content")

        return f"""Given that this is a **{intent_category}** query, extract parameters and respond with JSON in this exact format:

{{
  "filter_para": [{{"name": "PropertyName", "value": "value"}}, ...],
  "proj_para": ["PropertyName1", "PropertyName2", ...],
  "confidence": 0.0-1.0
}}

Rules:
- filter_para: Array of objects with "name" and "value" (can be empty for "show all" queries)
- proj_para: Array of property name strings (must be non-empty)
- All property names must match exactly from the available properties list
- Use canonical property names (e.g., "Category" not "category")
- Normalize values to canonical forms (e.g., "Level 1" for ground floor)
- {guidance}

Only output the JSON, no additional text."""

    @staticmethod
    def _build_param_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
        """Build Component 4: Few-Shot Examples for parameter extraction."""
        examples_text = "# Examples:\n\n"

        # Select examples grouped by parameter complexity
        selected_examples = {
            "simple_filter": [],      # 1 filter param
            "multi_filter": [],       # 2+ filter params
            "simple_projection": [],  # 1 projection param
            "multi_projection": [],   # 2+ projection params
            "no_filter": []           # Empty filter
        }

        # Categorize examples
        for ex in examples:
            params = ex.get("parameters", {})
            filter_para = params.get("filter_para", [])
            proj_para = params.get("proj_para", [])

            if len(filter_para) == 0:
                selected_examples["no_filter"].append(ex)
            elif len(filter_para) == 1:
                selected_examples["simple_filter"].append(ex)
            elif len(filter_para) >= 2:
                selected_examples["multi_filter"].append(ex)

            if len(proj_para) == 1:
                selected_examples["simple_projection"].append(ex)
            elif len(proj_para) >= 2:
                selected_examples["multi_projection"].append(ex)

        # Add examples from each category (limit to 3 per category)
        # Simple filter (1 param)
        if selected_examples["simple_filter"]:
            examples_text += "## Simple filter examples (WHERE with 1 condition)\n"
            for ex in selected_examples["simple_filter"][:3]:
                examples_text += PromptBuilder._format_param_example(ex)
            examples_text += "\n"

        # Multi-filter (2+ params)
        if selected_examples["multi_filter"]:
            examples_text += "## Multi-filter examples (WHERE with multiple conditions)\n"
            for ex in selected_examples["multi_filter"][:3]:
                examples_text += PromptBuilder._format_param_example(ex)
            examples_text += "\n"

        # Simple projection
        if selected_examples["simple_projection"]:
            examples_text += "## Simple projection examples (SELECT single property)\n"
            for ex in selected_examples["simple_projection"][:3]:
                examples_text += PromptBuilder._format_param_example(ex)
            examples_text += "\n"

        # Multi-projection
        if selected_examples["multi_projection"]:
            examples_text += "## Multi-projection examples (SELECT multiple properties)\n"
            for ex in selected_examples["multi_projection"][:2]:
                examples_text += PromptBuilder._format_param_example(ex)
            examples_text += "\n"

        # No filter
        if selected_examples["no_filter"]:
            examples_text += "## No filter examples (show all elements)\n"
            for ex in selected_examples["no_filter"][:2]:
                examples_text += PromptBuilder._format_param_example(ex)
            examples_text += "\n"

        return examples_text.strip()

    @staticmethod
    def _format_param_example(example: Dict[str, Any]) -> str:
        """Format a single parameter extraction example."""
        query = example.get("query", "")
        params = example.get("parameters", {})
        filter_para = params.get("filter_para", [])
        proj_para = params.get("proj_para", [])

        # Format filter_para as JSON array of objects
        filter_json = [{"name": f["name"], "value": f["value"]} for f in filter_para]

        # Format proj_para as JSON array of strings (already in correct format)
        proj_json = proj_para

        output = f'Query: "{query}"\n'
        output += f'Output: {{"filter_para": {filter_json}, "proj_para": {proj_json}}}\n\n'

        return output

    @staticmethod
    def _build_param_user_query(query: str, intent_category: str) -> str:
        """Build Component 5: User Query for parameter extraction."""
        return f'# Now extract parameters from this {intent_category} query:\n\nQuery: "{query}"\nOutput:'


def _format_json_value(value):
    """Format value for JSON output in examples."""
    if value is None:
        return "null"
    elif isinstance(value, str):
        return f'"{value}"'
    else:
        return str(value)
