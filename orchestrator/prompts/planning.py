"""Planning prompt templates for Tree-of-Thought planner (BP-MCP-05, BP-PROMPT-08).

Templates:
- ``planning.base_graph``  — base graph specification prompt
- ``planning.cot_graph``   — chain-of-thought wrapper
- ``planning.value_graph`` — evaluation / scoring prompt
- ``planning.vote_graph``  — candidate selection prompt
"""

from .registry import PromptRegistry

# ── planning.base_graph ──────────────────────────────────────────
# Placeholders: {problem}, {agent_lines}, {agent_names_str}, {current_plan}

_BASE_GRAPH = (
    "<instructions>\n"
    "Eres un arquitecto de workflows de agentes especializado en crear grafos StateGraph.\n"
    "Tu objetivo es disenar un grafo de ejecucion estructurado que permita mayor control sobre el flujo.\n"
    "</instructions>\n\n"
    "<context>\n"
    "Problema a resolver: {problem}\n\n"
    "Agentes disponibles:\n"
    "{agent_lines}\n\n"
    "IMPORTANTE: Los nombres exactos de agentes disponibles son: {agent_names_str}\n"
    "Cuando crees nodos de tipo 'agent', el campo 'agent' DEBE ser uno de estos nombres exactos.\n"
    "Cuando crees parallel_groups, los nombres en 'parallel_nodes' deben corresponder a nombres de nodos validos.\n\n"
    "Especificacion actual del grafo:\n{current_plan}\n"
    "</context>\n\n"
    "<output_format>\n"
    "Genera el siguiente componente del grafo en formato JSON COMPACTO (una sola linea).\n"
    "Responde UNICAMENTE con JSON valido en UNA LINEA, sin saltos de linea ni explicaciones.\n\n"
    "Formato esperado (TODO EN UNA LINEA):\n"
    '{{"component_type": "node|edge|parallel_group", "name": "nombre_unico", '
    '"type": "agent|router|parallel|condition|start|end", "agent": "NOMBRE_AGENTE_EXACTO_DE_LA_LISTA", '
    '"objective": "descripcion_especifica", "expected_output": "resultado_esperado", '
    '"from_node": "nodo_origen_para_edges", "to_node": "nodo_destino_para_edges", '
    '"edge_type": "direct|conditional|parallel", '
    '"condition": {{"type": "state_check", "field": "campo", "operator": "equals", "value": "valor"}}, '
    '"parallel_nodes": ["nombre_nodo_1", "nombre_nodo_2"]}}\n'
    "</output_format>\n\n"
    "REGLAS CRITICAS:\n"
    "1. Para nodos tipo 'agent': el campo 'agent' es OBLIGATORIO y debe ser exactamente uno de: "
    "{agent_names_str}\n"
    "2. Para parallel_groups: primero crea los nodos individuales, luego el parallel_group que los referencia\n"
    "3. El 'name' del nodo puede ser descriptivo pero el 'agent' debe ser el nombre exacto del agente\n\n"
    "<examples>\n"
    '{{"component_type":"node","name":"research","type":"agent","agent":"Researcher",'
    '"objective":"Gather market information","expected_output":"Research report"}}\n'
    '{{"component_type":"node","name":"analysis","type":"agent","agent":"Analyst",'
    '"objective":"Analyze data","expected_output":"Analysis report"}}\n'
    '{{"component_type":"edge","from_node":"research","to_node":"analysis","edge_type":"direct"}}\n'
    '{{"component_type":"parallel_group","name":"parallel_research","parallel_nodes":["research","analysis"]}}\n'
    "</examples>"
)

PromptRegistry.register("planning.base_graph", _BASE_GRAPH)

# ── planning.cot_graph ───────────────────────────────────────────
# Placeholder: {base_prompt}

_COT_GRAPH = (
    "{base_prompt}\n\n"
    "Piensa paso a paso:\n"
    "1. ¿Qué tipo de componente necesita el grafo ahora?\n"
    "2. ¿Cómo se conecta con componentes existentes?\n"
    "3. ¿Se puede ejecutar en paralelo con otros componentes?\n"
    "4. ¿Necesita condiciones especiales para la ejecución?\n\n"
    "Después de pensar, devuelve ÚNICAMENTE el JSON del componente."
)

PromptRegistry.register("planning.cot_graph", _COT_GRAPH)

# ── planning.value_graph ─────────────────────────────────────────
# Placeholders: {problem}, {plan}

_VALUE_GRAPH = (
    "<instructions>\n"
    "Evalua la calidad de la siguiente especificacion de grafo StateGraph.\n"
    "</instructions>\n\n"
    "<context>\n"
    "Problema: {problem}\n"
    "Especificacion del grafo:\n{plan}\n"
    "</context>\n\n"
    "<output_format>\n"
    'Responde en JSON estricto: {{"score": <0-10>, "reason": "..."}}.\n\n'
    "Criterios de evaluacion:\n"
    "- Cobertura del problema (resuelve todos los aspectos?)\n"
    "- Estructura del grafo (es logica y eficiente?)\n"
    "- Uso apropiado de agentes (asignaciones correctas?)\n"
    "- Oportunidades de paralelizacion (maximiza eficiencia?)\n"
    "- Manejo de condiciones y routing (control de flujo adecuado?)\n"
    "</output_format>\n\n"
    "<examples>\n"
    '{{"score": 8, "reason": "Buena cobertura con paralelizacion apropiada, pero falta nodo de validacion final"}}\n'
    '{{"score": 3, "reason": "Solo cubre investigacion, no incluye analisis ni sintesis de resultados"}}\n'
    "</examples>"
)

PromptRegistry.register("planning.value_graph", _VALUE_GRAPH)

# ── planning.vote_graph ──────────────────────────────────────────
# Placeholders: {problem}, {choices}

_VOTE_GRAPH = (
    "Selecciona la mejor especificación de grafo para resolver el problema.\n"
    "Responde con el número de la opción más prometedora.\n\n"
    "Problema: {problem}\n\n{choices}\n"
)

PromptRegistry.register("planning.vote_graph", _VOTE_GRAPH)
