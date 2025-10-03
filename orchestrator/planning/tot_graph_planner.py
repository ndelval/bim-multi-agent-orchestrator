"""
Tree-of-Thought Graph Planner for StateGraph generation.

This module provides an enhanced ToT planner that can generate structured
StateGraph specifications instead of linear task assignments, giving users
more control over agent workflow orchestration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from .graph_specifications import (
    StateGraphSpec,
    GraphNodeSpec,
    GraphEdgeSpec,
    NodeType,
    EdgeType,
    RoutingStrategy,
    GraphCondition,
    ParallelGroup,
    create_simple_sequential_graph,
)
from .tot_planner import Task, solve, gpt_usage, _TOT_AVAILABLE, _try_import_tot
from orchestrator.core.config import AgentConfig, MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphPlanningSettings:
    """Configuration options for the ToT graph planner."""

    backend: str = "gpt-4"
    temperature: float = 0.7
    max_steps: int = 5
    n_generate_sample: int = 3
    n_evaluate_sample: int = 2
    n_select_sample: int = 2

    # Graph-specific settings
    enable_parallel_planning: bool = True
    enable_conditional_routing: bool = True
    max_parallel_groups: int = 3
    max_nodes_per_graph: int = 20

    # Fallback behavior configuration
    enable_auto_fallback: bool = True  # Auto-create sequential edges on validation errors
    preserve_tot_intent: bool = False  # If True, raise error instead of falling back
    strict_validation: bool = False  # If True, fail on any validation error


class GraphPlanningTask(Task):
    """ToT task for generating StateGraph specifications."""

    def __init__(
        self,
        problem_statement: str,
        agent_catalog: Sequence[AgentConfig],
        max_steps: int,
        settings: GraphPlanningSettings,
    ) -> None:
        super().__init__()
        self.instances = [problem_statement]
        self.agent_catalog = list(agent_catalog)
        self.steps = max_steps
        self.settings = settings

        # Use newline termination for structured output
        self.stops: List[Optional[str]] = ["\n"] * max_steps
        self.value_cache: Dict[str, float] = {}

        # Graph planning state
        self.current_graph_spec: Optional[StateGraphSpec] = None

    # --------------- Task protocol implementations ---------------
    def __len__(self) -> int:
        return len(self.instances)

    def get_input(self, idx: int) -> str:
        return self.instances[idx]

    def test_output(self, idx: int, output: str) -> Dict[str, Any]:
        """Validate graph specification output."""
        try:
            graph_data = self._parse_graph_output(output)
            return {
                "valid": True,
                "nodes": len(graph_data.get("nodes", [])),
                "edges": len(graph_data.get("edges", [])),
                "parallel_groups": len(graph_data.get("parallel_groups", [])),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    # --------------- Graph generation prompts ---------------
    def _base_graph_prompt(self, x: str, y: str = "") -> str:
        """Generate base prompt for graph planning."""
        agent_lines = [
            f"- {agent.name}: {agent.role}. Objetivo: {agent.goal}"
            for agent in self.agent_catalog
        ]

        # Extract just the agent names for explicit listing
        agent_names = [agent.name for agent in self.agent_catalog]
        agent_names_str = ", ".join(agent_names)

        current_plan = y.strip() if y.strip() else "<Sin especificación de grafo>"

        return (
            "Eres un arquitecto de workflows de agentes especializado en crear grafos StateGraph.\n"
            "Tu objetivo es diseñar un grafo de ejecución estructurado que permita mayor control sobre el flujo.\n\n"
            f"Problema a resolver: {x.strip()}\n\n"
            "Agentes disponibles:\n"
            f"{chr(10).join(agent_lines)}\n\n"
            f"IMPORTANTE: Los nombres exactos de agentes disponibles son: {agent_names_str}\n"
            "Cuando crees nodos de tipo 'agent', el campo 'agent' DEBE ser uno de estos nombres exactos.\n"
            "Cuando crees parallel_groups, los nombres en 'parallel_nodes' deben corresponder a nombres de nodos válidos.\n\n"
            f"Especificación actual del grafo:\n{current_plan}\n\n"
            "Genera el siguiente componente del grafo en formato JSON COMPACTO (una sola línea).\n"
            "Responde ÚNICAMENTE con JSON válido en UNA LÍNEA, sin saltos de línea ni explicaciones.\n\n"
            "Formato esperado (TODO EN UNA LÍNEA):\n"
            '{"component_type": "node|edge|parallel_group", "name": "nombre_unico", '
            '"type": "agent|router|parallel|condition|start|end", "agent": "NOMBRE_AGENTE_EXACTO_DE_LA_LISTA", '
            '"objective": "descripción_específica", "expected_output": "resultado_esperado", '
            '"from_node": "nodo_origen_para_edges", "to_node": "nodo_destino_para_edges", '
            '"edge_type": "direct|conditional|parallel", '
            '"condition": {"type": "state_check", "field": "campo", "operator": "equals", "value": "valor"}, '
            '"parallel_nodes": ["nombre_nodo_1", "nombre_nodo_2"]}\n\n'
            "REGLAS CRÍTICAS:\n"
            "1. Para nodos tipo 'agent': el campo 'agent' es OBLIGATORIO y debe ser exactamente uno de: "
            f"{agent_names_str}\n"
            "2. Para parallel_groups: primero crea los nodos individuales, luego el parallel_group que los referencia\n"
            "3. El 'name' del nodo puede ser descriptivo pero el 'agent' debe ser el nombre exacto del agente\n\n"
            "Ejemplos válidos:\n"
            '{"component_type":"node","name":"research_task","type":"agent","agent":"Researcher",'
            '"objective":"Gather market information","expected_output":"Research report"}\n'
            '{"component_type":"node","name":"analysis_task","type":"agent","agent":"Analyst",'
            '"objective":"Analyze data","expected_output":"Analysis report"}\n'
            '{"component_type":"parallel_group","name":"parallel_research","parallel_nodes":["research_task","analysis_task"]}\n'
        )

    def standard_prompt_wrap(self, x: str, y: str = "") -> str:
        return self._base_graph_prompt(x, y)

    def cot_prompt_wrap(self, x: str, y: str = "") -> str:
        prompt = (
            self._base_graph_prompt(x, y) + "\n\nPiensa paso a paso:\n"
            "1. ¿Qué tipo de componente necesita el grafo ahora?\n"
            "2. ¿Cómo se conecta con componentes existentes?\n"
            "3. ¿Se puede ejecutar en paralelo con otros componentes?\n"
            "4. ¿Necesita condiciones especiales para la ejecución?\n\n"
            "Después de pensar, devuelve ÚNICAMENTE el JSON del componente."
        )
        return prompt

    def propose_prompt_wrap(self, x: str, y: str = "") -> str:
        return self._base_graph_prompt(x, y)

    def value_prompt_wrap(self, x: str, y: str) -> str:
        """Evaluate graph specification quality."""
        plan = y.strip() if y.strip() else "<Sin especificación>"

        return (
            "Evalúa la calidad de la siguiente especificación de grafo StateGraph.\n"
            'Responde en JSON estricto: {"score": <0-10>, "reason": "..."}.\n\n'
            f"Problema: {x.strip()}\n"
            f"Especificación del grafo:\n{plan}\n\n"
            "Criterios de evaluación:\n"
            "- Cobertura del problema (¿resuelve todos los aspectos?)\n"
            "- Estructura del grafo (¿es lógica y eficiente?)\n"
            "- Uso apropiado de agentes (¿asignaciones correctas?)\n"
            "- Oportunidades de paralelización (¿maximiza eficiencia?)\n"
            "- Manejo de condiciones y routing (¿control de flujo adecuado?)\n"
        )

    def value_outputs_unwrap(self, x: str, y: str, value_outputs: List[str]) -> float:
        """Extract best score from evaluation outputs."""
        best_score = 0.0
        for raw in value_outputs:
            try:
                data = json.loads(raw)
                score = float(data.get("score", 0))
            except Exception:
                # Fallback: extract numeric score
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
                score = float(match.group(1)) if match else 0.0
            best_score = max(best_score, score)
        return best_score

    def vote_prompt_wrap(self, x: str, ys: List[str]) -> str:
        """Select best graph specification from candidates."""
        enumerated = "\n".join(f"Opción {i+1}:\n{y.strip()}" for i, y in enumerate(ys))
        return (
            "Selecciona la mejor especificación de grafo para resolver el problema.\n"
            "Responde con el número de la opción más prometedora.\n\n"
            f"Problema: {x.strip()}\n\n{enumerated}\n"
        )

    def vote_outputs_unwrap(
        self, vote_outputs: List[str], n_candidates: int
    ) -> List[int]:
        """Extract votes from voting outputs."""
        votes = [0] * n_candidates
        for raw in vote_outputs:
            match = re.search(r"(\d+)", raw)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < n_candidates:
                    votes[idx] += 1
        return votes

    # --------------- Graph parsing and construction ---------------
    def _parse_graph_output(self, output: str) -> Dict[str, Any]:
        """Parse ToT output into graph components."""
        components = {"nodes": [], "edges": [], "parallel_groups": []}

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                component = json.loads(line)
                comp_type = component.get("component_type", "")

                if comp_type == "node":
                    components["nodes"].append(self._create_node_spec(component))
                elif comp_type == "edge":
                    components["edges"].append(self._create_edge_spec(component))
                elif comp_type == "parallel_group":
                    components["parallel_groups"].append(
                        self._create_parallel_group(component)
                    )

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse graph component: {line}")
                continue

        return components

    def _create_node_spec(self, component: Dict[str, Any]) -> GraphNodeSpec:
        """Create GraphNodeSpec from component data."""
        node_type_map = {
            "agent": NodeType.AGENT,
            "router": NodeType.ROUTER,
            "parallel": NodeType.PARALLEL,
            "condition": NodeType.CONDITION,
            "start": NodeType.START,
            "end": NodeType.END,
        }

        return GraphNodeSpec(
            name=component.get("name", ""),
            type=node_type_map.get(component.get("type", "agent"), NodeType.AGENT),
            agent=component.get("agent"),
            objective=component.get("objective", ""),
            expected_output=component.get("expected_output", ""),
            tools=component.get("tools", []),
            routing_strategy=(
                RoutingStrategy.LLM_BASED
                if component.get("type") == "router"
                else RoutingStrategy.RULE_BASED
            ),
        )

    def _create_edge_spec(self, component: Dict[str, Any]) -> GraphEdgeSpec:
        """Create GraphEdgeSpec from component data."""
        edge_type_map = {
            "direct": EdgeType.DIRECT,
            "conditional": EdgeType.CONDITIONAL,
            "parallel": EdgeType.PARALLEL,
            "aggregation": EdgeType.AGGREGATION,
        }

        condition = None
        if component.get("condition"):
            cond_data = component["condition"]
            condition = GraphCondition(
                type=cond_data.get("type", "state_check"),
                field=cond_data.get("field"),
                operator=cond_data.get("operator", "equals"),
                value=cond_data.get("value"),
                description=cond_data.get("description", ""),
            )

        return GraphEdgeSpec(
            from_node=component.get("from_node", ""),
            to_node=component.get("to_node", ""),
            type=edge_type_map.get(
                component.get("edge_type", "direct"), EdgeType.DIRECT
            ),
            condition=condition,
            label=component.get("label", ""),
            description=component.get("description", ""),
        )

    def _create_parallel_group(self, component: Dict[str, Any]) -> ParallelGroup:
        """Create ParallelGroup from component data."""
        return ParallelGroup(
            group_id=component.get("name", ""),
            nodes=component.get("parallel_nodes", []),
            aggregation_strategy=component.get("aggregation_strategy", "collect_all"),
            description=component.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Public API for Graph Planning
# ---------------------------------------------------------------------------


def generate_graph_with_tot(
    prompt: str,
    recall_snippets: Sequence[str],
    agent_catalog: Sequence[AgentConfig],
    settings: Optional[GraphPlanningSettings] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> Dict[str, Any]:
    """
    Generate StateGraph specification using Tree-of-Thought planning.

    Args:
        prompt: User's problem statement
        recall_snippets: Context from memory retrieval
        agent_catalog: Available agents for graph construction
        settings: Graph planning configuration
        memory_config: Memory system configuration

    Returns:
        Dictionary containing:
        - graph_spec: StateGraphSpec object
        - metadata: Planning metadata and usage information
        - fallback_assignments: Linear assignments as fallback
    """
    if not _TOT_AVAILABLE and not _try_import_tot():
        logger.warning("ToT not available, falling back to sequential graph")
        return _generate_fallback_graph(prompt, recall_snippets, agent_catalog)

    settings = settings or GraphPlanningSettings()
    if not agent_catalog:
        raise ValueError("agent_catalog must not be empty")

    # Build enhanced problem statement for graph planning
    problem = _build_graph_problem_statement(
        prompt, recall_snippets, agent_catalog, memory_config
    )

    logger.info(
        "Invoking ToT graph planner (backend=%s, steps=%d, agents=%d)",
        settings.backend,
        settings.max_steps,
        len(agent_catalog),
    )
    logger.debug("ToT graph planning problem:\n%s", problem)

    # Create graph planning task
    task = GraphPlanningTask(
        problem, agent_catalog, max_steps=settings.max_steps, settings=settings
    )

    # Configure ToT solver arguments
    args = SimpleNamespace(
        backend=settings.backend,
        temperature=settings.temperature,
        method_generate="sample",
        method_evaluate="value",
        method_select="greedy",
        n_generate_sample=settings.n_generate_sample,
        n_evaluate_sample=settings.n_evaluate_sample,
        n_select_sample=settings.n_select_sample,
        prompt_sample="cot",
    )

    try:
        plans, info = solve(args, task, idx=0, to_print=False)
    except Exception as exc:
        logger.exception("ToT graph planner failed")
        # Fallback to sequential graph
        return _generate_fallback_graph(prompt, recall_snippets, agent_catalog)

    best_plan = plans[0] if plans else ""

    try:
        # Parse graph specification from ToT output
        graph_spec = _build_graph_spec_from_plan(
            best_plan, agent_catalog, f"tot_graph_{len(agent_catalog)}_agents", settings
        )

        # Generate fallback assignments for compatibility
        fallback_assignments = _extract_fallback_assignments(graph_spec)

        logger.info(
            "ToT graph planner completed: %d nodes, %d edges, %d parallel groups",
            len(graph_spec.nodes),
            len(graph_spec.edges),
            len(graph_spec.parallel_groups),
        )

    except Exception as e:
        logger.warning(f"Failed to parse graph from ToT output: {e}")
        # Fallback to sequential graph
        return _generate_fallback_graph(prompt, recall_snippets, agent_catalog)

    metadata = {
        "raw_plan": best_plan,
        "tot_usage": gpt_usage(settings.backend),
        "planning_steps": info.get("steps") if isinstance(info, dict) else info,
        "graph_type": "tot_generated",
        "planning_method": "tree_of_thought",
    }

    return {
        "graph_spec": graph_spec,
        "metadata": metadata,
        "fallback_assignments": fallback_assignments,
    }


def _build_graph_problem_statement(
    prompt: str,
    recall_snippets: Sequence[str],
    agent_catalog: Sequence[AgentConfig],
    memory_config: Optional[MemoryConfig] = None,
) -> str:
    """Build enhanced problem statement for graph planning."""
    context_lines = [f"- {snippet}" for snippet in recall_snippets if snippet]
    agents_lines = [
        f"- {agent.name}: {agent.role}. Objetivo: {agent.goal}"
        for agent in agent_catalog
    ]

    memory_block = ""
    if memory_config:
        from .tot_planner import summarize_memory_config

        summary = summarize_memory_config(memory_config)
        if summary:
            memory_block = f"Configuración de memoria:\n{summary}\n\n"

    return (
        f"Solicitud del usuario: {prompt.strip()}\n\n"
        "Contexto recuperado:\n"
        f"{chr(10).join(context_lines) if context_lines else '- (sin contexto relevante)'}\n\n"
        "Agentes disponibles:\n"
        f"{chr(10).join(agents_lines)}\n\n"
        f"{memory_block}"
        "Diseña un grafo StateGraph estructurado que:\n"
        "1. Maximice la eficiencia mediante ejecución paralela cuando sea posible\n"
        "2. Use routing condicional para control de flujo inteligente\n"
        "3. Asigne tareas apropiadamente a agentes especializados\n"
        "4. Maneje dependencias y agregación de resultados\n"
        "5. Proporcione puntos de control y recuperación de errores\n"
    )


def _build_graph_spec_from_plan(
    plan_text: str,
    agent_catalog: Sequence[AgentConfig],
    graph_name: str,
    settings: Optional[GraphPlanningSettings] = None
) -> StateGraphSpec:
    """Build StateGraphSpec from ToT planning output."""

    if settings is None:
        settings = GraphPlanningSettings()

    # Initialize empty graph spec
    graph_spec = StateGraphSpec(
        name=graph_name,
        description=f"StateGraph generated by ToT planner with {len(agent_catalog)} agents",
    )

    # Parse components from plan
    components = {"nodes": [], "edges": [], "parallel_groups": []}
    logger.info(plan_text.strip().split("\n"))

    # Handle both line-separated and concatenated JSON objects
    for line in plan_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Parse multiple JSON objects from the same line
        for component in _parse_json_objects(line):
            comp_type = component.get("component_type", "")

            if comp_type == "node":
                node_spec = _create_node_from_component(component)
                components["nodes"].append(node_spec)
            elif comp_type == "edge":
                edge_spec = _create_edge_from_component(component)
                components["edges"].append(edge_spec)
            elif comp_type == "parallel_group":
                parallel_spec = _create_parallel_from_component(component)
                components["parallel_groups"].append(parallel_spec)

                # Auto-create nodes from parallel_nodes if they don't exist
                parallel_nodes = component.get("parallel_nodes", [])
                if parallel_nodes:
                    _create_nodes_from_parallel_group(
                        parallel_nodes, agent_catalog, components["nodes"]
                    )

    # If no valid components found, create sequential graph from agents
    if not components["nodes"]:
        logger.info("No graph components found, creating sequential fallback")
        return _create_sequential_graph_from_agents(agent_catalog, graph_name)

    # Add parsed components to graph spec
    for node in components["nodes"]:
        graph_spec.add_node(node)

    # Track edge failures for better visibility
    edge_errors = []
    edges_added = 0

    for edge in components["edges"]:
        try:
            graph_spec.add_edge(edge)
            edges_added += 1
        except ValueError as e:
            error_msg = f"Edge {edge.from_node}→{edge.to_node} failed: {e}"
            logger.error(error_msg)
            edge_errors.append({
                'edge': f"{edge.from_node}→{edge.to_node}",
                'error': str(e)
            })

    # Surface critical failures
    if edge_errors:
        logger.warning(f"Edge validation: {edges_added} succeeded, {len(edge_errors)} failed")
        if edges_added == 0 and len(components["edges"]) > 0:
            logger.error("CRITICAL: All edges failed validation - graph will be disconnected")
            for err in edge_errors:
                logger.error(f"  - {err['edge']}: {err['error']}")

    for parallel_group in components["parallel_groups"]:
        try:
            graph_spec.add_parallel_group(parallel_group)
        except ValueError as e:
            logger.warning(f"Skipping invalid parallel group: {e}")

    # Ensure graph has start and end nodes
    _ensure_start_end_nodes(graph_spec)

    # Validate and fix graph if needed
    errors = graph_spec.validate()
    if errors:
        logger.warning(f"Graph validation errors: {errors}")

        # Check fallback configuration
        if settings.strict_validation:
            raise ValueError(f"Graph validation failed with strict_validation enabled: {errors}")

        if settings.preserve_tot_intent:
            raise ValueError(
                f"Graph validation errors detected and preserve_tot_intent=True. "
                f"Errors: {errors}. ToT LLM output will not be modified."
            )

        # Auto-fix some common issues (respects enable_auto_fallback)
        if settings.enable_auto_fallback:
            logger.info("Auto-fallback enabled, attempting to fix graph validation errors")
            _auto_fix_graph(graph_spec)
        else:
            logger.warning("Auto-fallback disabled, graph may have validation errors")

    return graph_spec


def _parse_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Parse multiple JSON objects from a single string.

    Handles cases where JSON objects are concatenated without separators.
    Example: '{"a":1}{"b":2}' -> [{"a":1}, {"b":2}]
    """
    objects = []
    decoder = json.JSONDecoder()
    idx = 0

    while idx < len(text):
        # Skip whitespace
        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx >= len(text):
            break

        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            objects.append(obj)
            idx += end_idx
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON at position {idx}: {e}")
            # ERROR RECOVERY: Skip to next JSON object and continue parsing
            # This prevents edge loss when earlier components have parse errors
            next_obj_start = text.find('{', idx + 1)
            if next_obj_start == -1:
                logger.warning("No more JSON objects found after parse error")
                break
            logger.info(f"Recovering: Skipping to next object at position {next_obj_start}")
            idx = next_obj_start
            continue

    return objects


def _create_node_from_component(component: Dict[str, Any]) -> GraphNodeSpec:
    """Create GraphNodeSpec from component dictionary."""
    node_type_map = {
        "agent": NodeType.AGENT,
        "router": NodeType.ROUTER,
        "parallel": NodeType.PARALLEL,
        "condition": NodeType.CONDITION,
        "start": NodeType.START,
        "end": NodeType.END,
    }

    return GraphNodeSpec(
        name=component.get("name", ""),
        type=node_type_map.get(component.get("type", "agent"), NodeType.AGENT),
        agent=component.get("agent"),
        objective=component.get("objective", ""),
        expected_output=component.get("expected_output", ""),
        tools=component.get("tools", []),
    )


def _create_edge_from_component(component: Dict[str, Any]) -> GraphEdgeSpec:
    """Create GraphEdgeSpec from component dictionary."""
    edge_type_map = {
        "direct": EdgeType.DIRECT,
        "conditional": EdgeType.CONDITIONAL,
        "parallel": EdgeType.PARALLEL,
        "aggregation": EdgeType.AGGREGATION,
    }

    condition = None
    if component.get("condition"):
        cond_data = component["condition"]
        condition = GraphCondition(
            type=cond_data.get("type", "state_check"),
            field=cond_data.get("field"),
            operator=cond_data.get("operator", "equals"),
            value=cond_data.get("value"),
        )

    return GraphEdgeSpec(
        from_node=component.get("from_node", ""),
        to_node=component.get("to_node", ""),
        type=edge_type_map.get(component.get("edge_type", "direct"), EdgeType.DIRECT),
        condition=condition,
    )


def _create_parallel_from_component(component: Dict[str, Any]) -> ParallelGroup:
    """Create ParallelGroup from component dictionary."""
    return ParallelGroup(
        group_id=component.get("name", ""),
        nodes=component.get("parallel_nodes", []),
        aggregation_strategy=component.get("aggregation_strategy", "collect_all"),
    )


def _create_nodes_from_parallel_group(
    parallel_nodes: List[str],
    agent_catalog: Sequence[AgentConfig],
    existing_nodes: List[GraphNodeSpec],
) -> None:
    """
    Auto-create GraphNodeSpec objects from parallel_nodes list.

    Maps node names to agents from catalog and creates missing nodes.
    Modifies existing_nodes list in-place.
    """
    existing_node_names = {node.name for node in existing_nodes}

    for node_name in parallel_nodes:
        # Skip if node already exists
        if node_name in existing_node_names:
            continue

        # Try to match node name to an agent
        matched_agent = _match_node_to_agent(node_name, agent_catalog)

        if matched_agent:
            # Create node with matched agent
            node_spec = GraphNodeSpec(
                name=node_name,
                type=NodeType.AGENT,
                agent=matched_agent.name,
                objective=matched_agent.goal or f"Execute parallel task: {matched_agent.role}",
                expected_output=f"Results from {matched_agent.name}",
                tools=[],
            )
        else:
            # Fallback: use first available agent if no match found
            # This prevents agent=None which would cause compilation errors
            fallback_agent = agent_catalog[0] if agent_catalog else None
            if fallback_agent:
                logger.warning(
                    f"Node '{node_name}' could not be matched to an agent. "
                    f"Using fallback agent '{fallback_agent.name}'. "
                    f"Available agents: {[a.name for a in agent_catalog]}"
                )
                node_spec = GraphNodeSpec(
                    name=node_name,
                    type=NodeType.AGENT,
                    agent=fallback_agent.name,
                    objective=f"Execute task: {node_name}",
                    expected_output=f"Results from {node_name}",
                    tools=[],
                )
            else:
                # No agents available at all - this should never happen but handle it
                logger.error(f"Cannot create node '{node_name}': no agents available in catalog")
                raise ValueError(f"Cannot create agent node '{node_name}': agent catalog is empty")

        existing_nodes.append(node_spec)
        existing_node_names.add(node_name)
        logger.debug(f"Auto-created node '{node_name}' from parallel_group")


def _match_node_to_agent(
    node_name: str, agent_catalog: Sequence[AgentConfig]
) -> Optional[AgentConfig]:
    """
    Match a node name to an agent from the catalog.

    Uses fuzzy matching: exact name, contains, role similarity, or keyword-based matching.
    """
    node_lower = node_name.lower().replace("_", " ").replace("-", " ")

    # Try exact match first
    for agent in agent_catalog:
        if agent.name.lower() == node_lower:
            return agent

    # Try contains match
    for agent in agent_catalog:
        agent_lower = agent.name.lower()
        if agent_lower in node_lower or node_lower in agent_lower:
            return agent

    # Try role-based match
    for agent in agent_catalog:
        role_lower = agent.role.lower() if agent.role else ""
        if role_lower and (role_lower in node_lower or node_lower in role_lower):
            return agent

    # Try keyword-based matching for common task patterns
    keyword_mappings = {
        # Research-related keywords
        "gather": ["researcher", "research"],
        "collect": ["researcher", "research"],
        "find": ["researcher", "research"],
        "search": ["researcher", "research"],
        "investigate": ["researcher", "research"],
        # Analysis-related keywords
        "analyze": ["analyst", "analysis"],
        "review": ["analyst", "analysis"],
        "evaluate": ["analyst", "analysis"],
        "assess": ["analyst", "analysis"],
        # Quality/Testing keywords
        "quality": ["tester", "qa", "quality", "standards"],
        "test": ["tester", "qa", "quality"],
        "validate": ["tester", "qa", "quality"],
        "verify": ["tester", "qa", "quality"],
        "standards": ["standards", "qa", "quality"],
        # Planning keywords
        "plan": ["planner", "planning"],
        "design": ["planner", "architect"],
        "strategy": ["planner", "strategist"],
        # Implementation keywords
        "implement": ["implementer", "developer"],
        "build": ["implementer", "developer"],
        "create": ["implementer", "developer"],
        "develop": ["implementer", "developer"],
        # Writing/Documentation keywords
        "write": ["writer", "documentation"],
        "document": ["writer", "documentation"],
        "report": ["writer", "documentation"],
    }

    # Check if any keyword in node_name matches agent patterns
    for keyword, agent_patterns in keyword_mappings.items():
        if keyword in node_lower:
            for agent in agent_catalog:
                agent_name_lower = agent.name.lower()
                role_lower = agent.role.lower() if agent.role else ""
                # Check if any pattern matches agent name or role
                for pattern in agent_patterns:
                    if pattern in agent_name_lower or pattern in role_lower:
                        logger.debug(
                            f"Matched node '{node_name}' to agent '{agent.name}' via keyword '{keyword}'"
                        )
                        return agent

    return None


def _create_sequential_graph_from_agents(
    agent_catalog: Sequence[AgentConfig], graph_name: str
) -> StateGraphSpec:
    """Create a sequential graph from available agents."""
    assignments = []
    for i, agent in enumerate(agent_catalog):
        assignments.append(
            {
                "agent": agent.name,
                "objective": f"Execute step {i+1} according to agent role: {agent.role}",
                "expected_output": agent.goal or f"Results from {agent.name}",
                "tools": [],
            }
        )

    return create_simple_sequential_graph(graph_name, assignments)


def _ensure_start_end_nodes(graph_spec: StateGraphSpec) -> None:
    """Ensure graph has proper start and end nodes."""
    node_names = {node.name for node in graph_spec.nodes}

    # Add start node if missing
    if "start" not in node_names:
        start_node = GraphNodeSpec(
            name="start",
            type=NodeType.START,
            objective="Initialize workflow",
            expected_output="Ready to begin execution",
        )
        graph_spec.nodes.insert(0, start_node)

    # Add end node if missing
    if "end" not in node_names:
        end_node = GraphNodeSpec(
            name="end",
            type=NodeType.END,
            objective="Complete workflow",
            expected_output="Final results",
        )
        graph_spec.nodes.append(end_node)


def _auto_fix_graph(graph_spec: StateGraphSpec) -> None:
    """
    Apply automatic fixes to graph specification.

    Fixes applied:
    1. Create sequential edges if none exist (handles ToT LLM failures)
    2. Create edges for parallel groups (fan-out/fan-in pattern)
    """
    _auto_create_sequential_edges(graph_spec)  # NEW: Handle missing edges first
    _create_parallel_edges(graph_spec)  # Then handle parallel groups


def _auto_create_sequential_edges(graph_spec: StateGraphSpec) -> None:
    """
    Auto-create sequential edges if graph has no edges.

    Creates a simple sequential flow: start -> node1 -> node2 -> ... -> end
    This is a fallback when the ToT LLM fails to generate edge specifications.

    Args:
        graph_spec: Graph specification to fix
    """
    if graph_spec.edges:
        # Graph already has edges, don't interfere
        return

    # Get agent nodes (exclude start/end)
    agent_nodes = [n for n in graph_spec.nodes if n.type == NodeType.AGENT]

    if not agent_nodes:
        # No agent nodes to connect
        logger.warning("No agent nodes found to create sequential edges")
        return

    logger.info(f"Auto-creating sequential edges for {len(agent_nodes)} agent nodes")

    # Connect start to first agent
    first_edge = GraphEdgeSpec(
        from_node="start",
        to_node=agent_nodes[0].name,
        type=EdgeType.DIRECT,
        label="Begin workflow"
    )
    try:
        graph_spec.add_edge(first_edge)
        logger.debug(f"Added edge: start -> {agent_nodes[0].name}")
    except ValueError as e:
        logger.warning(f"Could not add start edge: {e}")

    # Connect agents sequentially
    for i in range(len(agent_nodes) - 1):
        edge = GraphEdgeSpec(
            from_node=agent_nodes[i].name,
            to_node=agent_nodes[i+1].name,
            type=EdgeType.DIRECT,
            label=f"Step {i+1} to {i+2}"
        )
        try:
            graph_spec.add_edge(edge)
            logger.debug(f"Added edge: {agent_nodes[i].name} -> {agent_nodes[i+1].name}")
        except ValueError as e:
            logger.warning(f"Could not add sequential edge: {e}")

    # Connect last agent to end
    last_edge = GraphEdgeSpec(
        from_node=agent_nodes[-1].name,
        to_node="end",
        type=EdgeType.DIRECT,
        label="Complete workflow"
    )
    try:
        graph_spec.add_edge(last_edge)
        logger.debug(f"Added edge: {agent_nodes[-1].name} -> end")
    except ValueError as e:
        logger.warning(f"Could not add end edge: {e}")

    logger.info(f"Auto-created {len(agent_nodes) + 1} sequential edges")


def _create_parallel_edges(graph_spec: StateGraphSpec) -> None:
    """
    Create fan-out/fan-in edges for parallel groups.

    For each parallel group:
    - Creates edges from START to all parallel nodes (fan-out)
    - Creates edges from all parallel nodes to END (fan-in)
    - Marks edges as PARALLEL type
    """
    if not graph_spec.parallel_groups:
        return

    existing_edges = {(e.from_node, e.to_node) for e in graph_spec.edges}
    node_names = {n.name for n in graph_spec.nodes}

    for parallel_group in graph_spec.parallel_groups:
        # Ensure all nodes in the group exist
        group_nodes = [n for n in parallel_group.nodes if n in node_names]

        if not group_nodes:
            logger.warning(
                f"Parallel group '{parallel_group.group_id}' has no valid nodes"
            )
            continue

        # Create fan-out edges from START to each parallel node
        for node_name in group_nodes:
            edge_key = ("start", node_name)
            if edge_key not in existing_edges:
                edge = GraphEdgeSpec(
                    from_node="start",
                    to_node=node_name,
                    type=EdgeType.PARALLEL,
                    label=f"Parallel execution: {node_name}",
                    description=f"Part of parallel group: {parallel_group.group_id}",
                )
                try:
                    graph_spec.add_edge(edge)
                    existing_edges.add(edge_key)
                    logger.debug(f"Added fan-out edge: start → {node_name}")
                except ValueError as e:
                    logger.warning(f"Could not add fan-out edge: {e}")

        # Create fan-in edges from each parallel node to END
        for node_name in group_nodes:
            edge_key = (node_name, "end")
            if edge_key not in existing_edges:
                edge = GraphEdgeSpec(
                    from_node=node_name,
                    to_node="end",
                    type=EdgeType.AGGREGATION,
                    label=f"Aggregate results from {node_name}",
                    description=f"Part of parallel group: {parallel_group.group_id}",
                )
                try:
                    graph_spec.add_edge(edge)
                    existing_edges.add(edge_key)
                    logger.debug(f"Added fan-in edge: {node_name} → end")
                except ValueError as e:
                    logger.warning(f"Could not add fan-in edge: {e}")


def _extract_fallback_assignments(graph_spec: StateGraphSpec) -> List[Dict[str, Any]]:
    """Extract assignment-style tasks from graph spec for compatibility."""
    assignments = []

    for node in graph_spec.nodes:
        if node.type == NodeType.AGENT and node.agent:
            assignments.append(
                {
                    "agent": node.agent,
                    "objective": node.objective or f"Execute {node.name}",
                    "expected_output": node.expected_output
                    or f"Results from {node.name}",
                    "tags": [],
                }
            )

    return assignments


def _generate_fallback_graph(
    prompt: str, recall_snippets: Sequence[str], agent_catalog: Sequence[AgentConfig]
) -> Dict[str, Any]:
    """Generate fallback sequential graph when ToT fails."""
    logger.info("Generating fallback sequential graph")

    assignments = []
    for i, agent in enumerate(agent_catalog):
        assignments.append(
            {
                "agent": agent.name,
                "objective": f"Address user request: {prompt}",
                "expected_output": agent.goal or f"Results from {agent.name}",
                "tags": ["fallback"],
            }
        )

    graph_spec = create_simple_sequential_graph(
        f"fallback_sequential_{len(agent_catalog)}_agents", assignments
    )

    return {
        "graph_spec": graph_spec,
        "metadata": {
            "graph_type": "sequential_fallback",
            "planning_method": "fallback",
            "reason": "ToT graph planning unavailable or failed",
        },
        "fallback_assignments": assignments,
    }


# Export main functions
__all__ = ["generate_graph_with_tot", "GraphPlanningSettings", "GraphPlanningTask"]
