"""Mock engineering agent configuration for orchestration scaffolding."""
from __future__ import annotations

from typing import Dict, List

from orchestrator.core.config import (
    AgentConfig,
    ExecutionConfig,
    MemoryConfig,
    MemoryProvider,
    OrchestratorConfig,
    ProcessType,
)

MOCK_INSTRUCTION_TEMPLATE = (
    "You are a mock specialist used for integration scaffolding. "
    "Always answer with the exact string '{placeholder}' and nothing else."
)

SPECIALIST_DEFINITIONS: List[Dict[str, str]] = [
    {
        "name": "pythoncad_parser",
        "role": "CAD Intake Specialist",
        "goal": "Produce structured CAD entities for downstream reasoning.",
        "placeholder": "CAD_PARSER_MOCK_OUTPUT",
    },
    {
        "name": "pythongeometric_analyzer",
        "role": "Geometric Analysis Specialist",
        "goal": "Evaluate dimensions, tolerances and geometric relationships.",
        "placeholder": "GEOMETRY_ANALYZER_MOCK_OUTPUT",
    },
    {
        "name": "pythonspatial_intelligence",
        "role": "Spatial Intelligence Specialist",
        "goal": "Assess spatial layouts, visibility and pathing constraints.",
        "placeholder": "SPATIAL_INTELLIGENCE_MOCK_OUTPUT",
    },
    {
        "name": "pythonconstraint_solver",
        "role": "Constraint Solver",
        "goal": "Resolve clashes and constraint networks.",
        "placeholder": "CONSTRAINT_SOLVER_MOCK_OUTPUT",
    },
    {
        "name": "pythonstandards_validator",
        "role": "Standards & Compliance Analyst",
        "goal": "Validate diseños contra normas y regulaciones aplicables.",
        "placeholder": "STANDARDS_VALIDATOR_MOCK_OUTPUT",
    },
    {
        "name": "pythonmaterials_bom",
        "role": "Materials & BOM Analyst",
        "goal": "Resumir hallazgos de materiales y coste.",
        "placeholder": "MATERIALS_BOM_MOCK_OUTPUT",
    },
    {
        "name": "pythonsimulation_orchestrator",
        "role": "Simulation Planner",
        "goal": "Recomendar escenarios y secuencia de simulaciones.",
        "placeholder": "SIMULATION_ORCHESTRATOR_MOCK_OUTPUT",
    },
    {
        "name": "pythonrisk_safety",
        "role": "Risk & Safety Analyst",
        "goal": "Identificar riesgos y proponer mitigaciones.",
        "placeholder": "RISK_SAFETY_MOCK_OUTPUT",
    },
    {
        "name": "pythondesign_optimizer",
        "role": "Design Optimisation Specialist",
        "goal": "Equilibrar trade-offs de diseño multiobjetivo.",
        "placeholder": "DESIGN_OPTIMIZER_MOCK_OUTPUT",
    },
    {
        "name": "pythonreport_synthesizer",
        "role": "Report Synthesiser",
        "goal": "Ensamblar entregables finales.",
        "placeholder": "REPORT_SYNTHESIZER_MOCK_OUTPUT",
    },
    {
        "name": "pythonhuman_interface",
        "role": "Human Interface Agent",
        "goal": "Explicar el estado del sistema a los stakeholders.",
        "placeholder": "HUMAN_INTERFACE_MOCK_OUTPUT",
    },
]

ORCHESTRATOR_INSTRUCTIONS = (
    "You are the lead engineering orchestrator. Analyse the project context, use MEMORY RECALL CONTEXT when provided, "
    "and decide whether the user needs a direct answer or a structured engineering plan."
    "If the user asks for information that is already known, respond concisely citing MEMORY RECALL CONTEXT."
    "Only produce detailed plans or task graphs when explicitly requested."
    "Specialist agents available: "
    + ", ".join(defn["name"] for defn in SPECIALIST_DEFINITIONS)
    + "."
)


def _mock_agent(name: str, role: str, goal: str, placeholder: str, tools: List[str] | None = None) -> AgentConfig:
    return AgentConfig(
        name=name,
        role=role,
        goal=goal,
        backstory="Placeholder agent awaiting full implementation.",
        instructions=MOCK_INSTRUCTION_TEMPLATE.format(placeholder=placeholder),
        tools=tools or [],
    )


def create_mock_engineering_config() -> OrchestratorConfig:
    """Return an orchestrator configuration with one real orchestrator agent and mock specialists."""

    placeholders: Dict[str, str] = {d["name"]: d["placeholder"] for d in SPECIALIST_DEFINITIONS}

    orchestrator_agent = AgentConfig(
        name="engineering_orchestrator",
        role="Lead Engineering Orchestrator",
        goal="Plan and coordinate all engineering work streams.",
        backstory=(
            "Seasoned systems engineer capable of combining CAD insights, regulatory knowledge and project objectives "
            "into coherent execution plans."
        ),
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        tools=[],
    )

    specialist_agents = [
        _mock_agent(
            name=definition["name"],
            role=definition["role"],
            goal=definition["goal"],
            placeholder=definition["placeholder"],
        )
        for definition in SPECIALIST_DEFINITIONS
    ]

    execution_config = ExecutionConfig(
        process=ProcessType.WORKFLOW,
        verbose=1,
        max_iterations=4,
        memory=False,
        user_id="mock-user",
        async_execution=False,
    )

    memory_config = MemoryConfig(
        provider=MemoryProvider.RAG,
        use_embedding=False,
        config={},
        short_db=None,
        long_db=None,
        rag_db_path=None,
        embedder=None,
    )

    return OrchestratorConfig(
        name="MockEngineeringOrchestrator",
        memory_config=memory_config,
        execution_config=execution_config,
        agents=[orchestrator_agent, *specialist_agents],
        tasks=[],
        embedder=None,
        custom_config={
            "mock": True,
            "mock_placeholders": placeholders,
        },
    )


__all__ = ["create_mock_engineering_config"]
