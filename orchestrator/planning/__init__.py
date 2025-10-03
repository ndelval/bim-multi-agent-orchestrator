"""Planning utilities for orchestrator."""

from .tot_planner import (  # noqa: F401
    PlanningSettings,
    generate_plan_with_tot,
    generate_graph_with_tot,
    plan_with_graph_compilation,
    create_graph_from_assignments,
    get_planning_capabilities,
    validate_planning_environment,
    summarize_memory_config,
)

# Graph planning modules (optional imports)
try:
    from .graph_specifications import (  # noqa: F401
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
    GRAPH_SPECS_AVAILABLE = True
except ImportError:
    GRAPH_SPECS_AVAILABLE = False

try:
    from .tot_graph_planner import (  # noqa: F401
        GraphPlanningSettings,
        GraphPlanningTask,
    )
    GRAPH_PLANNER_AVAILABLE = True
except ImportError:
    GRAPH_PLANNER_AVAILABLE = False

try:
    from .graph_compiler import (  # noqa: F401
        GraphCompiler,
        compile_tot_graph,
        compile_and_validate_graph,
    )
    GRAPH_COMPILER_AVAILABLE = True
except ImportError:
    GRAPH_COMPILER_AVAILABLE = False

# Convenience function to check graph planning availability
def is_graph_planning_available() -> bool:
    """Check if full graph planning pipeline is available."""
    return GRAPH_SPECS_AVAILABLE and GRAPH_PLANNER_AVAILABLE and GRAPH_COMPILER_AVAILABLE
