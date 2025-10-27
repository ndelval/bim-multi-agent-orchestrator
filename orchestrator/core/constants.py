"""
Configuration constants for the orchestrator system.

This module contains system-wide constants to avoid magic numbers in the codebase.
"""

# Execution limits
MAX_EXECUTION_DEPTH = 100  # Maximum execution depth before recursion limit
MAX_GRAPH_ITERATIONS = 100  # Maximum iterations for graph execution
DEFAULT_MAX_ITERATIONS = 25  # Default maximum iterations for agent execution
DEPTH_WARNING_THRESHOLD = 50  # Warn when execution depth exceeds this value

# Memory and retrieval settings
DEFAULT_MEMORY_LIMIT = 10  # Default number of memory results to retrieve
GRAPH_CANDIDATE_LIMIT = 5  # Maximum graph candidates to consider
MIN_SIMILARITY_SCORE = 0.5  # Minimum similarity score for matches (50%)

# Workflow settings
DEFAULT_MAX_CONCURRENT_TASKS = 5  # Default maximum concurrent tasks in workflow
EXECUTION_PATH_DISPLAY_LIMIT = 5  # Number of recent nodes to display in execution path

# Planning settings
TOT_MAX_BRANCHES = 4  # Maximum branches for Tree-of-Thought planning
DEFAULT_PLANNING_MAX_STEPS = 5  # Default maximum steps for planning workflows
