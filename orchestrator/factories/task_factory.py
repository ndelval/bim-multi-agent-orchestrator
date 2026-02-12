"""
Task factory for creating and managing tasks with dependency management.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque

from ..integrations.langchain_integration import (
    LangChainAgent as Agent,
    LangChainTask as Task,
)

from ..core.config import TaskConfig
from ..core.exceptions import TaskExecutionError, DependencyError, TemplateError


logger = logging.getLogger(__name__)


class BaseTaskTemplate(ABC):
    """Base class for task templates."""

    @abstractmethod
    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create a task from configuration."""
        pass

    @abstractmethod
    def get_default_config(self, agent_name: str) -> TaskConfig:
        """Get default configuration for this task type."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type identifier."""
        pass


class ResearchTaskTemplate(BaseTaskTemplate):
    """Template for research tasks."""

    @property
    def task_type(self) -> str:
        return "research"

    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="research_task",
            description=(
                "Research current best practices for the given topic. "
                "Provide comprehensive information with citations."
            ),
            expected_output="Bullet list of best practices with 3–5 citations",
            agent=agent_name,
            async_execution=True,
            is_start=True,
        )

    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create a research task."""
        try:
            return Task(
                name=config.name,
                description=config.description,
                expected_output=config.expected_output,
                agent=agent,
                async_execution=config.async_execution,
                is_start=config.is_start,
                **kwargs,
            )
        except Exception as e:
            raise TaskExecutionError(
                f"Failed to create research task '{config.name}': {str(e)}"
            )


class PlanningTaskTemplate(BaseTaskTemplate):
    """Template for planning tasks."""

    @property
    def task_type(self) -> str:
        return "planning"

    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="plan_task",
            description=(
                "Based on the goal and available information, propose a minimal plan. "
                "Include steps, owners, and acceptance criteria."
            ),
            expected_output="Plan with steps, ownership, criteria, and risks",
            agent=agent_name,
            async_execution=True,
            is_start=True,
        )

    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create a planning task."""
        try:
            return Task(
                name=config.name,
                description=config.description,
                expected_output=config.expected_output,
                agent=agent,
                async_execution=config.async_execution,
                is_start=config.is_start,
                **kwargs,
            )
        except Exception as e:
            raise TaskExecutionError(
                f"Failed to create planning task '{config.name}': {str(e)}"
            )


class ImplementationTaskTemplate(BaseTaskTemplate):
    """Template for implementation tasks."""

    @property
    def task_type(self) -> str:
        return "implementation"

    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="implement_task",
            description=(
                "Implement the solution based on the plan and research. "
                "Describe modules, responsibilities, and interfaces."
            ),
            expected_output="Implementation details covering modules and interfaces",
            agent=agent_name,
            async_execution=False,
        )

    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create an implementation task."""
        try:
            return Task(
                name=config.name,
                description=config.description,
                expected_output=config.expected_output,
                agent=agent,
                async_execution=config.async_execution,
                **kwargs,
            )
        except Exception as e:
            raise TaskExecutionError(
                f"Failed to create implementation task '{config.name}': {str(e)}"
            )


class TestingTaskTemplate(BaseTaskTemplate):
    """Template for testing tasks."""

    @property
    def task_type(self) -> str:
        return "testing"

    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="test_task",
            description=(
                "Propose a lean test strategy to validate the implementation. "
                "List test cases and expected outcomes."
            ),
            expected_output="Test plan with critical cases and pass criteria",
            agent=agent_name,
            async_execution=False,
        )

    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create a testing task."""
        try:
            return Task(
                name=config.name,
                description=config.description,
                expected_output=config.expected_output,
                agent=agent,
                async_execution=config.async_execution,
                **kwargs,
            )
        except Exception as e:
            raise TaskExecutionError(
                f"Failed to create testing task '{config.name}': {str(e)}"
            )


class ReviewTaskTemplate(BaseTaskTemplate):
    """Template for review/decision tasks."""

    @property
    def task_type(self) -> str:
        return "review"

    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="review_task",
            description=(
                "Review all outputs for quality and coherence. "
                "Decide whether to approve or request revisions."
            ),
            expected_output=(
                "Decision: approved | needs_revision with actionable feedback"
            ),
            agent=agent_name,
            task_type="decision",
            async_execution=False,
        )

    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create a review task."""
        try:
            return Task(
                name=config.name,
                description=config.description,
                expected_output=config.expected_output,
                agent=agent,
                task_type=config.task_type,
                condition=config.condition,
                async_execution=config.async_execution,
                **kwargs,
            )
        except Exception as e:
            raise TaskExecutionError(
                f"Failed to create review task '{config.name}': {str(e)}"
            )


class DocumentationTaskTemplate(BaseTaskTemplate):
    """Template for documentation tasks."""

    @property
    def task_type(self) -> str:
        return "documentation"

    def get_default_config(self, agent_name: str) -> TaskConfig:
        return TaskConfig(
            name="writeup_task",
            description=(
                "Create comprehensive documentation combining all outputs. "
                "Include executive summary and technical details."
            ),
            expected_output="Complete documentation with summary and technical details",
            agent=agent_name,
            async_execution=False,
        )

    def create_task(self, config: TaskConfig, agent: Agent, **kwargs) -> Task:
        """Create a documentation task."""
        try:
            return Task(
                name=config.name,
                description=config.description,
                expected_output=config.expected_output,
                agent=agent,
                async_execution=config.async_execution,
                **kwargs,
            )
        except Exception as e:
            raise TaskExecutionError(
                f"Failed to create documentation task '{config.name}': {str(e)}"
            )


class TaskFactory:
    """Factory for creating tasks with dependency management."""

    def __init__(self):
        """Initialize the task factory."""
        self._templates: Dict[str, BaseTaskTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default task templates."""
        default_templates = [
            ResearchTaskTemplate(),
            PlanningTaskTemplate(),
            ImplementationTaskTemplate(),
            TestingTaskTemplate(),
            ReviewTaskTemplate(),
            DocumentationTaskTemplate(),
        ]

        for template in default_templates:
            self.register_template(template)

    def register_template(self, template: BaseTaskTemplate) -> None:
        """Register a task template."""
        if not isinstance(template, BaseTaskTemplate):
            raise TemplateError(f"Template must inherit from BaseTaskTemplate")

        self._templates[template.task_type] = template
        logger.info(f"Registered task template: {template.task_type}")

    def unregister_template(self, task_type: str) -> None:
        """Unregister a task template."""
        if task_type in self._templates:
            del self._templates[task_type]
            logger.info(f"Unregistered task template: {task_type}")

    def get_template(self, task_type: str) -> Optional[BaseTaskTemplate]:
        """Get a task template by type."""
        return self._templates.get(task_type)

    def list_templates(self) -> List[str]:
        """List all registered task templates."""
        return list(self._templates.keys())

    def create_task(
        self,
        config: TaskConfig,
        agent: Agent,
        context_tasks: Optional[List[Task]] = None,
        task_type: Optional[str] = None,
        **kwargs,
    ) -> Task:
        """
        Create a task from configuration.

        Args:
            config: Task configuration
            agent: Agent to assign to the task
            context_tasks: List of context tasks for dependency
            task_type: Override task type (defaults to inferring from config)
            **kwargs: Additional arguments to pass to task creation

        Returns:
            Created task instance
        """
        # Determine task type
        if task_type is None:
            task_type = self._infer_task_type(config)

        # Get template
        template = self.get_template(task_type)
        if template is None:
            raise TaskExecutionError(f"No template found for task type: {task_type}")

        # Prepare task arguments
        task_kwargs = kwargs.copy()
        if context_tasks:
            task_kwargs["context"] = context_tasks

        # Create task
        try:
            task = template.create_task(config, agent, **task_kwargs)
            logger.info(f"Created task '{config.name}' of type '{task_type}'")
            return task
        except Exception as e:
            raise TaskExecutionError(f"Failed to create task '{config.name}': {str(e)}")

    def create_tasks_from_configs(
        self,
        configs: List[TaskConfig],
        agents: Dict[str, Agent],
        validate_dependencies: bool = True,
    ) -> List[Task]:
        """
        Create multiple tasks from configurations with dependency resolution.

        Args:
            configs: List of task configurations
            agents: Dictionary mapping agent names to agent instances
            validate_dependencies: Whether to validate task dependencies

        Returns:
            List of created tasks
        """
        if validate_dependencies:
            self.validate_task_dependencies(configs)

        # Create tasks in dependency order
        task_map: Dict[str, Task] = {}
        tasks = []

        # Sort tasks by dependencies (topological sort)
        sorted_configs = self._topological_sort(configs)

        for config in sorted_configs:
            # Get agent
            agent = agents.get(config.agent)
            if agent is None:
                raise TaskExecutionError(
                    f"Agent '{config.agent}' not found for task '{config.name}'"
                )

            # Get context tasks
            context_tasks = []
            for context_task_name in config.context:
                context_task = task_map.get(context_task_name)
                if context_task is not None:
                    context_tasks.append(context_task)

            # Create task
            task = self.create_task(config, agent, context_tasks)

            # Store for later reference
            task_map[config.name] = task
            tasks.append(task)

        # Set next_tasks relationships
        for config in configs:
            task = task_map[config.name]
            next_tasks = [
                task_map[name] for name in config.next_tasks if name in task_map
            ]
            if next_tasks:
                task.next_tasks = next_tasks

        return tasks

    def _infer_task_type(self, config: TaskConfig) -> str:
        """Infer task type from configuration."""
        description_lower = config.description.lower()
        name_lower = config.name.lower()

        # Map common patterns to task types
        type_patterns = {
            "research": ["research", "search", "gather", "information", "investigate"],
            "planning": ["plan", "design", "strategy", "propose", "architect"],
            "implementation": ["implement", "build", "create", "develop", "code"],
            "testing": ["test", "validate", "verify", "check", "qa"],
            "review": ["review", "decision", "approve", "evaluate", "assess"],
            "documentation": ["document", "write", "summary", "report", "writeup"],
        }

        for task_type, patterns in type_patterns.items():
            for pattern in patterns:
                if pattern in description_lower or pattern in name_lower:
                    return task_type

        # Default task type
        return "implementation"

    def validate_task_dependencies(self, configs: List[TaskConfig]) -> None:
        """
        Validate task dependencies for cycles and missing references.

        Args:
            configs: List of task configurations to validate

        Raises:
            DependencyError: If dependency validation fails
        """
        task_names = {config.name for config in configs}

        # Check for missing references
        for config in configs:
            for context_task in config.context:
                if context_task not in task_names:
                    raise DependencyError(
                        f"Task '{config.name}' references non-existent context task '{context_task}'"
                    )

            for next_task in config.next_tasks:
                if next_task not in task_names:
                    raise DependencyError(
                        f"Task '{config.name}' references non-existent next task '{next_task}'"
                    )

        # Check for circular dependencies
        if self._has_circular_dependencies(configs):
            raise DependencyError("Circular dependencies detected in task graph")

    def _has_circular_dependencies(self, configs: List[TaskConfig]) -> bool:
        """Check if task configurations have circular dependencies."""
        # Build adjacency list
        graph = defaultdict(list)
        for config in configs:
            for context_task in config.context:
                graph[context_task].append(config.name)
            for next_task in config.next_tasks:
                graph[config.name].append(next_task)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check each node
        for config in configs:
            if config.name not in visited:
                if has_cycle(config.name):
                    return True

        return False

    def _topological_sort(self, configs: List[TaskConfig]) -> List[TaskConfig]:
        """Sort tasks in dependency order using topological sort."""
        # Build in-degree count and adjacency list
        in_degree = {config.name: 0 for config in configs}
        graph = defaultdict(list)
        config_map = {config.name: config for config in configs}

        for config in configs:
            for context_task in config.context:
                if context_task in config_map:
                    graph[context_task].append(config.name)
                    in_degree[config.name] += 1

        # Kahn's algorithm
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        sorted_names = []

        while queue:
            current = queue.popleft()
            sorted_names.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Return configs in sorted order
        return [config_map[name] for name in sorted_names]

    def get_dependency_graph(self, configs: List[TaskConfig]) -> Dict[str, List[str]]:
        """Get dependency graph as adjacency list."""
        graph = defaultdict(list)
        for config in configs:
            for context_task in config.context:
                graph[context_task].append(config.name)
            for next_task in config.next_tasks:
                graph[config.name].append(next_task)
        return dict(graph)

    def get_execution_order(self, configs: List[TaskConfig]) -> List[List[str]]:
        """Get task execution order as parallel execution levels."""
        # Validate dependencies first
        self.validate_task_dependencies(configs)

        # Build in-degree count
        in_degree = {config.name: 0 for config in configs}
        config_map = {config.name: config for config in configs}

        for config in configs:
            for context_task in config.context:
                if context_task in config_map:
                    in_degree[config.name] += 1

        # Group tasks by execution level
        levels = []
        remaining = set(config.name for config in configs)

        while remaining:
            # Find tasks with no dependencies
            current_level = [name for name in remaining if in_degree[name] == 0]

            if not current_level:
                raise DependencyError(
                    "Cannot resolve task dependencies - possible circular reference"
                )

            levels.append(current_level)

            # Remove current level tasks and update in-degrees
            for task_name in current_level:
                remaining.remove(task_name)
                config = config_map[task_name]

                # Update in-degrees for tasks that depend on this one
                for other_config in configs:
                    if task_name in other_config.context:
                        in_degree[other_config.name] -= 1

        return levels

    def get_default_config(self, task_type: str, agent_name: str) -> TaskConfig:
        """Get default configuration for a task type."""
        template = self.get_template(task_type)
        if template is None:
            raise TemplateError(f"No template found for task type: {task_type}")

        return template.get_default_config(agent_name)

    def create_default_task(
        self, task_type: str, agent: Agent, name: Optional[str] = None, **kwargs
    ) -> Task:
        """Create a task with default configuration."""
        config = self.get_default_config(task_type, agent.name)
        if name:
            config.name = name

        return self.create_task(config, agent, task_type=task_type, **kwargs)
