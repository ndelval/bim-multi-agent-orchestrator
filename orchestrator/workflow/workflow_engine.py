"""
Workflow engine for task execution with DAG support and parallel processing.
"""

import asyncio
import time
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

# Compatibility layer - supports both PraisonAI and LangChain
try:
    # Try LangChain first (new system)
    from ..integrations.langchain_integration import LangChainTask as Task, LangChainAgent as Agent
    USING_LANGCHAIN = True
except ImportError:
    # Fallback to PraisonAI (legacy system)
    from ..integrations.praisonai import Task, Agent
    USING_LANGCHAIN = False

from ..core.config import TaskConfig, ProcessType
from ..core.exceptions import WorkflowError, TaskExecutionError, DependencyError


logger = logging.getLogger(__name__)
logger.info(f"WorkflowEngine initialized with {'LangChain' if USING_LANGCHAIN else 'PraisonAI'} backend")


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStrategy(str, Enum):
    """Execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MIXED = "mixed"


@dataclass
class TaskExecution:
    """Task execution state."""
    task: Task
    config: TaskConfig
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None
    retries: int = 0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


@dataclass
class WorkflowMetrics:
    """Workflow execution metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_duration: float = 0.0
    parallel_efficiency: float = 0.0


class WorkflowEngine:
    """Engine for executing task workflows with DAG support."""
    
    def __init__(
        self,
        process_type: ProcessType = ProcessType.WORKFLOW,
        max_concurrent_tasks: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None
    ):
        """
        Initialize workflow engine.
        
        Args:
            process_type: Type of process (workflow, sequential, hierarchical)
            max_concurrent_tasks: Maximum number of concurrent tasks
            max_retries: Maximum retries for failed tasks
            retry_delay: Delay between retries in seconds
            timeout: Timeout for individual tasks in seconds
        """
        self.process_type = process_type
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        self.executions: Dict[str, TaskExecution] = {}
        self.execution_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.metrics = WorkflowMetrics()
        
        # Execution state
        self.is_running = False
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Callbacks
        self.on_task_start: Optional[Callable[[str, TaskExecution], None]] = None
        self.on_task_complete: Optional[Callable[[str, TaskExecution], None]] = None
        self.on_task_fail: Optional[Callable[[str, TaskExecution], None]] = None
        self.on_workflow_complete: Optional[Callable[[WorkflowMetrics], None]] = None
    
    def add_tasks(self, tasks: List[Task], configs: List[TaskConfig]) -> None:
        """Add tasks to the workflow."""
        if len(tasks) != len(configs):
            raise WorkflowError("Number of tasks and configs must match")
        
        # Clear previous state
        self.executions.clear()
        self.execution_graph.clear()
        self.reverse_graph.clear()
        
        # Create task executions
        for task, config in zip(tasks, configs):
            execution = TaskExecution(
                task=task,
                config=config,
                dependencies=set(config.context),
                dependents=set(config.next_tasks)
            )
            self.executions[config.name] = execution
        
        # Build execution graph
        self._build_execution_graph()
        
        # Validate graph
        self._validate_execution_graph()
        
        # Update metrics
        self.metrics.total_tasks = len(tasks)
        logger.info(f"Added {len(tasks)} tasks to workflow")
    
    def _build_execution_graph(self) -> None:
        """Build the execution dependency graph."""
        for task_name, execution in self.executions.items():
            # Add dependencies from context
            for dep_name in execution.config.context:
                if dep_name in self.executions:
                    self.execution_graph[dep_name].add(task_name)
                    self.reverse_graph[task_name].add(dep_name)
            
            # Add dependencies from next_tasks
            for next_task in execution.config.next_tasks:
                if next_task in self.executions:
                    self.execution_graph[task_name].add(next_task)
                    self.reverse_graph[next_task].add(task_name)
    
    def _validate_execution_graph(self) -> None:
        """Validate the execution graph for cycles and missing dependencies."""
        # Check for missing dependencies
        all_task_names = set(self.executions.keys())
        for task_name, execution in self.executions.items():
            for dep_name in execution.dependencies:
                if dep_name not in all_task_names:
                    raise DependencyError(f"Task '{task_name}' depends on non-existent task '{dep_name}'")
            
            for next_task in execution.dependents:
                if next_task not in all_task_names:
                    raise DependencyError(f"Task '{task_name}' references non-existent next task '{next_task}'")
        
        # Check for circular dependencies
        if self._has_cycles():
            raise DependencyError("Circular dependencies detected in workflow")
    
    def _has_cycles(self) -> bool:
        """Check if the execution graph has cycles using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.execution_graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_name in self.executions:
            if task_name not in visited:
                if dfs(task_name):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[List[str]]:
        """Get task execution order as parallel execution levels."""
        in_degree = {name: len(self.reverse_graph[name]) for name in self.executions}
        levels = []
        remaining = set(self.executions.keys())
        
        while remaining:
            # Find tasks with no dependencies
            current_level = [name for name in remaining if in_degree[name] == 0]
            
            if not current_level:
                raise DependencyError("Cannot resolve task dependencies")
            
            levels.append(current_level)
            
            # Remove current level tasks and update in-degrees
            for task_name in current_level:
                remaining.remove(task_name)
                for dependent in self.execution_graph[task_name]:
                    in_degree[dependent] -= 1
        
        return levels
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        
        for task_name, execution in self.executions.items():
            if execution.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.executions[dep].status == TaskStatus.COMPLETED
                    for dep in execution.dependencies
                    if dep in self.executions
                )
                
                if deps_completed:
                    execution.status = TaskStatus.READY
                    ready_tasks.append(task_name)
        
        return ready_tasks
    
    async def execute_workflow(self) -> Dict[str, Any]:
        """Execute the entire workflow."""
        if self.is_running:
            raise WorkflowError("Workflow is already running")
        
        try:
            self.is_running = True
            self.metrics.start_time = time.time()
            
            # Reset state
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            
            # Choose execution strategy
            strategy = self._determine_execution_strategy()
            
            if strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential()
            elif strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel()
            else:
                result = await self._execute_mixed()
            
            # Calculate metrics
            self.metrics.end_time = time.time()
            self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
            self.metrics.completed_tasks = len(self.completed_tasks)
            self.metrics.failed_tasks = len(self.failed_tasks)
            self.metrics.skipped_tasks = self.metrics.total_tasks - self.metrics.completed_tasks - self.metrics.failed_tasks
            
            # Calculate parallel efficiency
            total_task_time = sum(
                (exec.end_time or 0) - (exec.start_time or 0)
                for exec in self.executions.values()
                if exec.end_time and exec.start_time
            )
            self.metrics.parallel_efficiency = (
                total_task_time / self.metrics.total_duration
                if self.metrics.total_duration > 0 else 0
            )
            
            # Call completion callback
            if self.on_workflow_complete:
                self.on_workflow_complete(self.metrics)
            
            logger.info(f"Workflow completed in {self.metrics.total_duration:.2f}s")
            return result
            
        finally:
            self.is_running = False
    
    def _determine_execution_strategy(self) -> ExecutionStrategy:
        """Determine the best execution strategy."""
        if self.process_type == ProcessType.SEQUENTIAL:
            return ExecutionStrategy.SEQUENTIAL
        
        # Check if any tasks can run in parallel
        execution_levels = self.get_execution_order()
        has_parallel_tasks = any(len(level) > 1 for level in execution_levels)
        
        if has_parallel_tasks and self.max_concurrent_tasks > 1:
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.SEQUENTIAL
    
    async def _execute_sequential(self) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        logger.info("Executing workflow sequentially")
        
        execution_levels = self.get_execution_order()
        results = {}
        
        for level in execution_levels:
            for task_name in level:
                try:
                    result = await self._execute_task(task_name)
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {str(e)}")
                    if not self._should_continue_on_failure(task_name):
                        raise WorkflowError(f"Workflow stopped due to critical task failure: {task_name}")
        
        return results
    
    async def _execute_parallel(self) -> Dict[str, Any]:
        """Execute tasks in parallel where possible."""
        logger.info("Executing workflow in parallel")
        
        execution_levels = self.get_execution_order()
        results = {}
        
        for level in execution_levels:
            if len(level) == 1:
                # Single task, execute directly
                task_name = level[0]
                try:
                    result = await self._execute_task(task_name)
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {str(e)}")
                    if not self._should_continue_on_failure(task_name):
                        raise WorkflowError(f"Workflow stopped due to critical task failure: {task_name}")
            else:
                # Multiple tasks, execute in parallel
                semaphore = asyncio.Semaphore(min(self.max_concurrent_tasks, len(level)))
                
                async def execute_with_semaphore(task_name: str):
                    async with semaphore:
                        return await self._execute_task(task_name)
                
                # Create tasks for parallel execution
                tasks = [execute_with_semaphore(task_name) for task_name in level]
                
                # Execute and gather results
                level_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for task_name, result in zip(level, level_results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {task_name} failed: {str(result)}")
                        if not self._should_continue_on_failure(task_name):
                            raise WorkflowError(f"Workflow stopped due to critical task failure: {task_name}")
                    else:
                        results[task_name] = result
        
        return results
    
    async def _execute_mixed(self) -> Dict[str, Any]:
        """Execute workflow with mixed strategy (adaptive)."""
        logger.info("Executing workflow with mixed strategy")
        return await self._execute_parallel()  # Use parallel as default mixed strategy
    
    async def _execute_task(self, task_name: str) -> Any:
        """Execute a single task with retries and error handling."""
        execution = self.executions[task_name]
        
        for attempt in range(self.max_retries + 1):
            try:
                # Update status
                execution.status = TaskStatus.RUNNING
                execution.start_time = time.time()
                execution.retries = attempt
                self.running_tasks.add(task_name)
                
                # Call start callback
                if self.on_task_start:
                    self.on_task_start(task_name, execution)
                
                logger.debug(f"Executing task: {task_name} (attempt {attempt + 1})")
                
                # Execute the task - handle both LangChain and PraisonAI interfaces
                if USING_LANGCHAIN:
                    # LangChain task execution
                    context_data = {
                        "task_name": task_name,
                        "config": execution.config,
                        "dependencies": [
                            self.executions[dep].result for dep in execution.dependencies 
                            if dep in self.executions and self.executions[dep].result
                        ]
                    }
                    
                    if self.timeout:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(execution.task.execute, context_data),
                            timeout=self.timeout
                        )
                    else:
                        result = await asyncio.to_thread(execution.task.execute, context_data)
                else:
                    # PraisonAI task execution (legacy)
                    if hasattr(execution.task, 'aexecute'):
                        # Async execution
                        if self.timeout:
                            result = await asyncio.wait_for(
                                execution.task.aexecute(),
                                timeout=self.timeout
                            )
                        else:
                            result = await execution.task.aexecute()
                    else:
                        # Sync execution in thread pool
                        if self.timeout:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(execution.task.execute),
                                timeout=self.timeout
                            )
                        else:
                            result = await asyncio.to_thread(execution.task.execute)
                
                # Task completed successfully
                execution.status = TaskStatus.COMPLETED
                execution.end_time = time.time()
                execution.result = result
                self.running_tasks.discard(task_name)
                self.completed_tasks.add(task_name)
                
                # Call completion callback
                if self.on_task_complete:
                    self.on_task_complete(task_name, execution)
                
                logger.info(f"Task completed: {task_name} in {execution.end_time - execution.start_time:.2f}s")
                return result
                
            except Exception as e:
                execution.error = e
                logger.warning(f"Task {task_name} attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    # Wait before retry
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # All retries exhausted
                    execution.status = TaskStatus.FAILED
                    execution.end_time = time.time()
                    self.running_tasks.discard(task_name)
                    self.failed_tasks.add(task_name)
                    
                    # Call failure callback
                    if self.on_task_fail:
                        self.on_task_fail(task_name, execution)
                    
                    raise TaskExecutionError(f"Task {task_name} failed after {self.max_retries + 1} attempts: {str(e)}")
    
    def _should_continue_on_failure(self, task_name: str) -> bool:
        """Determine if workflow should continue after a task failure."""
        execution = self.executions[task_name]
        
        # Check if any dependents are critical
        for dependent_name in execution.dependents:
            if dependent_name in self.executions:
                dependent_execution = self.executions[dependent_name]
                # If dependent is a decision task or final task, consider it critical
                if (dependent_execution.config.task_type == "decision" or 
                    not dependent_execution.dependents):
                    return False
        
        return True
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "is_running": self.is_running,
            "total_tasks": len(self.executions),
            "pending_tasks": len([e for e in self.executions.values() if e.status == TaskStatus.PENDING]),
            "ready_tasks": len([e for e in self.executions.values() if e.status == TaskStatus.READY]),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "metrics": {
                "start_time": self.metrics.start_time,
                "current_time": time.time() if self.is_running else self.metrics.end_time,
                "duration": (time.time() - self.metrics.start_time) if self.is_running and self.metrics.start_time else self.metrics.total_duration
            }
        }
    
    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_name not in self.executions:
            return None
        
        execution = self.executions[task_name]
        return {
            "name": task_name,
            "status": execution.status.value,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "duration": (execution.end_time - execution.start_time) if execution.end_time and execution.start_time else None,
            "retries": execution.retries,
            "error": str(execution.error) if execution.error else None,
            "dependencies": list(execution.dependencies),
            "dependents": list(execution.dependents)
        }
    
    def cancel_workflow(self) -> None:
        """Cancel the running workflow."""
        if self.is_running:
            logger.info("Cancelling workflow execution")
            self.is_running = False
            
            # Mark running tasks as failed
            for task_name in self.running_tasks:
                execution = self.executions[task_name]
                execution.status = TaskStatus.FAILED
                execution.end_time = time.time()
                execution.error = Exception("Workflow cancelled")
    
    def reset_workflow(self) -> None:
        """Reset workflow state for re-execution."""
        if self.is_running:
            raise WorkflowError("Cannot reset running workflow")
        
        # Reset task executions
        for execution in self.executions.values():
            execution.status = TaskStatus.PENDING
            execution.start_time = None
            execution.end_time = None
            execution.result = None
            execution.error = None
            execution.retries = 0
        
        # Reset state
        self.running_tasks.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        
        # Reset metrics
        self.metrics = WorkflowMetrics(total_tasks=len(self.executions))
        
        logger.info("Workflow reset for re-execution")
    
    def export_execution_graph(self) -> Dict[str, Any]:
        """Export execution graph for visualization."""
        nodes = []
        edges = []
        
        for task_name, execution in self.executions.items():
            nodes.append({
                "id": task_name,
                "label": task_name,
                "status": execution.status.value,
                "agent": execution.config.agent_name,
                "async": execution.config.async_execution,
                "start": execution.config.is_start
            })
            
            for dependent in execution.dependents:
                edges.append({
                    "from": task_name,
                    "to": dependent,
                    "type": "dependency"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "total_duration": self.metrics.total_duration,
                "parallel_efficiency": self.metrics.parallel_efficiency
            }
        }