"""
Main orchestrator class that coordinates all components.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Optional, Any, Callable, Union, Sequence
from pathlib import Path
from textwrap import dedent

from ..integrations.praisonai import PraisonAIAgents, Agent, Task, is_available
from .embedding_utils import get_embedding_dimensions, build_embedder_config

from .config import OrchestratorConfig, AgentConfig, TaskConfig
from .exceptions import (
    OrchestratorError, 
    ConfigurationError, 
    AgentCreationError, 
    TaskExecutionError,
    WorkflowError
)
from ..factories.agent_factory import AgentFactory
from ..factories.task_factory import TaskFactory
from ..memory.memory_manager import MemoryManager
from ..workflow.workflow_engine import WorkflowEngine, WorkflowMetrics


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator class for managing multi-agent workflows.
    
    This class integrates all components (agents, tasks, memory, workflow)
    and provides a high-level API for building and executing orchestrator systems.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration object. If None, creates default configuration.
        """
        self.config = config or OrchestratorConfig()
        
        # Initialize components
        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory()
        self.memory_manager: Optional[MemoryManager] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        
        # State
        self.agents: Dict[str, Agent] = {}
        self.tasks: List[Task] = []
        self.praisonai_system: Optional[PraisonAIAgents] = None
        self.is_initialized = False
        
        # Callbacks
        self.on_workflow_start: Optional[Callable[[], None]] = None
        self.on_workflow_complete: Optional[Callable[[WorkflowMetrics], None]] = None
        self.on_task_start: Optional[Callable[[str, Any], None]] = None
        self.on_task_complete: Optional[Callable[[str, Any], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Initialize if config has agents and tasks
        if self.config.agents and self.config.tasks:
            self.initialize()
    
    def initialize(self) -> None:
        """Initialize all components."""
        try:
            logger.info(f"Initializing orchestrator: {self.config.name}")
            
            # Initialize memory manager
            if self.config.execution_config.memory and not self.memory_manager:
                self.memory_manager = MemoryManager(self.config.memory_config)
                logger.info("Memory manager initialized")

            # Initialize workflow engine
            self._initialize_workflow_engine()

            # Create agents
            self._create_agents()

            # Create tasks (if any declared in config)
            if self.config.tasks:
                self._create_tasks()
                self._create_praisonai_system()
            
            self.is_initialized = True
            logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise OrchestratorError(f"Initialization failed: {str(e)}")
    
    def _create_agents(self) -> None:
        """Create agents from configuration."""
        try:
            self.agents = {}
            enabled_agents = self.config.get_enabled_agents()
            
            for agent_config in enabled_agents:
                agent = self.agent_factory.create_agent(agent_config)
                self.agents[agent_config.name] = agent
                logger.debug(f"Created agent: {agent_config.name}")
            
            logger.info(f"Created {len(self.agents)} agents")
        except Exception as e:
            raise AgentCreationError(f"Failed to create agents: {str(e)}")
    
    def _create_tasks(self) -> None:
        """Create tasks from configuration."""
        try:
            if not self.agents:
                raise TaskExecutionError("No agents available for task creation")
            
            self.tasks = self.task_factory.create_tasks_from_configs(
                self.config.tasks,
                self.agents,
                validate_dependencies=True
            )
            
            logger.info(f"Created {len(self.tasks)} tasks")
        except Exception as e:
            raise TaskExecutionError(f"Failed to create tasks: {str(e)}")
    
    def _create_praisonai_system(self) -> None:
        """Create the PraisonAI system."""
        try:
            # Prepare memory configuration
            memory_config = None
            embedder_config = None
            
            if self.memory_manager:
                # Convert MemoryConfig dataclass to dict expected by PraisonAIAgents
                mc = self.config.memory_config
                memory_config = {
                    "provider": mc.provider.value,
                    "use_embedding": mc.use_embedding,
                    "config": dict(mc.config or {}),
                }
                # Only include file paths if provided to avoid NoneType errors in praisonaiagents
                if mc.short_db:
                    memory_config["short_db"] = mc.short_db
                if mc.long_db:
                    memory_config["long_db"] = mc.long_db
                if mc.rag_db_path:
                    memory_config["rag_db_path"] = mc.rag_db_path
                if mc.embedder:
                    memory_config["embedder"] = {
                        "provider": mc.embedder.provider,
                        "config": mc.embedder.config,
                    }
                # Ensure Mem0 receives embedder/llm inside its internal config
                try:
                    if memory_config and memory_config.get("provider") == "mem0":
                        mem0_cfg = memory_config.setdefault("config", {})
                        # If top-level embedder exists, mirror into mem0 config
                        top_embedder = memory_config.get("embedder")
                        if top_embedder and "embedder" not in mem0_cfg:
                            mem0_cfg["embedder"] = top_embedder
                        # If orchestrator embedder set, prefer that
                        if self.config.embedder:
                            # Compute embedding_dims if missing using centralized utility
                            emb_conf = dict(self.config.embedder.config or {})
                            if "embedding_dims" not in emb_conf:
                                model = emb_conf.get("model", "")
                                emb_conf["embedding_dims"] = get_embedding_dimensions(model)
                            mem0_cfg["embedder"] = {
                                "provider": self.config.embedder.provider,
                                "config": emb_conf,
                            }
                        # Provide a sensible default LLM for Mem0 updates if not provided
                        if "llm" not in mem0_cfg:
                            mem0_cfg["llm"] = {
                                "provider": "openai",
                                "config": {"model": "gpt-4o-mini"}
                            }
                except Exception as e:
                    # Log enrichment failures but don't break creation
                    logger.warning(f"Failed to enrich Mem0 config with embedder: {e}")
                if self.config.embedder:
                    embedder_config = {
                        "provider": self.config.embedder.provider,
                        "config": self.config.embedder.config
                    }
            
            self.praisonai_system = PraisonAIAgents(
                agents=list(self.agents.values()),
                tasks=self.tasks,
                process=self.config.execution_config.process.value,
                verbose=self.config.execution_config.verbose,
                max_iter=self.config.execution_config.max_iter,
                name=self.config.name,
                memory=self.config.execution_config.memory,
                memory_config=memory_config,
                embedder=embedder_config,
                user_id=self.config.execution_config.user_id
            )
            
            logger.info("PraisonAI system created")
        except Exception as e:
            raise OrchestratorError(f"Failed to create PraisonAI system: {str(e)}")
    
    async def run(self) -> Any:
        """
        Run the orchestrator workflow.
        
        Returns:
            The result of the workflow execution.
        """
        if not self.is_initialized:
            raise OrchestratorError("Orchestrator not initialized")
        
        if not self.praisonai_system:
            raise OrchestratorError("PraisonAI system not created")
        
        try:
            logger.info(f"Starting orchestrator workflow: {self.config.name}")
            
            # Call start callback
            if self.on_workflow_start:
                self.on_workflow_start()
            
            # Optional global recall context from memory (inject into all tasks)
            recall_content = None
            try:
                recall_content = self._build_recall_content()
            except Exception as e:
                # Log recall build failures but don't fail the entire run
                logger.warning(f"Recall content build failed, continuing without recall: {e}")
                recall_content = None

            # Use async execution if supported
            if self.config.execution_config.async_execution:
                try:
                    result = await self.praisonai_system.astart(content=recall_content)
                except AttributeError:
                    # Fallback to sync execution
                    result = await asyncio.to_thread(self.praisonai_system.start, recall_content)
            else:
                result = await asyncio.to_thread(self.praisonai_system.start, recall_content)
            
            logger.info("Orchestrator workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            if self.on_error:
                self.on_error(e)
            raise WorkflowError(f"Workflow execution failed: {str(e)}")
    
    def run_sync(self) -> Any:
        """
        Run the orchestrator workflow synchronously.
        
        Returns:
            The result of the workflow execution.
        """
        return asyncio.run(self.run())

    # ------------------------- Memory Recall Helpers -------------------------
    def _build_recall_content(self) -> Optional[str]:
        """Build a global recall context string from memory based on custom_config.

        Expects custom_config.recall like:
          {
            "query": "preferencias|contexto...",
            "limit": 5,
            "agent_id": "mecanico",
            "run_id": "proyecto_A",
            "user_id": "override_user",  # defaults to execution_config.user_id
            "rerank": true
          }
        """
        if not self.memory_manager:
            return None
        recall_cfg = (self.config.custom_config or {}).get("recall")
        if not recall_cfg:
            return None

        query = recall_cfg.get("query")
        if not query:
            return None

        limit = recall_cfg.get("limit", recall_cfg.get("top_k", 5))
        user_id = recall_cfg.get("user_id", self.config.execution_config.user_id)
        agent_id = recall_cfg.get("agent_id")
        run_id = recall_cfg.get("run_id")
        rerank = recall_cfg.get("rerank")

        try:
            # Try provider's filtered retrieval if available
            results = self.memory_manager.retrieve_filtered(
                query,
                limit=limit,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                rerank=rerank,
            )
        except Exception as e:
            # Fallback to simple retrieval if filtered retrieval fails
            logger.warning(f"Filtered memory retrieval failed, falling back to simple retrieval: {e}")
            results = self.memory_manager.retrieve(query, limit=limit)

        if not results:
            return None

        # Format a compact context string with citations
        lines = ["MEMORY RECALL CONTEXT:"]
        for r in results:
            content = (r.get("content") or "").strip()
            if not content:
                continue
            md = r.get("metadata", {}) or {}
            src = md.get("filename") or md.get("sheet_id") or r.get("id")
            if src:
                lines.append(f"- {content} [src: {src}]")
            else:
                lines.append(f"- {content}")
        return "\n".join(lines) if len(lines) > 1 else None
    
    def add_agent(self, agent_config: AgentConfig) -> Agent:
        """
        Add a new agent to the orchestrator.
        
        Args:
            agent_config: Configuration for the new agent.
        
        Returns:
            The created agent.
        """
        try:
            agent = self.agent_factory.create_agent(agent_config)
            self.agents[agent_config.name] = agent
            self.config.agents.append(agent_config)
            
            # Reinitialize if already initialized
            if self.is_initialized:
                self._create_praisonai_system()
            
            logger.info(f"Added agent: {agent_config.name}")
            return agent
        except Exception as e:
            raise AgentCreationError(f"Failed to add agent: {str(e)}")
    
    def add_task(self, task_config: TaskConfig) -> Task:
        """
        Add a new task to the orchestrator.
        
        Args:
            task_config: Configuration for the new task.
        
        Returns:
            The created task.
        """
        try:
            # Check if agent exists
            agent = self.agents.get(task_config.agent_name)
            if not agent:
                raise TaskExecutionError(f"Agent '{task_config.agent_name}' not found")
            
            # Create task
            task = self.task_factory.create_task(task_config, agent)
            self.tasks.append(task)
            self.config.tasks.append(task_config)
            
            # Reinitialize if already initialized
            if self.is_initialized:
                self._create_praisonai_system()
            
            logger.info(f"Added task: {task_config.name}")
            return task
        except Exception as e:
            raise TaskExecutionError(f"Failed to add task: {str(e)}")
    
    def register_agent_template(self, template) -> None:
        """Register a custom agent template."""
        self.agent_factory.register_template(template)
    
    def register_task_template(self, template) -> None:
        """Register a custom task template."""
        self.task_factory.register_template(template)
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def get_task(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        for task in self.tasks:
            if task.name == name:
                return task
        return None
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        if self.workflow_engine:
            return self.workflow_engine.get_workflow_status()
        return {"status": "not_initialized"}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator system."""
        info = {
            "name": self.config.name,
            "initialized": self.is_initialized,
            "agents": {
                "total": len(self.config.agents),
                "enabled": len(self.config.get_enabled_agents()),
                "types": list(self.agent_factory.list_templates())
            },
            "tasks": {
                "total": len(self.config.tasks),
                "types": list(self.task_factory.list_templates())
            },
            "execution": {
                "process": self.config.execution_config.process.value,
                "async": self.config.execution_config.async_execution,
                "memory": self.config.execution_config.memory
            }
        }
        
        if self.memory_manager:
            info["memory"] = self.memory_manager.get_provider_info()
        
        return info

    def export_config(self, file_path: Union[str, Path]) -> None:
        """Export current configuration to file."""
        self.config.save_to_file(file_path)
        logger.info(f"Configuration exported to: {file_path}")

    def create_graph_tool(self, *, user_id: Optional[str] = None, run_id: Optional[str] = None):
        """Create a GraphRAG lookup tool if the memory manager supports it."""
        if not self.memory_manager:
            raise MemoryError("Memory manager not initialized; cannot create graph tool")
        user = user_id or self.config.execution_config.user_id
        run = run_id or self.config.execution_config.user_id
        return self.memory_manager.create_graph_tool(default_user_id=user, default_run_id=run)
    
    def import_config(self, file_path: Union[str, Path]) -> None:
        """Import configuration from file and reinitialize."""
        new_config = OrchestratorConfig.from_file(file_path)
        self.config = new_config
        
        # Reinitialize with new config
        self.is_initialized = False
        self.initialize()
        
        logger.info(f"Configuration imported from: {file_path}")
    
    def merge_config(self, other_config: OrchestratorConfig) -> None:
        """Merge another configuration with current one."""
        self.config = self.config.merge(other_config)
        
        # Reinitialize with merged config
        self.is_initialized = False
        self.initialize()
        
        logger.info("Configuration merged and reinitialized")
    
    def reset(self) -> None:
        """Reset the orchestrator to initial state."""
        self.agents.clear()
        self.tasks.clear()
        self.praisonai_system = None
        self.is_initialized = False
        
        if self.memory_manager:
            self.memory_manager.cleanup()
            self.memory_manager = None
        
        if self.workflow_engine:
            self.workflow_engine.reset_workflow()
            self.workflow_engine = None
        
        logger.info("Orchestrator reset")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.memory_manager:
            self.memory_manager.cleanup()
        
        if self.workflow_engine:
            self.workflow_engine.cancel_workflow()
        
        self.reset()
        logger.info("Orchestrator cleanup completed")
    
    # Callback handlers
    def _on_task_start(self, task_name: str, execution: Any) -> None:
        """Handle task start event."""
        logger.debug(f"Task started: {task_name}")
        if self.on_task_start:
            self.on_task_start(task_name, execution)
    
    def _on_task_complete(self, task_name: str, execution: Any) -> None:
        """Handle task completion event."""
        logger.debug(f"Task completed: {task_name}")
        if self.on_task_complete:
            self.on_task_complete(task_name, execution)
    
    def _on_task_fail(self, task_name: str, execution: Any) -> None:
        """Handle task failure event."""
        logger.warning(f"Task failed: {task_name}")
        if self.on_error:
            self.on_error(execution.error)
    
    def _on_workflow_complete(self, metrics: WorkflowMetrics) -> None:
        """Handle workflow completion event."""
        logger.info(f"Workflow completed - Duration: {metrics.total_duration:.2f}s, "
                   f"Tasks: {metrics.completed_tasks}/{metrics.total_tasks}")
        if self.on_workflow_complete:
            self.on_workflow_complete(metrics)
    
    # Class methods for quick creation
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Orchestrator":
        """Create orchestrator from configuration file."""
        config = OrchestratorConfig.from_file(file_path)
        return cls(config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Orchestrator":
        """Create orchestrator from configuration dictionary."""
        config = OrchestratorConfig.from_dict(config_dict)
        return cls(config)
    
    @classmethod
    def from_env(cls, prefix: str = "ORCHESTRATOR_") -> "Orchestrator":
        """Create orchestrator from environment variables."""
        config = OrchestratorConfig.from_env(prefix)
        return cls(config)

    @classmethod
    def create_default(cls, name: str = "DefaultOrchestrator") -> "Orchestrator":
        """Create orchestrator with default configuration and common agents/tasks."""
        from ..factories.agent_factory import (
            OrchestratorAgentTemplate, ResearcherAgentTemplate, 
            PlannerAgentTemplate, ImplementerAgentTemplate,
            TesterAgentTemplate, WriterAgentTemplate
        )
        from ..factories.task_factory import (
            ResearchTaskTemplate, PlanningTaskTemplate,
            ImplementationTaskTemplate, TestingTaskTemplate,
            DocumentationTaskTemplate
        )
        
        # Create configuration with default agents and tasks
        config = OrchestratorConfig(name=name)
        
        # Add default agents
        agent_templates = [
            OrchestratorAgentTemplate(),
            ResearcherAgentTemplate(), 
            PlannerAgentTemplate(),
            ImplementerAgentTemplate(),
            TesterAgentTemplate(),
            WriterAgentTemplate()
        ]
        
        for template in agent_templates:
            agent_config = template.get_default_config()
            config.agents.append(agent_config)
        
        # Add default tasks with dependencies
        task_configs = [
            TaskConfig(
                name="research_task",
                description="Research best practices for the given topic.",
                expected_output="Comprehensive research findings with citations",
                agent_name="Researcher",
                async_execution=True,
                is_start=True
            ),
            TaskConfig(
                name="plan_task", 
                description="Create an actionable plan based on research.",
                expected_output="Detailed plan with steps and acceptance criteria",
                agent_name="Planner",
                async_execution=True,
                is_start=True
            ),
            TaskConfig(
                name="implement_task",
                description="Implement the solution according to the plan.",
                expected_output="Implementation details and design notes",
                agent_name="Implementer",
                context=["research_task", "plan_task"]
            ),
            TaskConfig(
                name="test_task",
                description="Create test strategy for the implementation.",
                expected_output="Test plan with cases and expected outcomes",
                agent_name="Tester",
                context=["implement_task"]
            ),
            TaskConfig(
                name="document_task",
                description="Create comprehensive documentation.",
                expected_output="Final documentation with summary and details",
                agent_name="Writer",
                context=["research_task", "plan_task", "implement_task", "test_task"]
            )
        ]
        
        config.tasks.extend(task_configs)
        
        return cls(config)

    # ------------------------- Dynamic planning helpers -------------------------
    def _initialize_workflow_engine(self) -> None:
        """Create or reset the workflow engine with standard callbacks."""
        self.workflow_engine = WorkflowEngine(
            process_type=self.config.execution_config.process,
            max_concurrent_tasks=5,
            max_retries=3,
            timeout=None,
        )
        self.workflow_engine.on_task_start = self._on_task_start
        self.workflow_engine.on_task_complete = self._on_task_complete
        self.workflow_engine.on_task_fail = self._on_task_fail
        self.workflow_engine.on_workflow_complete = self._on_workflow_complete

    def plan_from_prompt(
        self,
        prompt: str,
        agent_sequence: Sequence[str],
        *,
        recall_snippets: Optional[Sequence[str]] = None,
        assignments: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Generate task plan dynamically from prompt and selected agents."""
        if not agent_sequence:
            raise ValueError("agent_sequence must contain at least one agent")

        if not self.is_initialized:
            self.initialize()

        enabled_agents = {agent.name: agent for agent in self.config.get_enabled_agents()}
        missing_agents = [name for name in agent_sequence if name not in enabled_agents]
        if missing_agents:
            raise AgentCreationError(
                f"Agents not available in orchestrator configuration: {missing_agents}"
            )

        # Build new task configurations
        dynamic_tasks: List[TaskConfig] = []
        previous_task_name: Optional[str] = None
        assignment_iter = list(assignments or [])
        for index, agent_name in enumerate(agent_sequence):
            agent_cfg = enabled_agents[agent_name]
            task_name = self._generate_task_name(agent_name, index)
            assignment_payload = assignment_iter[index] if index < len(assignment_iter) else {}
            objective = assignment_payload.get("objective") or assignment_payload.get("description")
            deliverable = assignment_payload.get("expected_output") or assignment_payload.get("deliverable")
            tags = assignment_payload.get("tags")
            description = self._compose_task_description(
                agent_cfg,
                prompt,
                recall_snippets,
                task_hint=self._task_type_hint(agent_name),
                assignment_objective=objective,
                assignment_tags=tags,
            )
            expected_output = self._compose_expected_output(agent_cfg, prompt, deliverable=deliverable)
            dynamic_tasks.append(
                TaskConfig(
                    name=task_name,
                    description=description,
                    expected_output=expected_output,
                    agent_name=agent_name,
                    async_execution=False,
                    is_start=index == 0,
                    context=[previous_task_name] if previous_task_name else [],
                )
            )
            previous_task_name = task_name

        # Replace current task configuration and rebuild execution stack
        self.config.tasks = dynamic_tasks
        self.tasks = []

        # Refresh workflow engine state and PraisonAI system
        self._initialize_workflow_engine()
        self._create_tasks()
        if self.workflow_engine and self.tasks:
            self.workflow_engine.add_tasks(self.tasks, self.config.tasks)
        self._create_praisonai_system()

    @staticmethod
    def _generate_task_name(agent_name: str, index: int) -> str:
        slug = agent_name.lower().replace(" ", "_")
        return f"{slug}_task_{index + 1}"

    @staticmethod
    def _compose_task_description(
        agent_cfg: AgentConfig,
        prompt: str,
        recall_snippets: Optional[Sequence[str]] = None,
        *,
        task_hint: Optional[str] = None,
        assignment_objective: Optional[str] = None,
        assignment_tags: Optional[Sequence[str]] = None,
    ) -> str:
        recall_block = ""
        if recall_snippets:
            formatted = "\n".join(f"  - {snippet}" for snippet in recall_snippets if snippet)
            if formatted:
                recall_block = f"\nContexto recuperado:\n{formatted}"

        hint_block = f"\nTipo de tarea sugerido: {task_hint}" if task_hint else ""
        objective_block = f"\nObjetivo específico: {assignment_objective}" if assignment_objective else ""
        tags_block = ""
        if assignment_tags:
            tag_str = ", ".join(str(tag) for tag in assignment_tags if tag)
            if tag_str:
                tags_block = f"\nEtiquetas: {tag_str}"

        return (
            dedent(
                f"""
                Rol del agente: {agent_cfg.role}
                Objetivo base: {agent_cfg.goal}{hint_block}{objective_block}{tags_block}

                Prompt actual:
                {prompt}
                """
            ).strip()
            + recall_block
            + "\n\nSigue tus instrucciones base y entrega un resultado concreto y accionable."
        )

    @staticmethod
    def _compose_expected_output(agent_cfg: AgentConfig, prompt: str, *, deliverable: Optional[str] = None) -> str:
        goal = agent_cfg.goal or "Produce un entregable"
        if deliverable:
            return f"{deliverable} (objetivo base: {goal}). Contexto: {prompt}"
        return f"{goal}. Responde al prompt: {prompt}"

    @staticmethod
    def _task_type_hint(agent_name: str) -> Optional[str]:
        hints = {
            "Researcher": "research",
            "Analyst": "analysis",
            "Planner": "planning",
            "StandardsAgent": "review",
            "QuickResponder": "documentation",
        }
        return hints.get(agent_name)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Orchestrator(name='{self.config.name}', "
                f"agents={len(self.agents)}, tasks={len(self.tasks)}, "
                f"initialized={self.is_initialized})")
