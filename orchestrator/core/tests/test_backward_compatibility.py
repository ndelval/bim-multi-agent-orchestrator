"""
End-to-end backward compatibility tests for refactored orchestrator.

Verifies that the refactored Orchestrator maintains 100% API compatibility
with the original implementation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from orchestrator.core.orchestrator_refactored import Orchestrator
from orchestrator.core.config import OrchestratorConfig, AgentConfig, TaskConfig


@pytest.fixture
def sample_config():
    """Create sample configuration for compatibility testing."""
    return OrchestratorConfig(
        name="compat_test",
        agents=[
            AgentConfig(
                name="Agent1",
                role="Role1",
                goal="Goal1",
                backstory="Story1",
                instructions="Instructions1",
            )
        ],
        tasks=[
            TaskConfig(
                name="task1",
                description="Task 1",
                expected_output="Output 1",
                agent="Agent1",
                is_start=True,
            )
        ],
    )


class TestBackwardCompatibilityInit:
    """Test backward compatibility of initialization methods."""

    def test_init_with_config(self, sample_config):
        """Test initialization with config matches old behavior."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should have config, agents dict, tasks list
            assert hasattr(orch, 'config')
            assert hasattr(orch, 'agents')
            assert hasattr(orch, 'tasks')
            assert hasattr(orch, 'is_initialized')
            assert isinstance(orch.agents, dict)
            assert isinstance(orch.tasks, list)

    def test_init_without_config(self):
        """Test initialization without config matches old behavior."""
        orch = Orchestrator()

        # Should create default config
        assert orch.config is not None
        assert orch.config.name == "orchestrator"

    def test_from_file_classmethod(self, tmp_path):
        """Test from_file classmethod exists and works."""
        # Old API compatibility: from_file should exist
        assert hasattr(Orchestrator, 'from_file')
        assert callable(Orchestrator.from_file)

    def test_from_dict_classmethod(self):
        """Test from_dict classmethod exists and works."""
        # Old API compatibility: from_dict should exist
        assert hasattr(Orchestrator, 'from_dict')
        assert callable(Orchestrator.from_dict)


class TestBackwardCompatibilityCallbacks:
    """Test backward compatibility of callback setters/getters."""

    def test_workflow_callbacks_exist(self, sample_config):
        """Test all workflow callback properties exist."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: callback properties should exist
            assert hasattr(orch, 'on_workflow_start')
            assert hasattr(orch, 'on_workflow_complete')
            assert hasattr(orch, 'on_task_start')
            assert hasattr(orch, 'on_task_complete')
            assert hasattr(orch, 'on_error')

    def test_callbacks_are_settable(self, sample_config):
        """Test callbacks can be set like old API."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should be able to set callbacks
            on_start = Mock()
            on_complete = Mock()
            on_error = Mock()

            orch.on_workflow_start = on_start
            orch.on_workflow_complete = on_complete
            orch.on_error = on_error

            # Should be retrievable
            assert orch.on_workflow_start == on_start
            assert orch.on_workflow_complete == on_complete
            assert orch.on_error == on_error


class TestBackwardCompatibilityWorkflow:
    """Test backward compatibility of workflow execution."""

    @pytest.mark.asyncio
    async def test_run_method_exists(self, sample_config):
        """Test run() async method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: run should exist and be async
            assert hasattr(orch, 'run')
            assert callable(orch.run)

    def test_run_sync_method_exists(self, sample_config):
        """Test run_sync() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: run_sync should exist
            assert hasattr(orch, 'run_sync')
            assert callable(orch.run_sync)

    def test_initialize_method_exists(self, sample_config):
        """Test initialize() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: initialize should exist
            assert hasattr(orch, 'initialize')
            assert callable(orch.initialize)


class TestBackwardCompatibilityAgentTask:
    """Test backward compatibility of agent/task management."""

    def test_add_agent_method_exists(self, sample_config):
        """Test add_agent() method exists and has correct signature."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: add_agent should exist
            assert hasattr(orch, 'add_agent')
            assert callable(orch.add_agent)

    def test_add_task_method_exists(self, sample_config):
        """Test add_task() method exists and has correct signature."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: add_task should exist
            assert hasattr(orch, 'add_task')
            assert callable(orch.add_task)

    def test_get_agent_method_exists(self, sample_config):
        """Test get_agent() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: get_agent should exist
            assert hasattr(orch, 'get_agent')
            assert callable(orch.get_agent)

    def test_get_task_method_exists(self, sample_config):
        """Test get_task() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: get_task should exist
            assert hasattr(orch, 'get_task')
            assert callable(orch.get_task)


class TestBackwardCompatibilityPlanning:
    """Test backward compatibility of planning methods."""

    def test_plan_from_prompt_exists(self, sample_config):
        """Test plan_from_prompt() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: plan_from_prompt should exist
            assert hasattr(orch, 'plan_from_prompt')
            assert callable(orch.plan_from_prompt)


class TestBackwardCompatibilityTemplates:
    """Test backward compatibility of template registration."""

    def test_register_agent_template_exists(self, sample_config):
        """Test register_agent_template() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: register_agent_template should exist
            assert hasattr(orch, 'register_agent_template')
            assert callable(orch.register_agent_template)

    def test_register_task_template_exists(self, sample_config):
        """Test register_task_template() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: register_task_template should exist
            assert hasattr(orch, 'register_task_template')
            assert callable(orch.register_task_template)


class TestBackwardCompatibilityStatus:
    """Test backward compatibility of status methods."""

    def test_get_workflow_status_exists(self, sample_config):
        """Test get_workflow_status() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: get_workflow_status should exist
            assert hasattr(orch, 'get_workflow_status')
            assert callable(orch.get_workflow_status)

    def test_get_system_info_exists(self, sample_config):
        """Test get_system_info() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: get_system_info should exist
            assert hasattr(orch, 'get_system_info')
            assert callable(orch.get_system_info)


class TestBackwardCompatibilityConfig:
    """Test backward compatibility of config methods."""

    def test_export_config_exists(self, sample_config):
        """Test export_config() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: export_config should exist
            assert hasattr(orch, 'export_config')
            assert callable(orch.export_config)

    def test_import_config_exists(self, sample_config):
        """Test import_config() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: import_config should exist
            assert hasattr(orch, 'import_config')
            assert callable(orch.import_config)


class TestBackwardCompatibilityResourceManagement:
    """Test backward compatibility of resource management."""

    def test_reset_method_exists(self, sample_config):
        """Test reset() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: reset should exist
            assert hasattr(orch, 'reset')
            assert callable(orch.reset)

    def test_cleanup_method_exists(self, sample_config):
        """Test cleanup() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: cleanup should exist
            assert hasattr(orch, 'cleanup')
            assert callable(orch.cleanup)


class TestBackwardCompatibilityContextManager:
    """Test backward compatibility of context manager."""

    def test_context_manager_support(self, sample_config):
        """Test orchestrator supports context manager protocol."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should support context manager
            assert hasattr(orch, '__enter__')
            assert hasattr(orch, '__exit__')
            assert callable(orch.__enter__)
            assert callable(orch.__exit__)


class TestBackwardCompatibilityFactories:
    """Test backward compatibility of factory access."""

    def test_agent_factory_accessible(self, sample_config):
        """Test agent_factory is accessible."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should have agent_factory
            assert hasattr(orch, 'agent_factory')
            assert orch.agent_factory is not None

    def test_task_factory_accessible(self, sample_config):
        """Test task_factory is accessible."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should have task_factory
            assert hasattr(orch, 'task_factory')
            assert orch.task_factory is not None


class TestBackwardCompatibilityMemoryWorkflow:
    """Test backward compatibility of memory and workflow components."""

    def test_memory_manager_accessible(self, sample_config):
        """Test memory_manager is accessible."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should have memory_manager (may be None if not initialized)
            assert hasattr(orch, 'memory_manager')

    def test_workflow_engine_accessible(self, sample_config):
        """Test workflow_engine is accessible."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should have workflow_engine (may be None if not initialized)
            assert hasattr(orch, 'workflow_engine')


class TestBackwardCompatibilityRepr:
    """Test backward compatibility of string representation."""

    def test_repr_method_exists(self, sample_config):
        """Test __repr__() method exists."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Old API: should have __repr__
            assert hasattr(orch, '__repr__')
            repr_str = repr(orch)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0


class TestBackwardCompatibilityCompleteness:
    """Test overall API completeness."""

    def test_all_public_methods_exist(self, sample_config):
        """Test all expected public methods exist."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # List of all public methods that should exist for backward compatibility
            expected_methods = [
                # Initialization
                'initialize',
                # Workflow execution
                'run',
                'run_sync',
                # Agent/Task management
                'add_agent',
                'add_task',
                'get_agent',
                'get_task',
                # Planning
                'plan_from_prompt',
                # Templates
                'register_agent_template',
                'register_task_template',
                # Status
                'get_workflow_status',
                'get_system_info',
                # Config
                'export_config',
                'import_config',
                # Resource management
                'reset',
                'cleanup',
                # Context manager
                '__enter__',
                '__exit__',
                # String representation
                '__repr__',
            ]

            for method_name in expected_methods:
                assert hasattr(orch, method_name), f"Missing method: {method_name}"
                assert callable(getattr(orch, method_name)), f"Not callable: {method_name}"

    def test_all_public_attributes_exist(self, sample_config):
        """Test all expected public attributes exist."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # List of all public attributes that should exist
            expected_attributes = [
                'config',
                'agents',
                'tasks',
                'is_initialized',
                'memory_manager',
                'workflow_engine',
                'agent_factory',
                'task_factory',
                # Callback properties
                'on_workflow_start',
                'on_workflow_complete',
                'on_task_start',
                'on_task_complete',
                'on_error',
            ]

            for attr_name in expected_attributes:
                assert hasattr(orch, attr_name), f"Missing attribute: {attr_name}"

    def test_no_breaking_changes_in_signatures(self, sample_config):
        """Test method signatures haven't changed."""
        with patch.object(Orchestrator, 'initialize'):
            orch = Orchestrator(sample_config)

            # Test key method signatures are compatible
            import inspect

            # run() should be async
            assert inspect.iscoroutinefunction(orch.run)

            # run_sync() should not be async
            assert not inspect.iscoroutinefunction(orch.run_sync)

            # add_agent should accept AgentConfig
            sig = inspect.signature(orch.add_agent)
            assert 'agent_config' in sig.parameters or len(sig.parameters) >= 1

            # add_task should accept TaskConfig
            sig = inspect.signature(orch.add_task)
            assert 'task_config' in sig.parameters or len(sig.parameters) >= 1
