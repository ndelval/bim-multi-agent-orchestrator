"""
Routing configuration and strategy system for graph workflows.

Provides configurable route-to-agent mappings and routing strategies
following the Strategy Pattern for extensibility.
"""

import logging
from typing import Dict, Optional, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RoutingStrategy(Protocol):
    """Protocol for routing strategies."""

    def get_target_agent(self, route: str, agent_nodes: Dict[str, str]) -> str:
        """
        Get target agent node for a given route.

        Args:
            route: The route to map
            agent_nodes: Available agent nodes {AgentName: node_name}

        Returns:
            Target node name
        """
        ...


class DefaultRoutingStrategy:
    """
    Default routing strategy using configurable route mappings.

    Maps routes to agent names with fallback behavior for missing agents.
    """

    def __init__(
        self,
        route_mapping: Optional[Dict[str, str]] = None,
        fallback_agent: str = "orchestrator",
    ):
        """
        Initialize routing strategy.

        Args:
            route_mapping: Custom route-to-agent mappings
            fallback_agent: Fallback agent when route not found
        """
        self.route_mapping = route_mapping or self._default_route_mapping()
        self.fallback_agent = fallback_agent

    @staticmethod
    def _default_route_mapping() -> Dict[str, str]:
        """
        Default route-to-agent mapping.

        Returns:
            Dictionary mapping routes to preferred agent names
        """
        return {
            "quick": "QuickResponder",
            "research": "Researcher",
            "analysis": "Analyst",
            "planning": "Planner",
            "standards": "StandardsAgent",
        }

    def get_target_agent(self, route: str, agent_nodes: Dict[str, str]) -> str:
        """
        Get target agent node for a given route.

        Args:
            route: The route to map (e.g., 'research', 'analysis')
            agent_nodes: Available agent nodes {AgentName: node_name}

        Returns:
            Target node name for the route
        """
        # Get preferred agent name for this route
        preferred_agent = self.route_mapping.get(route, self.fallback_agent)

        # Find matching node by normalizing names
        target_node = self._find_matching_node(preferred_agent, agent_nodes)

        if target_node:
            logger.debug(
                "[RoutingStrategy] Route '%s' → Agent '%s' (node: %s)",
                route,
                preferred_agent,
                target_node,
            )
            return target_node

        # Fallback to first available agent
        if agent_nodes:
            fallback = list(agent_nodes.values())[0]
            logger.warning(
                "[RoutingStrategy] No agent for route '%s', using fallback: %s",
                route,
                fallback,
            )
            return fallback

        # Ultimate fallback
        logger.error("[RoutingStrategy] No agents available, using 'final_output'")
        return "final_output"

    def _find_matching_node(
        self, agent_name: str, agent_nodes: Dict[str, str]
    ) -> Optional[str]:
        """
        Find matching node name for agent.

        Normalizes both agent name and node names for flexible matching.

        Args:
            agent_name: Target agent name (e.g., 'QuickResponder')
            agent_nodes: Available agent nodes {AgentName: node_name}

        Returns:
            Matching node name or None
        """
        normalized_target = self._normalize_name(agent_name)

        for config_name, node_name in agent_nodes.items():
            if self._normalize_name(config_name) == normalized_target:
                return node_name

        return None

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize agent/node name for comparison.

        Args:
            name: Name to normalize

        Returns:
            Normalized lowercase name without spaces/underscores
        """
        return name.lower().replace(" ", "").replace("_", "")


class ChatRoutingStrategy(DefaultRoutingStrategy):
    """
    Routing strategy optimized for chat-style interactions.

    Extends default strategy with chat-specific node name handling.
    """

    def get_target_agent(self, route: str, agent_nodes: Dict[str, str]) -> str:
        """
        Get target agent for chat routing.

        Args:
            route: The route to map
            agent_nodes: Available agent nodes (values are node names, not agent names)

        Returns:
            Target node name for chat routing
        """
        # For chat graphs, agent_nodes values are already node names
        route_to_node = {
            "quick": "quickresponder",
            "research": "researcher",
            "analysis": "analyst",
            "planning": "planner",
            "standards": "standardsagent",
        }

        target_node = route_to_node.get(route)

        # Check if target node exists
        if target_node and target_node in agent_nodes.values():
            logger.debug("[ChatRoutingStrategy] Route '%s' → Node '%s'", route, target_node)
            return target_node

        # Fallback to first available agent or complete
        if agent_nodes:
            fallback = list(agent_nodes.values())[0]
            logger.warning(
                "[ChatRoutingStrategy] No node for route '%s', using fallback: %s",
                route,
                fallback,
            )
            return fallback

        return "complete"


class RoutingRegistry:
    """
    Registry for managing multiple routing strategies.

    Allows runtime selection and registration of custom routing strategies.
    """

    def __init__(self):
        """Initialize routing registry with default strategies."""
        self._strategies: Dict[str, RoutingStrategy] = {
            "default": DefaultRoutingStrategy(),
            "chat": ChatRoutingStrategy(),
        }

    def register_strategy(self, name: str, strategy: RoutingStrategy) -> None:
        """
        Register a custom routing strategy.

        Args:
            name: Unique name for the strategy
            strategy: Strategy implementation
        """
        self._strategies[name] = strategy
        logger.info("[RoutingRegistry] Registered strategy: %s", name)

    def get_strategy(self, name: str) -> RoutingStrategy:
        """
        Get routing strategy by name.

        Args:
            name: Strategy name

        Returns:
            RoutingStrategy instance

        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            logger.warning(
                "[RoutingRegistry] Strategy '%s' not found, using 'default'", name
            )
            return self._strategies["default"]

        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())


# Global registry instance
_global_registry = RoutingRegistry()


def get_routing_strategy(name: str = "default") -> RoutingStrategy:
    """
    Get routing strategy from global registry.

    Args:
        name: Strategy name (default: "default")

    Returns:
        RoutingStrategy instance
    """
    return _global_registry.get_strategy(name)


def register_routing_strategy(name: str, strategy: RoutingStrategy) -> None:
    """
    Register custom routing strategy globally.

    Args:
        name: Unique strategy name
        strategy: Strategy implementation
    """
    _global_registry.register_strategy(name, strategy)
