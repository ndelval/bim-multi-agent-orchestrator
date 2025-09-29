"""
Base classes and interfaces for memory providers.
"""

from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import logging

from ...core.config import EmbedderConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class IMemoryProvider(Protocol):
    """Interface for memory providers."""
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the memory provider."""
        ...
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory and return reference ID."""
        ...
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant content based on query."""
        ...

    # Optional enhanced retrieval with filters (not required for all providers)
    def retrieve_filtered(
        self,
        query: str,
        *,
        limit: int = 10,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve content with optional user/agent/run filters."""
        ...
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update existing memory entry."""
        ...
    
    def delete(self, ref_id: str) -> None:
        """Delete memory entry."""
        ...
    
    def health_check(self) -> bool:
        """Check if provider is healthy."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        ...


class BaseMemoryProvider(ABC):
    """Base class for memory providers."""
    
    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        """Initialize base provider."""
        self.embedder_config = embedder_config or EmbedderConfig()
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the memory provider."""
        pass
    
    @abstractmethod
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory and return reference ID."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant content based on query."""
        pass
    
    @abstractmethod
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update existing memory entry."""
        pass
    
    @abstractmethod
    def delete(self, ref_id: str) -> None:
        """Delete memory entry."""
        pass
    
    def health_check(self) -> bool:
        """Check if provider is healthy."""
        return self.is_initialized
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass