"""
Refactored memory manager as factory/coordinator for different memory providers.

This is the cleaned up version that uses modular providers instead of 
a monolithic implementation.
"""

from typing import Dict, Any, Optional, List
import logging

from .providers.base import IMemoryProvider
from .providers.rag_provider import RAGMemoryProvider
from .providers.mem0_provider import Mem0MemoryProvider
from .providers.hybrid_provider import HybridRAGMemoryProvider

from .document_schema import (
    sanitize_for_chroma,
    default_conversation_metadata,
    current_timestamp,
)

from ..core.config import MemoryConfig, MemoryProvider
from ..core.exceptions import MemoryError, ProviderError

logger = logging.getLogger(__name__)


class MemoryManager:
    """Factory/coordinator for different memory providers."""
    
    def __init__(self, config: MemoryConfig):
        """Initialize memory manager."""
        self.config = config
        self.provider: Optional[IMemoryProvider] = None
        self._initialize_provider()
    
    def _initialize_provider(self) -> None:
        """Initialize the configured memory provider."""
        try:
            if self.config.provider == MemoryProvider.RAG:
                self.provider = RAGMemoryProvider(self.config.embedder)
            elif self.config.provider == MemoryProvider.MEM0:
                self.provider = Mem0MemoryProvider(self.config.embedder)
            elif self.config.provider == MemoryProvider.HYBRID:
                self.provider = HybridRAGMemoryProvider(self.config.embedder)
            elif self.config.provider == MemoryProvider.MONGODB:
                # TODO: Import MongoDBMemoryProvider when extracted
                raise MemoryError(f"MongoDB provider not yet refactored. Use RAG or Mem0 for now.")
            else:
                raise MemoryError(f"Unsupported memory provider: {self.config.provider}")
            
            # Initialize the provider with appropriate configuration
            provider_config = self._build_provider_config()
            self.provider.initialize(provider_config)
            
            logger.info(f"Memory manager initialized with {self.config.provider.value} provider")
        except Exception as e:
            raise MemoryError(f"Failed to initialize memory manager: {str(e)}")
    
    def _build_provider_config(self) -> Dict[str, Any]:
        """Build provider-specific configuration."""
        config = self.config.config.copy() if self.config.config else {}
        
        # Add provider-specific config
        if self.config.provider == MemoryProvider.RAG:
            config.update({
                "short_db": self.config.short_db,
                "long_db": self.config.long_db,
                "rag_db_path": self.config.rag_db_path
            })
        elif self.config.provider == MemoryProvider.HYBRID:
            # Pass through hybrid configuration directly
            # The configuration is already structured correctly from CLI
            pass
        
        return config
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        return self.provider.store(content, metadata)
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve content from memory."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        return self.provider.retrieve(query, limit)
    
    def retrieve_filtered(
        self,
        query: str,
        *,
        limit: int = 10,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        rerank: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve content with optional filters."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        # Try filtered retrieval if provider supports it
        try:
            if hasattr(self.provider, 'retrieve_filtered'):
                filter_kwargs = {}
                if user_id:
                    filter_kwargs["user_id"] = user_id
                if agent_id:
                    filter_kwargs["agent_id"] = agent_id
                if run_id:
                    filter_kwargs["run_id"] = run_id
                if rerank is not None:
                    filter_kwargs["rerank"] = rerank
                
                filter_kwargs.update(kwargs)
                return self.provider.retrieve_filtered(query, limit=limit, **filter_kwargs)
        except Exception as e:
            logger.warning(f"Filtered retrieval failed, falling back to basic retrieve: {e}")
        
        # Fallback to basic retrieval
        return self.retrieve(query, limit)
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update existing memory entry."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        return self.provider.update(ref_id, content, metadata)
    
    def delete(self, ref_id: str) -> None:
        """Delete memory entry."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        return self.provider.delete(ref_id)
    
    def health_check(self) -> bool:
        """Check if memory manager is healthy."""
        if not self.provider:
            return False
        
        return self.provider.health_check()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.provider:
            self.provider.cleanup()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider": self.config.provider.value,
            "initialized": self.provider is not None and self.provider.health_check(),
            "embedder": {
                "provider": self.config.embedder.provider if self.config.embedder else None,
                "model": (self.config.embedder.config or {}).get("model") if self.config.embedder else None
            } if self.config.embedder else None
        }
    
    def retrieve_with_graph(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using graph-enhanced search if provider supports it.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            **kwargs: Additional provider-specific parameters (tags, document_ids,
                     sections, user_id, run_id, agent_id)

        Returns:
            List of memory results with graph context if available

        Note:
            Falls back to standard retrieve() if provider doesn't implement
            graph-enhanced search. The hybrid provider uses RRF (Reciprocal Rank
            Fusion) and cross-encoder reranking instead of similarity thresholds.
        """
        if not self.provider:
            raise MemoryError("Memory provider not initialized")

        if hasattr(self.provider, "retrieve_with_graph"):
            return self.provider.retrieve_with_graph(
                query=query,
                limit=limit,
                **kwargs
            )
        else:
            # Graceful degradation for providers without graph support
            logger.debug(
                f"Provider {type(self.provider).__name__} doesn't support "
                "graph-enhanced retrieval. Falling back to standard retrieve()."
            )
            return self.provider.retrieve(query, limit=limit, **kwargs)

    def create_graph_tool(self, *, default_user_id: str, default_run_id: str):
        """Create a GraphRAG lookup tool if the provider supports it."""
        # This method would delegate to the provider if it supports graph operations
        # For now, we'll raise an error to indicate it's not implemented
        if not self.provider:
            raise MemoryError("Memory provider not initialized")

        if hasattr(self.provider, 'create_graph_tool'):
            return self.provider.create_graph_tool(
                default_user_id=default_user_id,
                default_run_id=default_run_id
            )
        else:
            raise MemoryError(f"Graph tools not supported by {self.config.provider.value} provider")