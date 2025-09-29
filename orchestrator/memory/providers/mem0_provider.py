"""
Mem0-based memory provider with graph support.
"""

from typing import Dict, Any, Optional, List
import logging

from .base import BaseMemoryProvider
from ...core.config import EmbedderConfig
from ...core.exceptions import ProviderError, ConfigurationError
from ...core.embedding_utils import normalize_embedder_config

logger = logging.getLogger(__name__)


class Mem0MemoryProvider(BaseMemoryProvider):
    """Mem0-based memory provider with graph support."""
    
    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        """Initialize Mem0 provider."""
        super().__init__(embedder_config)
        self.graph_store_config: Optional[Dict[str, Any]] = None
        self.vector_store_config: Optional[Dict[str, Any]] = None
        self.llm_config: Optional[Dict[str, Any]] = None
        self._mem0_client = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Mem0 provider."""
        try:
            self.graph_store_config = config.get("graph_store")
            if not self.graph_store_config:
                raise ConfigurationError("Mem0 requires graph_store configuration")
            
            self.vector_store_config = config.get("vector_store")
            self.llm_config = config.get("llm")
            embedder_cfg = config.get("embedder")
            
            # Use centralized embedding utils instead of hardcoded dimensions
            if embedder_cfg:
                embedder_cfg = normalize_embedder_config(embedder_cfg)
            
            # Import mem0 only when needed
            try:
                from mem0 import Memory
                
                mem0_config = {
                    "graph_store": self.graph_store_config
                }
                
                if self.vector_store_config:
                    mem0_config["vector_store"] = self.vector_store_config
                
                if self.llm_config:
                    mem0_config["llm"] = self.llm_config
                if embedder_cfg:
                    mem0_config["embedder"] = embedder_cfg
                
                # Initialize with proper error handling
                self._mem0_client = self._init_mem0_client(mem0_config)
                
                self.is_initialized = True
                logger.info("Mem0 memory provider initialized successfully")
            except ImportError:
                raise ProviderError("mem0ai package required for Mem0 provider")
        except Exception as e:
            raise ProviderError(f"Failed to initialize Mem0 provider: {str(e)}")
    
    def _init_mem0_client(self, config: Dict[str, Any]):
        """Initialize Mem0 client with fallback handling."""
        try:
            # Prefer factory initializer for compatibility
            try:
                from mem0 import Memory
                return Memory.from_config(config_dict=config)
            except TypeError:
                # Backward compatibility with older signatures
                return Memory.from_config(config)
        except Exception as e:
            # If vector store is configured and connection fails, retry without it
            if "vector_store" in config:
                logger.warning(f"Mem0 vector store initialization failed ({e}); retrying without vector store")
                config_no_vector = {k: v for k, v in config.items() if k != "vector_store"}
                from mem0 import Memory
                return Memory.from_config(config_dict=config_no_vector)
            else:
                raise
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in Mem0."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            user_id = metadata.get("user_id", "default") if metadata else "default"
            agent_id = metadata.get("agent_id") if metadata else None
            run_id = metadata.get("run_id") if metadata else None
            
            # Build add parameters
            add_kwargs = {"user_id": user_id, "metadata": metadata or {}}
            if agent_id:
                add_kwargs["agent_id"] = agent_id
            if run_id:
                add_kwargs["run_id"] = run_id
            
            result = self._mem0_client.add(content, **add_kwargs)
            
            ref_id = result.get("id", str(result))
            logger.debug(f"Stored content with ID: {ref_id}")
            return ref_id
        except Exception as e:
            raise ProviderError(f"Failed to store content in Mem0: {str(e)}")
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve content from Mem0."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            raw = self._mem0_client.search(query=query, limit=limit)
            if isinstance(raw, dict):
                results = raw.get("results") or raw.get("memories") or []
            else:
                results = raw
            
            formatted_results = []
            for item in results or []:
                if isinstance(item, str):
                    formatted_results.append({
                        "id": None,
                        "content": item,
                        "metadata": {},
                        "provider": "mem0",
                        "score": 0,
                    })
                elif isinstance(item, dict):
                    formatted_results.append({
                        "id": item.get("id"),
                        "content": item.get("memory", item.get("content", "")),
                        "metadata": item.get("metadata", {}),
                        "provider": "mem0",
                        "score": item.get("score", 0),
                    })
            
            logger.debug(f"Retrieved {len(formatted_results)} results for query: {query}")
            return formatted_results
        except Exception as e:
            raise ProviderError(f"Failed to retrieve from Mem0: {str(e)}")
    
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
        """Retrieve content with filters."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            search_kwargs = {"query": query, "limit": limit}
            if user_id:
                search_kwargs["user_id"] = user_id
            if agent_id:
                search_kwargs["agent_id"] = agent_id
            if run_id:
                search_kwargs["run_id"] = run_id
            
            # Add any additional kwargs
            search_kwargs.update(kwargs)
            
            raw = self._mem0_client.search(**search_kwargs)
            if isinstance(raw, dict):
                results = raw.get("results") or raw.get("memories") or []
            else:
                results = raw
            
            # Format results using the same logic as retrieve
            formatted_results = []
            for item in results or []:
                if isinstance(item, str):
                    formatted_results.append({
                        "id": None,
                        "content": item,
                        "metadata": {},
                        "provider": "mem0",
                        "score": 0,
                    })
                elif isinstance(item, dict):
                    formatted_results.append({
                        "id": item.get("id"),
                        "content": item.get("memory", item.get("content", "")),
                        "metadata": item.get("metadata", {}),
                        "provider": "mem0",
                        "score": item.get("score", 0),
                    })
            
            return formatted_results
        except Exception as e:
            # Fallback to basic retrieve if filtered search fails
            logger.warning(f"Filtered search failed, falling back to basic search: {e}")
            return self.retrieve(query, limit)
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update content in Mem0."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            # Note: Mem0 may not support direct updates, so this is a basic implementation
            self._mem0_client.update(memory_id=ref_id, data=content)
            logger.debug(f"Updated content for ID: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to update Mem0 entry: {str(e)}")
    
    def delete(self, ref_id: str) -> None:
        """Delete content from Mem0."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            self._mem0_client.delete(memory_id=ref_id)
            logger.debug(f"Deleted content for ID: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to delete Mem0 entry: {str(e)}")