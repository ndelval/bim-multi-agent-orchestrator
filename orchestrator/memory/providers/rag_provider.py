"""
RAG-based memory provider using local storage.
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .base import BaseMemoryProvider
from ...core.config import EmbedderConfig
from ...core.exceptions import ProviderError

logger = logging.getLogger(__name__)


class RAGMemoryProvider(BaseMemoryProvider):
    """RAG-based memory provider using local storage."""
    
    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        """Initialize RAG provider."""
        super().__init__(embedder_config)
        self.short_db_path: Optional[str] = None
        self.long_db_path: Optional[str] = None
        self.rag_db_path: Optional[str] = None
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize RAG provider."""
        try:
            self.short_db_path = config.get("short_db", ".orchestrator/memory/short.db")
            self.long_db_path = config.get("long_db", ".orchestrator/memory/long.db")
            self.rag_db_path = config.get("rag_db_path", ".orchestrator/memory/chroma_db")
            
            # Create directories if they don't exist
            for db_path in [self.short_db_path, self.long_db_path, self.rag_db_path]:
                if db_path:
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.is_initialized = True
            logger.info("RAG memory provider initialized successfully")
        except Exception as e:
            raise ProviderError(f"Failed to initialize RAG provider: {str(e)}")
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in RAG memory."""
        if not self.is_initialized:
            raise ProviderError("RAG provider not initialized")
        
        try:
            ref_id = f"rag_{self._next_id}"
            self._next_id += 1
            
            entry = {
                "id": ref_id,
                "content": content,
                "metadata": metadata or {},
                "provider": "rag"
            }
            
            self._memory_store[ref_id] = entry
            logger.debug(f"Stored content with ID: {ref_id}")
            return ref_id
        except Exception as e:
            raise ProviderError(f"Failed to store content in RAG: {str(e)}")
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve content from RAG memory."""
        if not self.is_initialized:
            raise ProviderError("RAG provider not initialized")
        
        try:
            # Simple text-based search for now
            results = []
            query_lower = query.lower()
            
            for entry in self._memory_store.values():
                content_lower = entry["content"].lower()
                if query_lower in content_lower:
                    results.append(entry.copy())
                
                if len(results) >= limit:
                    break
            
            logger.debug(f"Retrieved {len(results)} results for query: {query}")
            return results
        except Exception as e:
            raise ProviderError(f"Failed to retrieve from RAG: {str(e)}")
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update content in RAG memory."""
        if not self.is_initialized:
            raise ProviderError("RAG provider not initialized")
        
        try:
            if ref_id not in self._memory_store:
                raise ProviderError(f"Entry not found: {ref_id}")
            
            self._memory_store[ref_id]["content"] = content
            if metadata:
                self._memory_store[ref_id]["metadata"].update(metadata)
            
            logger.debug(f"Updated content for ID: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to update RAG entry: {str(e)}")
    
    def delete(self, ref_id: str) -> None:
        """Delete content from RAG memory."""
        if not self.is_initialized:
            raise ProviderError("RAG provider not initialized")
        
        try:
            if ref_id in self._memory_store:
                del self._memory_store[ref_id]
                logger.debug(f"Deleted content for ID: {ref_id}")
            else:
                logger.warning(f"Entry not found for deletion: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to delete RAG entry: {str(e)}")