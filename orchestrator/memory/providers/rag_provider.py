"""
Simple in-memory provider with substring matching.

This provider stores documents in a Python dictionary and retrieves
them via case-insensitive substring search.  It is useful for tests
and lightweight deployments that do not need vector or graph backends.
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .base import BaseMemoryProvider
from ...core.config import EmbedderConfig
from ...core.exceptions import ProviderError

logger = logging.getLogger(__name__)

class SimpleMemoryProvider(BaseMemoryProvider):
    """Simple in-memory provider with substring matching."""

    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        """Initialize simple memory provider."""
        super().__init__(embedder_config)
        self.short_db_path: Optional[str] = None
        self.long_db_path: Optional[str] = None
        self.rag_db_path: Optional[str] = None
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize simple memory provider."""
        try:
            self.short_db_path = config.get("short_db", ".orchestrator/memory/short.db")
            self.long_db_path = config.get("long_db", ".orchestrator/memory/long.db")
            self.rag_db_path = config.get(
                "rag_db_path", ".orchestrator/memory/chroma_db"
            )

            # Create directories if they don't exist
            for db_path in [self.short_db_path, self.long_db_path, self.rag_db_path]:
                if db_path:
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            self.is_initialized = True
            logger.info("Simple memory provider initialized successfully")
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize simple memory provider: {str(e)}"
            )

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory."""
        if not self.is_initialized:
            raise ProviderError("Simple memory provider not initialized")

        try:
            ref_id = f"simple_{self._next_id}"
            self._next_id += 1

            entry = {
                "id": ref_id,
                "content": content,
                "metadata": metadata or {},
                "provider": "simple",
            }

            self._memory_store[ref_id] = entry
            logger.debug(f"Stored content with ID: {ref_id}")
            return ref_id
        except Exception as e:
            raise ProviderError(f"Failed to store content: {str(e)}")

    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve content via substring matching."""
        if not self.is_initialized:
            raise ProviderError("Simple memory provider not initialized")

        try:
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
            raise ProviderError(f"Failed to retrieve content: {str(e)}")

    def update(
        self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update content in memory."""
        if not self.is_initialized:
            raise ProviderError("Simple memory provider not initialized")

        try:
            if ref_id not in self._memory_store:
                raise ProviderError(f"Entry not found: {ref_id}")

            self._memory_store[ref_id]["content"] = content
            if metadata:
                self._memory_store[ref_id]["metadata"].update(metadata)

            logger.debug(f"Updated content for ID: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to update entry: {str(e)}")

    def delete(self, ref_id: str) -> None:
        """Delete content from memory."""
        if not self.is_initialized:
            raise ProviderError("Simple memory provider not initialized")

        try:
            if ref_id in self._memory_store:
                del self._memory_store[ref_id]
                logger.debug(f"Deleted content for ID: {ref_id}")
            else:
                logger.warning(f"Entry not found for deletion: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to delete entry: {str(e)}")


# Backward-compatible alias so existing imports keep working
RAGMemoryProvider = SimpleMemoryProvider
