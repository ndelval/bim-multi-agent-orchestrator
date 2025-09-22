"""
Memory manager with provider abstraction for different memory backends.
"""

from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import sqlite3
import json
import time
import uuid
import threading
from functools import lru_cache

from .document_schema import (
    sanitize_for_chroma,
    default_conversation_metadata,
    current_timestamp,
)

from ..core.config import MemoryConfig, EmbedderConfig, MemoryProvider
from ..core.exceptions import MemoryError, ProviderError, ConfigurationError


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
            self.short_db_path = config.get("short_db", ".praison/memory/short.db")
            self.long_db_path = config.get("long_db", ".praison/memory/long.db")
            self.rag_db_path = config.get("rag_db_path", ".praison/memory/chroma_db")
            
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


class MongoDBMemoryProvider(BaseMemoryProvider):
    """MongoDB-based memory provider."""
    
    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        """Initialize MongoDB provider."""
        super().__init__(embedder_config)
        self.connection_string: Optional[str] = None
        self.database_name: Optional[str] = None
        self.use_vector_search: bool = False
        self.max_pool_size: int = 100
        self._client = None
        self._database = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize MongoDB provider."""
        try:
            self.connection_string = config.get("connection_string")
            if not self.connection_string:
                raise ConfigurationError("MongoDB requires connection_string")
            
            self.database_name = config.get("database", "praisonai")
            self.use_vector_search = config.get("use_vector_search", True)
            self.max_pool_size = config.get("max_pool_size", 100)
            
            # Import pymongo only when needed
            try:
                from pymongo import MongoClient
                self._client = MongoClient(
                    self.connection_string,
                    maxPoolSize=self.max_pool_size
                )
                self._database = self._client[self.database_name]
                
                # Test connection
                self._client.admin.command("ping")
                
                self.is_initialized = True
                logger.info("MongoDB memory provider initialized successfully")
            except ImportError:
                raise ProviderError("pymongo package required for MongoDB provider")
        except Exception as e:
            raise ProviderError(f"Failed to initialize MongoDB provider: {str(e)}")
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in MongoDB."""
        if not self.is_initialized:
            raise ProviderError("MongoDB provider not initialized")
        
        try:
            from bson import ObjectId
            
            document = {
                "content": content,
                "metadata": metadata or {},
                "provider": "mongodb"
            }
            
            collection = self._database["memory"]
            result = collection.insert_one(document)
            ref_id = str(result.inserted_id)
            
            logger.debug(f"Stored content with ID: {ref_id}")
            return ref_id
        except Exception as e:
            raise ProviderError(f"Failed to store content in MongoDB: {str(e)}")
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve content from MongoDB."""
        if not self.is_initialized:
            raise ProviderError("MongoDB provider not initialized")
        
        try:
            collection = self._database["memory"]
            
            # Simple text search
            cursor = collection.find({
                "$text": {"$search": query}
            }).limit(limit)
            
            results = []
            for doc in cursor:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                results.append(doc)
            
            logger.debug(f"Retrieved {len(results)} results for query: {query}")
            return results
        except Exception as e:
            raise ProviderError(f"Failed to retrieve from MongoDB: {str(e)}")

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
        """Retrieve with optional filters by user/agent/run.

        Note: current implementation uses $text search; vector search can be added later.
        """
        if not self.is_initialized:
            raise ProviderError("MongoDB provider not initialized")
        try:
            collection = self._database["memory"]
            mfilter: Dict[str, Any] = {"$text": {"$search": query}}
            if user_id:
                mfilter["metadata.user_id"] = user_id
            if agent_id:
                mfilter["metadata.agent_id"] = agent_id
            if run_id:
                mfilter["metadata.run_id"] = run_id
            cursor = collection.find(mfilter).limit(limit)
            results: List[Dict[str, Any]] = []
            for doc in cursor:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                results.append(doc)
            return results
        except Exception as e:
            raise ProviderError(f"Failed to retrieve (filtered) from MongoDB: {str(e)}")
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update content in MongoDB."""
        if not self.is_initialized:
            raise ProviderError("MongoDB provider not initialized")
        
        try:
            from bson import ObjectId
            
            collection = self._database["memory"]
            update_doc = {"content": content}
            if metadata:
                update_doc["metadata"] = metadata
            
            result = collection.update_one(
                {"_id": ObjectId(ref_id)},
                {"$set": update_doc}
            )
            
            if result.matched_count == 0:
                raise ProviderError(f"Entry not found: {ref_id}")
            
            logger.debug(f"Updated content for ID: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to update MongoDB entry: {str(e)}")
    
    def delete(self, ref_id: str) -> None:
        """Delete content from MongoDB."""
        if not self.is_initialized:
            raise ProviderError("MongoDB provider not initialized")
        
        try:
            from bson import ObjectId
            
            collection = self._database["memory"]
            result = collection.delete_one({"_id": ObjectId(ref_id)})
            
            if result.deleted_count == 0:
                logger.warning(f"Entry not found for deletion: {ref_id}")
            else:
                logger.debug(f"Deleted content for ID: {ref_id}")
        except Exception as e:
            raise ProviderError(f"Failed to delete MongoDB entry: {str(e)}")
    
    def health_check(self) -> bool:
        """Check MongoDB connection health."""
        if not self.is_initialized:
            return False
        
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False
    
    def cleanup(self) -> None:
        """Cleanup MongoDB resources."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            self.is_initialized = False


class HybridRAGMemoryProvider(BaseMemoryProvider):
    """Hybrid RAG provider combining vector search, lexical BM25 and cross-encoder reranking."""

    _cross_encoder_lock = threading.Lock()

    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        super().__init__(embedder_config)
        self._vector_client = None
        self._vector_collection = None
        self._vector_path: Optional[str] = None
        self._lex_conn: Optional[sqlite3.Connection] = None
        self._lex_lock = threading.RLock()
        self._lex_cleanup_every = 100
        self._lex_write_count = 0
        self._lex_ttl_seconds: Optional[float] = None
        self._lex_max_entries: Optional[int] = None
        self._rerank_enabled = False
        self._rerank_top_k = 10
        self._rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._rerank_available = False
        self._rerank_executor: Optional["ThreadPoolExecutor"] = None
        self._rerank_max_workers = 1
        self._score_cache: Dict[str, float] = {}
        self._embedding_model = (self.embedder_config.config or {}).get("model", "text-embedding-3-small")
        self._embedding_dimensions = self._infer_embedding_dims(self._embedding_model)
        self._graph_config: Dict[str, Any] = {}
        self._graph_driver = None

    def initialize(self, config: Dict[str, Any]) -> None:
        try:
            vector_cfg = config.get("vector_store", {}) or {}
            lexical_cfg = config.get("lexical", {}) or {}
            rerank_cfg = config.get("rerank", {}) or {}

            self._initialize_vector_store(vector_cfg)
            self._initialize_lexical_index(lexical_cfg)
            self._initialize_reranker(rerank_cfg)
            self._initialize_graph(config.get("graph") or {})

            self.is_initialized = True
            logger.info("Hybrid RAG provider initialized successfully")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to initialize Hybrid RAG provider: {str(e)}")

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if not self.is_initialized:
            raise ProviderError("Hybrid provider not initialized")

        meta = self._normalize_metadata(metadata)
        doc_id = meta.get("id") or str(uuid.uuid4())
        meta["id"] = doc_id

        embedding = self._generate_embedding(content)
        if embedding is not None and self._vector_collection:
            try:
                chroma_meta = sanitize_for_chroma(meta)
                self._vector_collection.upsert(
                    ids=[doc_id],
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[chroma_meta],
                )
            except Exception as e:
                logger.warning(f"Hybrid vector store upsert failed: {e}")

        self._store_lexical(doc_id, content, meta)
        self._upsert_graph(doc_id, content, meta)
        return doc_id

    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.retrieve_filtered(query, limit=limit)

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
        if not self.is_initialized:
            raise ProviderError("Hybrid provider not initialized")

        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        vector_results = self._search_vector(query, filters, limit)
        lexical_results = self._search_lexical(query, filters, limit)

        fused = self._reciprocal_rank_fusion([vector_results, lexical_results])
        reranked = self._rerank(query, fused)

        return reranked[:limit]

    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        meta = self._normalize_metadata(metadata)
        meta["id"] = ref_id
        self.delete(ref_id)
        self.store(content, meta)

    def delete(self, ref_id: str) -> None:
        if not self.is_initialized:
            raise ProviderError("Hybrid provider not initialized")

        if self._vector_collection:
            try:
                self._vector_collection.delete(ids=[ref_id])
            except Exception:
                pass

        with self._lex_lock:
            cursor = self._lex_conn.cursor()
            cursor.execute("SELECT rowid FROM documents WHERE id = ?", (ref_id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                cursor.execute("DELETE FROM documents WHERE rowid = ?", (rowid,))
                cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (rowid,))
                self._lex_conn.commit()

    def health_check(self) -> bool:
        if not self.is_initialized:
            return False
        vector_ok = True
        if self._vector_collection:
            try:
                self._vector_collection.count()
            except Exception:
                vector_ok = False
        lexical_ok = True
        if self._lex_conn is None:
            lexical_ok = False
        return vector_ok and lexical_ok

    def cleanup(self) -> None:
        if self._lex_conn:
            self._lex_conn.close()
            self._lex_conn = None
        if self._vector_client:
            try:
                self._vector_client.reset()
            except Exception:
                pass
            self._vector_client = None
            self._vector_collection = None
        if self._rerank_executor:
            self._rerank_executor.shutdown(wait=False)
            self._rerank_executor = None
        if self._graph_driver:
            self._graph_driver.close()
            self._graph_driver = None
        self.is_initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_vector_store(self, vector_cfg: Dict[str, Any]) -> None:
        provider = (vector_cfg.get("provider") or "chromadb").lower()
        if provider != "chromadb":
            raise ProviderError("Hybrid provider currently supports only 'chromadb' as vector store")

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise ProviderError("chromadb package required for Hybrid provider. Install with 'pip install chromadb'.") from exc

        path = vector_cfg.get("config", {}).get("path") or vector_cfg.get("path") or ".praison/hybrid_chroma"
        Path(path).mkdir(parents=True, exist_ok=True)

        self._vector_path = path
        settings = ChromaSettings(
            allow_reset=True,
            anonymized_telemetry=False,
        )
        self._vector_client = chromadb.PersistentClient(path=path, settings=settings)
        collection_name = vector_cfg.get("config", {}).get("collection") or vector_cfg.get("collection") or "hybrid_memory"
        try:
            self._vector_collection = self._vector_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise ProviderError(f"Failed to create Chroma collection: {exc}")

    def _initialize_lexical_index(self, lexical_cfg: Dict[str, Any]) -> None:
        db_path = lexical_cfg.get("db_path") or ".praison/hybrid_lexical.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lex_conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lex_lock:
            cursor = self._lex_conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    content,
                    tokenize='unicode61'
                )
                """
            )
            self._lex_conn.commit()

        self._lex_ttl_seconds = lexical_cfg.get("ttl_seconds")
        if self._lex_ttl_seconds is not None:
            try:
                self._lex_ttl_seconds = float(self._lex_ttl_seconds)
            except ValueError:
                self._lex_ttl_seconds = None

        self._lex_max_entries = lexical_cfg.get("max_entries")
        if self._lex_max_entries is not None:
            try:
                self._lex_max_entries = int(self._lex_max_entries)
            except ValueError:
                self._lex_max_entries = None

        if lexical_cfg.get("cleanup_interval"):
            try:
                self._lex_cleanup_every = max(1, int(lexical_cfg.get("cleanup_interval")))
            except ValueError:
                pass

    def _initialize_reranker(self, rerank_cfg: Dict[str, Any]) -> None:
        self._rerank_enabled = bool(rerank_cfg.get("enabled", True))
        self._rerank_top_k = int(rerank_cfg.get("top_k", 10))
        self._rerank_model_name = rerank_cfg.get("model") or self._rerank_model_name
        self._rerank_max_workers = max(1, int(rerank_cfg.get("max_workers", 1)))

        if not self._rerank_enabled:
            logger.info("Hybrid reranker disabled by configuration")
            return

        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            self._rerank_available = True
            from concurrent.futures import ThreadPoolExecutor

            self._rerank_executor = ThreadPoolExecutor(max_workers=self._rerank_max_workers)
        except ImportError:
            self._rerank_available = False
            logger.warning("sentence-transformers not installed; hybrid reranking disabled")

    def _initialize_graph(self, graph_cfg: Dict[str, Any]) -> None:
        enabled = graph_cfg.get("enabled")
        if not enabled:
            return

        uri = graph_cfg.get("uri")
        user = graph_cfg.get("user")
        password = graph_cfg.get("password")
        if not uri or not user or not password:
            logger.warning("Hybrid graph configuration incomplete; skipping graph sync")
            return

        try:
            from neo4j import GraphDatabase

            self._graph_driver = GraphDatabase.driver(uri, auth=(user, password))
            self._graph_config = {"uri": uri, "user": user}
            logger.info("Hybrid graph synchronization enabled")
        except ImportError:
            logger.warning("neo4j-driver not installed; skipping graph sync")
        except Exception as exc:
            logger.warning(f"Failed to initialize Neo4j driver: {exc}")
            self._graph_driver = None

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        try:
            import litellm

            response = litellm.embedding(model=self._embedding_model, input=text)
            return list(response.data[0]["embedding"])
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"litellm embedding failed: {e}")

        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.embeddings.create(model=self._embedding_model, input=text)
            return list(response.data[0].embedding)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _store_lexical(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        timestamp = time.time()
        meta_json = json.dumps(metadata or {})
        with self._lex_lock:
            cursor = self._lex_conn.cursor()
            cursor.execute("SELECT rowid FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                cursor.execute("DELETE FROM documents WHERE rowid = ?", (row[0],))
                cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (row[0],))

            cursor.execute(
                "INSERT INTO documents (id, content, metadata, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, content, meta_json, timestamp),
            )
            cursor.execute(
                "INSERT INTO documents_fts(rowid, content) VALUES ((SELECT rowid FROM documents WHERE id = ?), ?)",
                (doc_id, content),
            )
            self._lex_conn.commit()
            self._lex_write_count += 1

            if self._lex_write_count % self._lex_cleanup_every == 0:
                self._prune_lexical_index(cursor)
                self._lex_conn.commit()

    def _prune_lexical_index(self, cursor: sqlite3.Cursor) -> None:
        if self._lex_ttl_seconds:
            cutoff = time.time() - self._lex_ttl_seconds
            cursor.execute("SELECT rowid FROM documents WHERE created_at < ?", (cutoff,))
            rows = cursor.fetchall()
            if rows:
                rowids = [row[0] for row in rows]
                cursor.executemany("DELETE FROM documents WHERE rowid = ?", [(rid,) for rid in rowids])
                cursor.executemany("DELETE FROM documents_fts WHERE rowid = ?", [(rid,) for rid in rowids])

        if self._lex_max_entries:
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            if count > self._lex_max_entries:
                overflow = count - self._lex_max_entries
                cursor.execute("SELECT rowid FROM documents ORDER BY created_at ASC LIMIT ?", (overflow,))
                rows = cursor.fetchall()
                rowids = [row[0] for row in rows]
                cursor.executemany("DELETE FROM documents WHERE rowid = ?", [(rid,) for rid in rowids])
                cursor.executemany("DELETE FROM documents_fts WHERE rowid = ?", [(rid,) for rid in rowids])

    def _upsert_graph(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        if not self._graph_driver:
            return

        if not metadata.get("document_id"):
            return

        chunk_id = metadata.get("chunk_id") or doc_id
        params = {
            "document_id": metadata.get("document_id"),
            "title": metadata.get("title"),
            "discipline": metadata.get("discipline"),
            "doc_type": metadata.get("doc_type"),
            "version": metadata.get("version"),
            "effective_date": metadata.get("effective_date"),
            "language": metadata.get("language"),
            "source_url": metadata.get("source_url"),
            "tags": metadata.get("tags") if isinstance(metadata.get("tags"), list) else None,
            "chunk_id": chunk_id,
            "section": metadata.get("section"),
            "page_range": metadata.get("page_range"),
            "order": metadata.get("order"),
            "content": content,
            "ingest_timestamp": metadata.get("ingest_timestamp"),
        }

        try:
            with self._graph_driver.session() as session:
                session.execute_write(self._graph_write_tx, params)
        except Exception as exc:
            logger.debug(f"Hybrid graph upsert failed: {exc}")

    @staticmethod
    def _graph_write_tx(tx, params: Dict[str, Any]) -> None:
        tx.run(
            """
            MERGE (d:Document {document_id: $document_id})
            ON CREATE SET d.created_at = datetime($ingest_timestamp)
            SET d.title = COALESCE($title, d.title),
                d.discipline = $discipline,
                d.doc_type = $doc_type,
                d.version = $version,
                d.effective_date = $effective_date,
                d.language = $language,
                d.source_url = $source_url,
                d.tags = $tags

            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.content = $content,
                c.section = $section,
                c.page_range = $page_range,
                c.order_index = $order,
                c.ingest_timestamp = $ingest_timestamp

            MERGE (c)-[:BELONGS_TO]->(d)
            """,
            params,
        )

    def _search_vector(self, query: str, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        if not self._vector_collection:
            return []

        embedding = self._generate_embedding(query)
        if embedding is None:
            return []

        try:
            query_kwargs = {
                "query_embeddings": [embedding],
                "n_results": max(limit, self._rerank_top_k),
            }
            if filters:
                where_clause = self._build_chroma_where(filters)
                if where_clause:
                    query_kwargs["where"] = where_clause

            result = self._vector_collection.query(**query_kwargs)
            docs = result.get("documents", [[]])[0]
            ids = result.get("ids", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            vector_hits = []
            for idx, doc_id in enumerate(ids):
                if doc_id is None:
                    continue
                distance = distances[idx] if distances is not None and idx < len(distances) else None
                score = 0.0
                if distance is not None:
                    score = 1.0 / (1.0 + distance)
                vector_hits.append({
                    "id": doc_id,
                    "content": docs[idx] if docs and idx < len(docs) else "",
                    "metadata": metas[idx] if metas and idx < len(metas) else {},
                    "score": score,
                    "source": "vector",
                })
            return vector_hits
        except Exception as e:
            logger.warning(f"Hybrid vector search failed: {e}")
            return []

    def _search_lexical(self, query: str, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        if not self._lex_conn:
            return []

        with self._lex_lock:
            cursor = self._lex_conn.cursor()
            where_clauses = []
            params: List[Any] = []

            if filters.get("user_id"):
                where_clauses.append(
                    "(json_extract(metadata, '$.user_id') = ? OR json_extract(metadata, '$.user_id') = 'global')"
                )
                params.append(filters["user_id"])
            if filters.get("agent_id"):
                where_clauses.append("json_extract(metadata, '$.agent_id') = ?")
                params.append(filters["agent_id"])
            if filters.get("run_id"):
                where_clauses.append(
                    "(json_extract(metadata, '$.run_id') = ? OR json_extract(metadata, '$.run_id') = 'library')"
                )
                params.append(filters["run_id"])

            where_sql = " AND ".join(where_clauses)
            if where_sql:
                where_sql = " AND " + where_sql

            match_query = self._sanitize_match_query(query)

            sql = f"""
                SELECT d.id, d.content, d.metadata,
                       1.0 / (1.0 + bm25(documents_fts)) AS score
                FROM documents_fts
                JOIN documents d ON d.rowid = documents_fts.rowid
                WHERE documents_fts MATCH ?{where_sql}
                ORDER BY score DESC
                LIMIT ?
            """

            params = [match_query] + params + [max(limit, self._rerank_top_k)]

            try:
                rows = cursor.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                logger.debug(f"Lexical search failed ({e}); trying fallback query")
                rows = []

        results = []
        for row in rows:
            meta = {}
            if row[2]:
                try:
                    meta = json.loads(row[2])
                except json.JSONDecodeError:
                    meta = {}
            results.append({
                "id": row[0],
                "content": row[1],
                "metadata": meta,
                "score": float(row[3] or 0.0),
                "source": "lexical",
            })
        return results

    @staticmethod
    def _build_chroma_where(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            if key in {"user_id", "run_id"}:
                clauses.append({key: {"$in": [value, "global" if key == "user_id" else "library"]}})
            else:
                clauses.append({key: {"$eq": value}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _sanitize_match_query(query: str) -> str:
        if not query:
            return "*"
        cleaned = query.replace('"', ' ')
        for ch in "?.,;:!()[]{}<>|/\\":
            cleaned = cleaned.replace(ch, " ")
        cleaned = " ".join(cleaned.split())
        return cleaned or "*"

    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        meta = dict(metadata or {})
        meta.setdefault("provider", "hybrid")
        meta.setdefault("ingest_timestamp", current_timestamp())
        if "tags" in meta and isinstance(meta["tags"], str):
            try:
                meta["tags"] = [tag.strip() for tag in meta["tags"].split(",") if tag.strip()]
            except Exception:
                meta["tags"] = [meta["tags"]]
        return meta

    def _reciprocal_rank_fusion(self, result_sets: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for result_set in result_sets:
            for rank, item in enumerate(result_set):
                doc_id = item.get("id")
                if not doc_id:
                    continue
                fused.setdefault(doc_id, {
                    "id": doc_id,
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "score": 0.0,
                })
                fused[doc_id]["score"] += 1.0 / (k + rank + 1)

        return sorted(fused.values(), key=lambda hit: hit["score"], reverse=True)

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._rerank_enabled or not self._rerank_available or not candidates:
            for item in candidates:
                item.setdefault("provider", "hybrid")
            return candidates

        from concurrent.futures import Future

        if self._rerank_executor is None:
            return candidates

        future: Future = self._rerank_executor.submit(self._rerank_sync, query, candidates)
        try:
            return future.result()
        except Exception as e:
            logger.warning(f"Hybrid reranking failed: {e}")
            for item in candidates:
                item.setdefault("provider", "hybrid")
            return candidates

    def _rerank_sync(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        top_candidates = candidates[: self._rerank_top_k]
        scored: List[Dict[str, Any]] = []
        for item in top_candidates:
            doc_id = item.get("id") or ""
            content = item.get("content", "")
            cache_key = f"{doc_id}:{hash(content)}:{hash(query)}"
            score = self._score_cache.get(cache_key)
            if score is None:
                score = float(self._score_with_cross_encoder(self._rerank_model_name, query, content))
                if len(self._score_cache) > 2048:
                    self._score_cache.clear()
                self._score_cache[cache_key] = score
            reranked_item = dict(item)
            reranked_item["score"] = score
            reranked_item.setdefault("provider", "hybrid")
            scored.append(reranked_item)

        scored.sort(key=lambda hit: hit["score"], reverse=True)
        seen = set(hit["id"] for hit in scored)
        tail = [item for item in candidates if item.get("id") not in seen]
        for item in tail:
            item.setdefault("provider", "hybrid")
        return scored + tail

    @classmethod
    @lru_cache(maxsize=2)
    def _load_cross_encoder(cls, model_name: str):
        from sentence_transformers import CrossEncoder

        with cls._cross_encoder_lock:
            return CrossEncoder(model_name)

    @classmethod
    @lru_cache(maxsize=4096)
    def _score_with_cross_encoder(cls, model_name: str, query: str, document: str) -> float:
        model = cls._load_cross_encoder(model_name)
        score = model.predict([(query, document)])
        return float(score[0]) if hasattr(score, "__getitem__") else float(score)

    @staticmethod
    def _infer_embedding_dims(model_name: str) -> int:
        lookup = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-002": 1536,
        }
        model_lower = (model_name or "").lower()
        for key, dims in lookup.items():
            if key in model_lower:
                return dims
        return 1536
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
            # Ensure embedder has embedding_dims required by some Mem0 versions
            try:
                if embedder_cfg and isinstance(embedder_cfg, dict):
                    emb_conf = embedder_cfg.setdefault("config", {})
                    if "embedding_dims" not in emb_conf:
                        model = str(emb_conf.get("model", "")).lower()
                        dims_map = {
                            "text-embedding-3-large": 3072,
                            "text-embedding-3-small": 1536,
                            "text-embedding-ada-002": 1536,
                            "text-embedding-002": 1536,
                        }
                        # Fallback if unknown
                        dims = next((v for k, v in dims_map.items() if k in model), 1536)
                        emb_conf["embedding_dims"] = dims
            except Exception:
                pass
            
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
                
                # Prefer factory initializer for compatibility and add graceful fallback
                def _init_mem0(cfg):
                    try:
                        return Memory.from_config(config_dict=cfg)
                    except TypeError:
                        # Backward compatibility with older signatures
                        return Memory.from_config(cfg)

                try:
                    self._mem0_client = _init_mem0(mem0_config)
                except Exception as e:
                    # If vector store is configured and connection fails, retry without vector store
                    has_vector = "vector_store" in mem0_config
                    if has_vector:
                        logger.warning(
                            f"Mem0 vector store initialization failed ({e}); retrying without vector store"
                        )
                        cfg_no_vector = {k: v for k, v in mem0_config.items() if k != "vector_store"}
                        self._mem0_client = _init_mem0(cfg_no_vector)
                    else:
                        raise
                
                self.is_initialized = True
                logger.info("Mem0 memory provider initialized successfully")
            except ImportError:
                raise ProviderError("mem0ai package required for Mem0 provider")
        except Exception as e:
            raise ProviderError(f"Failed to initialize Mem0 provider: {str(e)}")
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in Mem0."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            user_id = metadata.get("user_id", "default") if metadata else "default"
            agent_id = metadata.get("agent_id") if metadata else None
            run_id = metadata.get("run_id") if metadata else None
            
            # Prefer explicit parameters supported by Mem0 for better partitioning
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
                else:
                    formatted_results.append({
                        "id": None,
                        "content": str(item),
                        "metadata": {},
                        "provider": "mem0",
                        "score": 0,
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
        rerank: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve content from Mem0 with filters and optional rerank."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        try:
            search_kwargs: Dict[str, Any] = {"query": query, "limit": limit}
            if user_id is not None:
                search_kwargs["user_id"] = user_id
            if agent_id is not None:
                search_kwargs["agent_id"] = agent_id
            if run_id is not None:
                search_kwargs["run_id"] = run_id
            if rerank is not None:
                search_kwargs["rerank"] = rerank
            # Pass through any extra kwargs (e.g., custom filters)
            search_kwargs.update(kwargs)

            raw = self._mem0_client.search(**search_kwargs)
            # Mem0 may return a list of dicts/strings or a dict with 'results'
            if isinstance(raw, dict):
                results = raw.get("results") or raw.get("memories") or []
            else:
                results = raw

            formatted_results: List[Dict[str, Any]] = []
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
                else:
                    # Fallback safe conversion
                    formatted_results.append({
                        "id": None,
                        "content": str(item),
                        "metadata": {},
                        "provider": "mem0",
                        "score": 0,
                    })
            return formatted_results
        except Exception as e:
            raise ProviderError(f"Failed to retrieve (filtered) from Mem0: {str(e)}")
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update content in Mem0."""
        if not self.is_initialized:
            raise ProviderError("Mem0 provider not initialized")
        
        try:
            # Mem0 doesn't have direct update, so we delete and add
            self.delete(ref_id)
            self.store(content, metadata)
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
            logger.warning(f"Failed to delete Mem0 entry {ref_id}: {str(e)}")
    
    def cleanup(self) -> None:
        """Cleanup Mem0 resources."""
        self._mem0_client = None
        self.is_initialized = False


class MemoryManager:
    """Manager for different memory providers."""
    
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
            elif self.config.provider == MemoryProvider.MONGODB:
                self.provider = MongoDBMemoryProvider(self.config.embedder)
            elif self.config.provider == MemoryProvider.MEM0:
                self.provider = Mem0MemoryProvider(self.config.embedder)
            elif self.config.provider == MemoryProvider.HYBRID:
                self.provider = HybridRAGMemoryProvider(self.config.embedder)
            else:
                raise MemoryError(f"Unsupported memory provider: {self.config.provider}")
            
            # Initialize the provider
            provider_config = self.config.config.copy()
            
            # Add provider-specific config
            if self.config.provider == MemoryProvider.RAG:
                provider_config.update({
                    "short_db": self.config.short_db,
                    "long_db": self.config.long_db,
                    "rag_db_path": self.config.rag_db_path
                })
            
            self.provider.initialize(provider_config)
            logger.info(f"Memory manager initialized with {self.config.provider.value} provider")
        except Exception as e:
            raise MemoryError(f"Failed to initialize memory manager: {str(e)}")
    
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
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve content with user/agent/run filters when provider supports it."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        # Prefer provider's filtered method if available
        retrieve_f = getattr(self.provider, "retrieve_filtered", None)
        if callable(retrieve_f):
            return retrieve_f(
                query,
                limit=limit,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                **kwargs,
            )
        # Fallback
        return self.provider.retrieve(query, limit)
    
    def update(self, ref_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update content in memory."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        self.provider.update(ref_id, content, metadata)
    
    def delete(self, ref_id: str) -> None:
        """Delete content from memory."""
        if not self.provider:
            raise MemoryError("Memory provider not initialized")
        
        self.provider.delete(ref_id)
    
    def health_check(self) -> bool:
        """Check memory provider health."""
        if not self.provider:
            return False
        
        return self.provider.health_check()
    
    def cleanup(self) -> None:
        """Cleanup memory resources."""
        if self.provider:
            self.provider.cleanup()
            self.provider = None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider": self.config.provider.value,
            "use_embedding": self.config.use_embedding,
            "initialized": self.provider is not None and self.provider.health_check(),
            "embedder": {
                "provider": self.config.embedder.provider,
                "model": self.config.embedder.config.get("model")
            } if self.config.embedder else None
        }
    
    def switch_provider(self, new_config: MemoryConfig) -> None:
        """Switch to a different memory provider."""
        # Cleanup current provider
        self.cleanup()
        
        # Update config and initialize new provider
        self.config = new_config
        self._initialize_provider()
        
        logger.info(f"Switched to {new_config.provider.value} memory provider")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        # Basic stats - could be extended per provider
        return {
            "provider": self.config.provider.value,
            "healthy": self.health_check(),
            "initialized": self.provider is not None
        }
