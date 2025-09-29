"""
Hybrid RAG memory provider combining vector search, lexical BM25 and cross-encoder reranking.

Extracted from the monolithic memory_manager.py to maintain modular architecture.
"""

from typing import Dict, Any, Optional, List, Set
import logging
import sqlite3
import json
import time
import uuid
import threading
from pathlib import Path
from functools import lru_cache

from .base import BaseMemoryProvider
from ..document_schema import sanitize_for_chroma, current_timestamp
from ...core.config import EmbedderConfig
from ...core.exceptions import ProviderError
from ...core.embedding_utils import get_embedding_dimensions

logger = logging.getLogger(__name__)


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
        self._embedding_dimensions = get_embedding_dimensions(self._embedding_model)
        self._graph_config: Dict[str, Any] = {}
        self._graph_driver = None
        self._graph_session_cache = threading.local()

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

        logger.info(
            "Hybrid retrieve query=%r limit=%d filters=%s",
            query,
            limit,
            filters or "{}",
        )

        vector_results = self._search_vector(query, filters, limit)
        self._log_hits("vector", vector_results)
        lexical_results = self._search_lexical(query, filters, limit)
        self._log_hits("lexical", lexical_results)

        fused = self._reciprocal_rank_fusion([vector_results, lexical_results])
        self._log_hits("fused", fused)
        reranked = self._rerank(query, fused)
        self._log_hits("reranked", reranked)

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
                close = getattr(self._vector_client, 'close', None)
                if callable(close):
                    close()
            except Exception:
                pass
            self._vector_client = None
            self._vector_collection = None
        if self._rerank_executor:
            self._rerank_executor.shutdown(wait=False)
            self._rerank_executor = None
        if self._graph_driver:
            session = getattr(self._graph_session_cache, "session", None)
            if session:
                try:
                    session.close()
                except Exception:
                    pass
                self._graph_session_cache.session = None
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
        # Expire entries based on explicit expires_at metadata
        cursor.execute(
            "SELECT rowid FROM documents WHERE json_extract(metadata, '$.expires_at') IS NOT NULL"
            " AND json_extract(metadata, '$.expires_at') < ?",
            (current_timestamp(),),
        )
        rows = cursor.fetchall()
        if rows:
            rowids = [row[0] for row in rows]
            cursor.executemany("DELETE FROM documents WHERE rowid = ?", [(rid,) for rid in rowids])
            cursor.executemany("DELETE FROM documents_fts WHERE rowid = ?", [(rid,) for rid in rowids])

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
        section_id = metadata.get("section_id") or (
            f"{metadata.get('document_id')}::{metadata.get('section')}"
            if metadata.get("document_id") and metadata.get("section")
            else None
        )
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
            "section_id": section_id,
            "section_title": metadata.get("section"),
            "references": metadata.get("references") if isinstance(metadata.get("references"), list) else [],
        }

        try:
            self._graph_execute_write(self._graph_write_tx, params)
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

        tags = params.get("tags") or []
        if tags:
            tx.run(
                """
                UNWIND $tags AS tag
                MERGE (t:Tag {name: tag})
                MERGE (d)-[:HAS_TAG]->(t)
                """,
                {"tags": tags, "document_id": params.get("document_id")},
            )

        if params.get("section_id"):
            tx.run(
                """
                MATCH (d:Document {document_id: $document_id})
                MERGE (s:Section {section_id: $section_id})
                SET s.title = $section_title
                MERGE (s)-[:BELONGS_TO]->(d)
                MERGE (c:Chunk {chunk_id: $chunk_id})-[:PART_OF]->(s)
                """,
                params,
            )

        if params.get("references"):
            tx.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                UNWIND $references AS ref_id
                MERGE (ref:Document {document_id: ref_id})
                MERGE (c)-[:REFERENCES]->(ref)
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
            elif isinstance(value, (list, tuple, set)):
                clauses.append({key: {"$in": list(value)}})
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
        elif "tags" in meta and not isinstance(meta["tags"], list):
            meta["tags"] = [meta["tags"]]
        else:
            meta.setdefault("tags", [])
        refs = meta.get("references")
        if isinstance(refs, str):
            meta["references"] = [ref.strip() for ref in refs.split(",") if ref.strip()]
        elif isinstance(refs, list):
            meta["references"] = refs
        elif not isinstance(refs, list):
            meta["references"] = []
        return meta

    def _graph_execute_write(self, func, params: Dict[str, Any]) -> None:
        driver = self._graph_driver
        if not driver:
            return
        session = getattr(self._graph_session_cache, "session", None)
        if session is None or session.closed():
            session = driver.session()
            self._graph_session_cache.session = session
        session.execute_write(func, params)

    def _graph_query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        driver = self._graph_driver
        if not driver:
            return []
        with driver.session() as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def _graph_candidate_chunks(
        self,
        *,
        tags: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
    ) -> List[str]:
        driver = self._graph_driver
        if not driver:
            return []

        candidates: Optional[Set[str]] = None

        if tags:
            records = self._graph_query(
                """
                MATCH (c:Chunk)-[:BELONGS_TO]->(:Document)-[:HAS_TAG]->(t:Tag)
                WHERE t.name IN $tags
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                {"tags": tags},
            )
            tag_set = {row["chunk_id"] for row in records}
            candidates = tag_set if candidates is None else candidates & tag_set

        if document_ids:
            records = self._graph_query(
                """
                MATCH (c:Chunk)-[:BELONGS_TO]->(d:Document)
                WHERE d.document_id IN $documents
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                {"documents": document_ids},
            )
            doc_set = {row["chunk_id"] for row in records}
            candidates = doc_set if candidates is None else candidates & doc_set

        if sections:
            records = self._graph_query(
                """
                MATCH (c:Chunk)-[:PART_OF]->(s:Section)
                WHERE s.section_id IN $sections OR s.title IN $sections
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                {"sections": sections},
            )
            section_set = {row["chunk_id"] for row in records}
            candidates = section_set if candidates is None else candidates & section_set

        if candidates is None:
            return []
        return list(candidates)

    def retrieve_with_graph(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if run_id:
            filters["run_id"] = run_id
        if agent_id:
            filters["agent_id"] = agent_id

        logger.info(
            "Hybrid graph retrieve query=%r limit=%d filters=%s tags=%s documents=%s sections=%s",
            query,
            limit,
            filters or "{}",
            tags or [],
            document_ids or [],
            sections or [],
        )

        candidate_ids = self._graph_candidate_chunks(
            tags=tags,
            document_ids=document_ids,
            sections=sections,
        )

        if not candidate_ids:
            logger.info("Hybrid graph retrieve: no graph matches, falling back to standard retrieval")
            return self.retrieve_filtered(
                query,
                limit=limit,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )

        logger.info(
            "Hybrid graph retrieve candidates (n=%d): %s",
            len(candidate_ids),
            candidate_ids[:5] if len(candidate_ids) > 5 else candidate_ids,
        )

        vector_filters = dict(filters)
        vector_filters["chunk_id"] = candidate_ids
        lexical_filters = dict(filters)

        vector_results = self._search_vector(query, vector_filters, max(limit, self._rerank_top_k))
        self._log_hits("graph_vector", vector_results)
        lexical_results = self._search_lexical(query, lexical_filters, max(limit, self._rerank_top_k))
        lexical_results = [res for res in lexical_results if res.get("id") in candidate_ids]
        self._log_hits("graph_lexical", lexical_results)

        fused = self._reciprocal_rank_fusion([vector_results, lexical_results])
        self._log_hits("graph_fused", fused)
        reranked = self._rerank(query, fused)
        self._log_hits("graph_reranked", reranked)
        return reranked[:limit]

    @staticmethod
    def _log_hits(stage: str, hits: List[Dict[str, Any]], *, top: int = 3) -> None:
        if not hits:
            logger.info("Hybrid %s: 0 hits", stage)
            return

        summary: List[str] = []
        for item in hits[:top]:
            score = item.get("score")
            if isinstance(score, (int, float)):
                score_repr = f"{score:.3f}"
            else:
                score_repr = str(score)
            summary.append(
                f"{item.get('id')} (score={score_repr}, source={item.get('source')})"
            )

        remaining = len(hits) - top
        if remaining > 0:
            summary.append(f"... +{remaining} more")

        logger.info("Hybrid %s: %s", stage, "; ".join(summary))

    def resync_graph(self) -> int:
        if not self._graph_driver or not self._lex_conn:
            return 0

        with self._lex_lock:
            cursor = self._lex_conn.cursor()
            rows = cursor.execute("SELECT id, content, metadata FROM documents").fetchall()

        count = 0
        for row_id, content, metadata_json in rows:
            if not metadata_json:
                continue
            try:
                meta = json.loads(metadata_json)
            except json.JSONDecodeError:
                continue
            if meta.get("content_type") != "document_chunk":
                continue
            meta = self._normalize_metadata(meta)
            self._upsert_graph(row_id, content, meta)
            count += 1
        return count

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

    def create_graph_tool(self, *, default_user_id: str, default_run_id: str):
        """Create a GraphRAG lookup tool if graph support is enabled."""
        if not self._graph_driver:
            raise ProviderError("Graph operations not available - Neo4j not configured")
        
        def graph_rag_lookup(
            query: str,
            *,
            top_k: int = 5,
            tags: Optional[List[str]] = None,
            document_ids: Optional[List[str]] = None,
            sections: Optional[List[str]] = None,
            user_id: Optional[str] = None,
            run_id: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
            """Lookup documents using graph-enhanced retrieval."""
            return self.retrieve_with_graph(
                query,
                limit=top_k,
                tags=tags,
                document_ids=document_ids,
                sections=sections,
                user_id=user_id or default_user_id,
                run_id=run_id or default_run_id,
                agent_id=None,
            )
        
        # Set tool metadata
        graph_rag_lookup.__name__ = "graph_rag_lookup"
        graph_rag_lookup.__doc__ = """Retrieve relevant documents using hybrid search with graph enhancement.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return (default: 5)
            tags: Filter by document tags
            document_ids: Filter by specific document IDs
            sections: Filter by document sections
            user_id: Filter by user ID (defaults to session user)
            run_id: Filter by run ID (defaults to session run)
            
        Returns:
            List of documents with content, metadata, and relevance scores
        """
        
        return graph_rag_lookup