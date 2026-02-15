"""GraphRAG lookup tool for agents.

Provides a LangChain StructuredTool wrapper around the hybrid memory
retrieval system so that LangGraph ReAct agents can invoke it with
full parameter schemas.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)

_BROAD_PATTERNS = {"all", "everything", "help", "*", "?", "any"}


def _parse_list(value: Optional[str | List[str]]) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    return None


class GraphRAGInput(BaseModel):
    """Input schema for the GraphRAG lookup tool."""

    query: str = Field(
        description="Natural language query to search for relevant documents. Must be 3+ characters.",
    )
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated tags to filter documents (e.g. 'standard, safety').",
    )
    documents: Optional[str] = Field(
        default=None,
        description="Comma-separated document IDs to prioritize.",
    )
    sections: Optional[str] = Field(
        default=None,
        description="Comma-separated section identifiers to filter by.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of fragments to return (1-50).",
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: Any) -> str:
        """Reject empty or too-short queries."""
        v = str(v).strip()
        if not v:
            raise ValueError("query cannot be empty")
        if len(v) < 3:
            raise ValueError(f"query must be at least 3 characters, got {len(v)}")
        return v


class GraphRAGTool:
    """Callable wrapper that queries MemoryManager.retrieve_with_graph."""

    def __init__(
        self,
        memory_manager,
        *,
        default_user_id: Optional[str] = None,
        default_run_id: Optional[str] = None,
    ):
        self.memory_manager = memory_manager
        self.default_user_id = default_user_id
        self.default_run_id = default_run_id

    def __call__(
        self,
        query: str,
        *,
        tags: Optional[str | List[str]] = None,
        documents: Optional[str | List[str]] = None,
        sections: Optional[str | List[str]] = None,
        top_k: int = 5,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        tags_list = _parse_list(tags)
        documents_list = _parse_list(documents)
        sections_list = _parse_list(sections)

        user = user_id or self.default_user_id
        run = run_id or self.default_run_id

        logger.info(
            "graph_rag_lookup query=%r tags=%s documents=%s sections=%s user=%s run=%s",
            query,
            tags_list,
            documents_list,
            sections_list,
            user,
            run,
        )

        if query.strip().lower() in _BROAD_PATTERNS:
            logger.warning(
                "graph_rag_lookup: query %r may be too broad; consider more specific terms",
                query,
            )

        try:
            top = int(top_k)
        except (TypeError, ValueError):
            top = 5

        results = self.memory_manager.retrieve_with_graph(
            query,
            limit=max(1, top),
            tags=tags_list,
            document_ids=documents_list,
            sections=sections_list,
            user_id=user,
            run_id=run,
            agent_id=agent_id,
        )

        logger.info(
            "graph_rag_lookup returned %d hits (top=%d)",
            len(results),
            top,
        )

        if not results:
            return "No relevant fragments found in the document database."

        formatted: List[str] = ["GraphRAG Results:"]
        for item in results[:top]:
            meta = item.get("metadata", {}) or {}
            document_id = meta.get("document_id") or meta.get("id")
            section = meta.get("section") or meta.get("section_title")
            source_url = meta.get("source_url")
            score = item.get("score")
            summary = item.get("content") or ""
            snippet = summary.replace("\n", " ")
            if len(snippet) > 320:
                snippet = snippet[:317] + "..."

            formatted.append(
                "- document: {doc} | section: {section} | score: {score:.3f}\n  {snippet}{source}".format(
                    doc=document_id or "unknown",
                    section=section or "-",
                    score=score if score is not None else 0.0,
                    snippet=snippet,
                    source=f"\n  source: {source_url}" if source_url else "",
                )
            )

        return "\n".join(formatted)


def create_graph_rag_tool(
    memory_manager,
    *,
    default_user_id: Optional[str] = None,
    default_run_id: Optional[str] = None,
) -> StructuredTool:
    """Return a LangChain StructuredTool wrapping GraphRAG retrieval.

    The returned tool can be used directly with LangGraph's
    ``create_react_agent`` and preserves the full parameter schema
    (query, tags, documents, sections, top_k) for the LLM to invoke.
    """

    tool_instance = GraphRAGTool(
        memory_manager,
        default_user_id=default_user_id,
        default_run_id=default_run_id,
    )

    def _graph_rag_lookup(
        query: str,
        tags: Optional[str] = None,
        documents: Optional[str] = None,
        sections: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        return tool_instance(
            query,
            tags=tags,
            documents=documents,
            sections=sections,
            top_k=top_k,
        )

    return StructuredTool.from_function(
        func=_graph_rag_lookup,
        name="graph_rag_lookup",
        description=(
            "Retrieves fragments from technical documents using hybrid search "
            "(vector + lexical + graph) with re-ranking. "
            "Returns a textual summary with fragments and citations.\n\n"
            "Examples:\n"
            "  graph_rag_lookup(query='IFC wall entity properties', tags='ifc,schema', top_k=3)\n"
            "  graph_rag_lookup(query='safety compliance requirements', documents='doc_001')\n\n"
            "Constraints:\n"
            "- query (required): Specific search terms, 3+ characters\n"
            "- tags (optional): Comma-separated filter tags\n"
            "- documents (optional): Comma-separated document IDs to prioritize\n"
            "- sections (optional): Comma-separated section IDs\n"
            "- top_k (optional): 1-50, default 5\n\n"
            "Avoid overly broad queries like 'everything' or 'all documents'."
        ),
        args_schema=GraphRAGInput,
    )
