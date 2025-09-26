"""GraphRAG lookup tool for agents."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any


logger = logging.getLogger(__name__)


def _parse_list(value: Optional[str | List[str]]) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    return None


class GraphRAGTool:
    """Callable wrapper that queries MemoryManager.retrieve_with_graph."""

    def __init__(self, memory_manager, *, default_user_id: Optional[str] = None, default_run_id: Optional[str] = None):
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
            return "No se encontraron fragmentos relevantes en la base documental." 

        formatted: List[str] = ["Resultados GraphRAG:"]
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
                "- documento: {doc} | sección: {section} | score: {score:.3f}\n  {snippet}{source}".format(
                    doc=document_id or "desconocido",
                    section=section or "-",
                    score=score if score is not None else 0.0,
                    snippet=snippet,
                    source=f"\n  fuente: {source_url}" if source_url else "",
                )
            )

        return "\n".join(formatted)


def create_graph_rag_tool(memory_manager, *, default_user_id: Optional[str] = None, default_run_id: Optional[str] = None):
    """Return a callable tool ready to register in PraisonAI agents."""

    tool = GraphRAGTool(memory_manager, default_user_id=default_user_id, default_run_id=default_run_id)

    def graph_rag_lookup(
        query: str,
        tags: Optional[str | List[str]] = None,
        documents: Optional[str | List[str]] = None,
        sections: Optional[str | List[str]] = None,
        top_k: int = 5,
    ) -> str:
        """Recupera fragmentos de documentos técnicos usando GraphRAG.

        Parámetros:
            query: consulta en lenguaje natural.
            tags: lista o cadena separada por comas con etiquetas (p.ej. "norma, seguridad").
            documents: IDs de documentos a priorizar.
            sections: identificadores de secciones específicas.
            top_k: número máximo de fragmentos a devolver.

        Devuelve un resumen textual con fragmentos y citas.
        """

        return tool(
            query,
            tags=tags,
            documents=documents,
            sections=sections,
            top_k=top_k,
        )

    graph_rag_lookup.__name__ = "graph_rag_lookup"
    return graph_rag_lookup
