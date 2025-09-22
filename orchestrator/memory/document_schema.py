"""Common document and chunk metadata schema utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional


ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def current_timestamp() -> str:
    return datetime.utcnow().strftime(ISO_TIMESTAMP_FORMAT)


@dataclass
class DocumentMetadata:
    """High-level metadata describing a technical document."""

    document_id: str
    title: str
    discipline: Optional[str] = None
    doc_type: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[str] = None
    language: Optional[str] = None
    source_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "document_id": self.document_id,
            "title": self.title,
            "discipline": self.discipline,
            "doc_type": self.doc_type,
            "version": self.version,
            "effective_date": self.effective_date,
            "language": self.language,
            "source_url": self.source_url,
            "tags": self.tags or None,
        }
        data.update(self.extra)
        return {k: v for k, v in data.items() if v not in (None, "", [], {})}


@dataclass
class ChunkMetadata:
    """Metadata describing a specific chunk/section within a document."""

    chunk_id: str
    content_type: str = "document_chunk"
    section: Optional[str] = None
    page_range: Optional[str] = None
    order: Optional[int] = None
    embedding_model: Optional[str] = None
    ingest_timestamp: str = field(default_factory=current_timestamp)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "content_type": self.content_type,
            "section": self.section,
            "page_range": self.page_range,
            "order": self.order,
            "embedding_model": self.embedding_model,
            "ingest_timestamp": self.ingest_timestamp,
        }
        data.update(self.extra)
        return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def merge_metadata(document: DocumentMetadata, chunk: ChunkMetadata, *, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge document and chunk metadata into a single dictionary."""

    merged: Dict[str, Any] = {}
    if base:
        merged.update(base)
    merged.update(document.to_dict())
    merged.update(chunk.to_dict())
    return merged


def sanitize_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata only contains Chroma-compatible primitives."""

    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            sanitized[key] = str(value)
        else:
            sanitized[key] = str(value)
    return sanitized


def default_conversation_metadata(user_id: str, run_id: str, agent_id: str, *, content_hash: Optional[str] = None) -> Dict[str, Any]:
    """Build minimal metadata for conversational turns."""

    data = {
        "content_type": "conversation",
        "user_id": user_id,
        "run_id": run_id,
        "agent_id": agent_id,
        "ingest_timestamp": current_timestamp(),
    }
    if content_hash:
        data["chunk_id"] = content_hash
    else:
        data["chunk_id"] = f"conv-{uuid.uuid4()}"
    return data
