"""Graph synchronization helpers for engineering documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GraphRelationships:
    document_id: str
    references: List[str]
    sections: List[Dict[str, str]]


def build_reference_relations(metadata: Dict[str, any]) -> GraphRelationships:
    references = metadata.get("references") or []
    sections = metadata.get("sections") or []
    doc_id = metadata.get("document_id") or metadata.get("id")
    return GraphRelationships(document_id=doc_id, references=references, sections=sections)

