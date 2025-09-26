"""Simple ETL utility to ingest engineering PDFs into the hybrid memory stack.

Usage:
    uv run python -m orchestrator.ingestion.sample_loader --input docs/manual.pdf \
        --discipline mechanical --doc-type standard --version 2024

Dependencies:
    pip install pypdf
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Dict, Any

from orchestrator.cli.main import resolve_memory_config
from orchestrator.memory.document_schema import (
    ChunkMetadata,
    DocumentMetadata,
    merge_metadata,
)
from orchestrator.memory.memory_manager import MemoryManager

from .extractors import extract_pdf_text
from .ingest_log import IngestionLog
from .metadata_loader import load_metadata_file


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> Iterable[str]:
    tokens = text.split()
    if not tokens:
        return []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        yield " ".join(tokens[start:end])
        if end == len(tokens):
            break
        start = max(0, end - overlap)


def compute_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def compute_expiry(days: int) -> str:
    return (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def load_manifest(metadata_path: str | None, pdf_path: Path) -> Dict[str, Any]:
    if not metadata_path:
        return {}
    manifest_path = Path(metadata_path)
    if manifest_path.is_dir():
        for suffix in (".yaml", ".yml", ".json"):
            candidate = manifest_path / (pdf_path.stem + suffix)
            if candidate.exists():
                return load_metadata_file(candidate)
        return {}
    return load_metadata_file(manifest_path)


def ingest_pdf(
    args: argparse.Namespace,
    pdf_path: Path,
    memory_manager: MemoryManager,
    log: IngestionLog,
) -> None:
    pages = extract_pdf_text(pdf_path)
    if not any(pages):
        print(f"⚠️  No extractable text found in {pdf_path.name}, skipping")
        return

    document_id = args.document_id or pdf_path.stem
    source_hash = compute_hash(pdf_path)

    if not args.force and not log.needs_ingest(document_id, source_hash):
        print(f"⏩  Skipping {pdf_path.name}; no changes detected")
        return

    manifest_meta = load_manifest(args.metadata, pdf_path)

    doc_meta = DocumentMetadata(
        document_id=document_id,
        title=manifest_meta.get("title") or args.title or pdf_path.stem.replace("_", " "),
        discipline=manifest_meta.get("discipline") or args.discipline,
        doc_type=manifest_meta.get("doc_type") or args.doc_type,
        version=manifest_meta.get("version") or args.version,
        effective_date=manifest_meta.get("effective_date") or args.effective_date,
        language=manifest_meta.get("language") or args.language or "es",
        source_url=manifest_meta.get("source_url") or args.source_url,
        tags=(manifest_meta.get("tags") or args.tags or []),
        extra={
            k: v
            for k, v in manifest_meta.items()
            if k
            not in {"title", "discipline", "doc_type", "version", "effective_date", "language", "source_url", "tags", "retention_days"}
        },
    )

    expires_at = None
    retention_days = manifest_meta.get("retention_days") or args.retention_days
    if retention_days:
        expires_at = compute_expiry(int(retention_days))

    chunk_order = 0
    for page_index, page_text in enumerate(pages, start=1):
        for segment in chunk_text(page_text, chunk_size=args.chunk_size, overlap=args.chunk_overlap):
            chunk_order += 1
            chunk_meta = ChunkMetadata(
                chunk_id=f"{document_id}-chunk-{uuid.uuid4().hex[:8]}",
                section=args.section,
                page_range=str(page_index),
                order=chunk_order,
                embedding_model=memory_manager.config.embedder.config.get("model"),
            )
            metadata = merge_metadata(doc_meta, chunk_meta)
            metadata["content_type"] = "document_chunk"
            metadata["user_id"] = args.user
            metadata["run_id"] = args.run
            if args.agent:
                metadata["agent_id"] = args.agent
            if expires_at:
                metadata["expires_at"] = expires_at
            references = manifest_meta.get("references")
            if references:
                metadata["references"] = references
            memory_manager.store(segment, metadata)

    log.record(document_id, pdf_path, source_hash)
    print(f"✅ Ingested {chunk_order} chunks from {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into hybrid memory")
    parser.add_argument("--input", required=True, help="Path to PDF file or directory")
    parser.add_argument("--discipline", help="Engineering discipline (mechanical, electrical, etc.)")
    parser.add_argument("--doc-type", dest="doc_type", help="Document type (standard, manual, etc.)")
    parser.add_argument("--version", help="Document version")
    parser.add_argument("--effective-date", dest="effective_date", help="Effective date (YYYY-MM-DD)")
    parser.add_argument("--language", help="Language code (default es)")
    parser.add_argument("--title", help="Override document title")
    parser.add_argument("--source-url", dest="source_url", help="Original source URL")
    parser.add_argument("--tags", nargs="*", help="List of tags applied to the document")
    parser.add_argument("--document-id", help="Custom document identifier")
    parser.add_argument("--section", help="Section label applied to all chunks")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--user", default=os.getenv("ORCH_USER", "global"), help="User partition for memory (default global)")
    parser.add_argument("--run", default=os.getenv("ORCH_RUN", "library"), help="Run/session partition (default library)")
    parser.add_argument("--agent", help="Agent identifier assigned to chunks")
    parser.add_argument("--metadata", help="Path to JSON/YAML manifest or directory")
    parser.add_argument("--retention-days", type=int, help="Retention window in days (sets expires_at)")
    parser.add_argument("--force", action="store_true", help="Force ingestion even if hash unchanged")
    parser.add_argument(
        "--memory-provider",
        choices=["mem0", "hybrid"],
        default=os.getenv("ORCH_MEMORY_PROVIDER", "hybrid"),
        help="Memory provider to use (default reads ORCH_MEMORY_PROVIDER)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    memory_config = resolve_memory_config(args.memory_provider)
    manager = MemoryManager(memory_config)
    ingest_log = IngestionLog()

    input_path = Path(args.input)
    if input_path.is_dir():
        pdf_files = sorted(path for path in input_path.glob("**/*.pdf"))
    else:
        pdf_files = [input_path]

    if not pdf_files:
        print(f"⚠️  No PDF files found under {input_path}")
        return

    for pdf in pdf_files:
        ingest_pdf(args, pdf, manager, ingest_log)

    ingest_log.close()


if __name__ == "__main__":
    main()
