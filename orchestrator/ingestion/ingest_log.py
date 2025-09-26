"""Simple ingestion log stored in SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..memory.document_schema import current_timestamp


DEFAULT_LOG_PATH = Path(".praison/ingestion_log.db")


@dataclass
class IngestionRecord:
    document_id: str
    source_path: str
    source_hash: str
    last_refresh: str


class IngestionLog:
    def __init__(self, db_path: Path = DEFAULT_LOG_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._ensure_tables()

    def close(self) -> None:
        self._conn.close()

    def _ensure_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_log (
                document_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                last_refresh TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, document_id: str) -> Optional[IngestionRecord]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT document_id, source_path, source_hash, last_refresh FROM ingest_log WHERE document_id = ?",
            (document_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return IngestionRecord(*row)

    def needs_ingest(self, document_id: str, source_hash: str) -> bool:
        record = self.get(document_id)
        if record is None:
            return True
        return record.source_hash != source_hash

    def record(self, document_id: str, source_path: Path, source_hash: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO ingest_log (document_id, source_path, source_hash, last_refresh)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(document_id)
            DO UPDATE SET source_path=excluded.source_path,
                          source_hash=excluded.source_hash,
                          last_refresh=excluded.last_refresh
            """,
            (document_id, str(source_path), source_hash, current_timestamp()),
        )
        self._conn.commit()

