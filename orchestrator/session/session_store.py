"""
SQLite-based persistence for session and user data.

Provides thread-safe storage and retrieval of session contexts with automatic
cleanup of old sessions.
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .user_context import SessionContext, UserContext

logger = logging.getLogger(__name__)


class SessionStore:
    """
    SQLite-based persistent storage for sessions and users.

    This class provides thread-safe CRUD operations for session management
    with automatic schema initialization and session cleanup.

    Attributes:
        db_path: Path to SQLite database file
        max_session_age_days: Maximum age for sessions before cleanup (default: 30)
    """

    def __init__(
        self,
        db_path: str = ".praison/sessions.db",
        max_session_age_days: int = 30,
    ):
        """
        Initialize session store with database path.

        Args:
            db_path: Path to SQLite database file (created if not exists)
            max_session_age_days: Maximum age for sessions before cleanup
        """
        self.db_path = db_path
        self.max_session_age_days = max_session_age_days
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.

        Returns:
            SQLite connection object

        Note:
            Creates new connection if none exists for this thread.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _initialize_schema(self) -> None:
        """
        Initialize database schema if not exists.

        Creates tables for users and sessions with proper indexes.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_sessions INTEGER DEFAULT 0,
                    preferences TEXT
                )
                """
            )

            # Sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    metadata TEXT,
                    turn_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )

            # Indexes for performance
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON sessions(user_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
                ON sessions(last_activity)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_is_active
                ON sessions(is_active)
                """
            )

            conn.commit()
            logger.info(f"Session store initialized at {self.db_path}")

    def save_user(self, user: UserContext) -> None:
        """
        Save or update user context.

        Args:
            user: UserContext object to persist

        Thread-safe operation using RLock.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO users
                (user_id, created_at, last_seen, total_sessions, preferences)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.created_at.isoformat(),
                    user.last_seen.isoformat(),
                    user.total_sessions,
                    json.dumps(user.preferences),
                ),
            )

            conn.commit()
            logger.debug(f"Saved user: {user.user_id}")

    def get_user(self, user_id: str) -> Optional[UserContext]:
        """
        Retrieve user context by ID.

        Args:
            user_id: Unique user identifier

        Returns:
            UserContext object if found, None otherwise
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT user_id, created_at, last_seen, total_sessions, preferences
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return UserContext.from_dict(
                {
                    "user_id": row["user_id"],
                    "created_at": row["created_at"],
                    "last_seen": row["last_seen"],
                    "total_sessions": row["total_sessions"],
                    "preferences": json.loads(row["preferences"]) if row["preferences"] else {},
                }
            )

    def save_session(self, session: SessionContext) -> None:
        """
        Save or update session context.

        Args:
            session: SessionContext object to persist

        Thread-safe operation using RLock.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, created_at, last_activity, metadata,
                 turn_count, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.user_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    json.dumps(session.metadata),
                    session.turn_count,
                    1 if session.is_active else 0,
                ),
            )

            conn.commit()
            logger.debug(f"Saved session: {session.session_id}")

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Retrieve session context by ID.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionContext object if found, None otherwise
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT session_id, user_id, created_at, last_activity,
                       metadata, turn_count, is_active
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return SessionContext.from_dict(
                {
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "created_at": row["created_at"],
                    "last_activity": row["last_activity"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "turn_count": row["turn_count"],
                    "is_active": bool(row["is_active"]),
                }
            )

    def get_user_sessions(
        self,
        user_id: str,
        active_only: bool = False,
        limit: int = 10,
    ) -> List[SessionContext]:
        """
        Retrieve sessions for a given user.

        Args:
            user_id: User identifier
            active_only: If True, return only active sessions
            limit: Maximum number of sessions to return

        Returns:
            List of SessionContext objects, ordered by last activity
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = """
                SELECT session_id, user_id, created_at, last_activity,
                       metadata, turn_count, is_active
                FROM sessions
                WHERE user_id = ?
            """
            params: List[Any] = [user_id]

            if active_only:
                query += " AND is_active = 1"

            query += " ORDER BY last_activity DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            sessions = []
            for row in rows:
                sessions.append(
                    SessionContext.from_dict(
                        {
                            "session_id": row["session_id"],
                            "user_id": row["user_id"],
                            "created_at": row["created_at"],
                            "last_activity": row["last_activity"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                            "turn_count": row["turn_count"],
                            "is_active": bool(row["is_active"]),
                        }
                    )
                )

            return sessions

    def cleanup_old_sessions(self) -> int:
        """
        Remove sessions older than max_session_age_days.

        Returns:
            Number of sessions cleaned up

        Automatically called periodically, but can be invoked manually.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cutoff_date = datetime.utcnow() - timedelta(days=self.max_session_age_days)
            cutoff_str = cutoff_date.isoformat()

            cursor.execute(
                """
                DELETE FROM sessions
                WHERE last_activity < ? AND is_active = 0
                """,
                (cutoff_str,),
            )

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old sessions")

            return deleted_count

    def deactivate_session(self, session_id: str) -> bool:
        """
        Mark session as inactive.

        Args:
            session_id: Session identifier

        Returns:
            True if session was found and deactivated, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE sessions
                SET is_active = 0, last_activity = ?
                WHERE session_id = ?
                """,
                (datetime.utcnow().isoformat(), session_id),
            )

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.debug(f"Deactivated session: {session_id}")

            return updated

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.debug("Session store connection closed")
