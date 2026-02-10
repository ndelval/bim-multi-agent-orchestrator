"""
Session lifecycle management coordinator.

Provides high-level API for managing user sessions with automatic user
context tracking and session cleanup.
"""

import logging
from typing import Optional
from datetime import datetime

from .user_context import SessionContext, UserContext
from .session_store import SessionStore

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Coordinates session lifecycle and user context management.

    This class provides the main API for creating and managing sessions,
    integrating SessionStore and UserContext management.

    Attributes:
        store: SessionStore instance for persistence
        current_session: Currently active session (if any)
        current_user: Currently active user context (if any)
    """

    def __init__(
        self,
        db_path: str = ".praison/sessions.db",
        max_session_age_days: int = 30,
    ):
        """
        Initialize session manager.

        Args:
            db_path: Path to SQLite database file
            max_session_age_days: Maximum age for sessions before cleanup
        """
        self.store = SessionStore(
            db_path=db_path,
            max_session_age_days=max_session_age_days,
        )
        self.current_session: Optional[SessionContext] = None
        self.current_user: Optional[UserContext] = None

    def create_session(
        self,
        user_id: str = "default_user",
        metadata: Optional[dict] = None,
    ) -> SessionContext:
        """
        Create a new session for a user.

        Args:
            user_id: User identifier (defaults to "default_user")
            metadata: Optional session metadata

        Returns:
            Created SessionContext object

        Side effects:
            - Creates or updates user context
            - Increments user's session count
            - Sets current_session and current_user
            - Persists session and user to database
        """
        # Get or create user context
        user = self.store.get_user(user_id)
        if user is None:
            user = UserContext(user_id=user_id)
            logger.info(f"Created new user: {user_id}")
        else:
            user.update_last_seen()
            logger.info(f"Resuming user: {user_id}")

        # Increment session count
        user.increment_sessions()

        # Create session
        session = SessionContext(
            user_id=user_id,
            metadata=metadata or {},
        )

        # Persist to database
        self.store.save_user(user)
        self.store.save_session(session)

        # Update current context
        self.current_user = user
        self.current_session = session

        logger.info(
            f"Created session {session.session_id} for user {user_id} "
            f"(session #{user.total_sessions})"
        )

        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Retrieve an existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionContext object if found, None otherwise
        """
        return self.store.get_session(session_id)

    def resume_session(self, session_id: str) -> bool:
        """
        Resume an existing session as the current session.

        Args:
            session_id: Session identifier to resume

        Returns:
            True if session was found and resumed, False otherwise

        Side effects:
            - Updates session activity timestamp
            - Loads associated user context
            - Sets current_session and current_user
        """
        session = self.store.get_session(session_id)
        if session is None:
            logger.warning(f"Session not found: {session_id}")
            return False

        if not session.is_active:
            logger.warning(f"Attempted to resume inactive session: {session_id}")
            return False

        # Update activity
        session.update_activity()
        self.store.save_session(session)

        # Load user context
        user = self.store.get_user(session.user_id)
        if user:
            user.update_last_seen()
            self.store.save_user(user)

        # Set as current
        self.current_session = session
        self.current_user = user

        logger.info(f"Resumed session {session_id} for user {session.user_id}")
        return True

    def record_turn(self) -> None:
        """
        Record a conversation turn in the current session.

        Side effects:
            - Increments session turn count
            - Updates session activity timestamp
            - Persists session to database

        Raises:
            RuntimeError: If no active session exists
        """
        if self.current_session is None:
            raise RuntimeError("No active session to record turn")

        self.current_session.increment_turn()
        self.store.save_session(self.current_session)

        logger.debug(
            f"Recorded turn #{self.current_session.turn_count} "
            f"in session {self.current_session.session_id}"
        )

    def end_session(self) -> None:
        """
        End the current session.

        Side effects:
            - Marks session as inactive
            - Persists session to database
            - Clears current_session and current_user
        """
        if self.current_session is None:
            logger.warning("No active session to end")
            return

        session_id = self.current_session.session_id
        self.current_session.deactivate()
        self.store.save_session(self.current_session)

        logger.info(
            f"Ended session {session_id} "
            f"({self.current_session.turn_count} turns)"
        )

        self.current_session = None
        self.current_user = None

    def cleanup_old_sessions(self) -> int:
        """
        Remove old inactive sessions from the database.

        Returns:
            Number of sessions cleaned up

        This should be called periodically to prevent database growth.
        """
        return self.store.cleanup_old_sessions()

    def get_session_info(self) -> dict:
        """
        Get information about the current session.

        Returns:
            Dictionary with session metadata

        Useful for debugging and monitoring.
        """
        if self.current_session is None:
            return {
                "active": False,
                "session_id": None,
                "user_id": None,
            }

        return {
            "active": True,
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "created_at": self.current_session.created_at.isoformat(),
            "last_activity": self.current_session.last_activity.isoformat(),
            "turn_count": self.current_session.turn_count,
            "user_total_sessions": (
                self.current_user.total_sessions if self.current_user else 0
            ),
        }

    def close(self) -> None:
        """
        Close session manager and underlying database connection.

        Should be called when shutting down the application.
        """
        if self.current_session and self.current_session.is_active:
            self.end_session()
        self.store.close()
        logger.info("Session manager closed")
