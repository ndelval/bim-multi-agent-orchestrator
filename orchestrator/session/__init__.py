"""Session management for multi-user support."""

from .session_manager import SessionManager
from .user_context import UserContext, SessionContext
from .session_store import SessionStore

# Alias for backwards compatibility
Session = SessionContext

__all__ = ["SessionManager", "UserContext", "SessionContext", "Session", "SessionStore"]
