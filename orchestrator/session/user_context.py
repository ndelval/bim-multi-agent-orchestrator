"""
User and session context dataclasses.

Defines the core data structures for session tracking and user management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


@dataclass
class SessionContext:
    """
    Represents a single conversation session.

    Attributes:
        session_id: Unique identifier for this session (UUID)
        user_id: Associated user identifier
        created_at: Timestamp when session was created
        last_activity: Timestamp of last interaction
        metadata: Additional session metadata (e.g., source, device)
        turn_count: Number of conversation turns in this session
        is_active: Whether the session is currently active
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    is_active: bool = True

    def update_activity(self) -> None:
        """Update last activity timestamp to current time."""
        self.last_activity = datetime.utcnow()

    def increment_turn(self) -> None:
        """Increment conversation turn count and update activity."""
        self.turn_count += 1
        self.update_activity()

    def deactivate(self) -> None:
        """Mark session as inactive."""
        self.is_active = False
        self.update_activity()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "turn_count": self.turn_count,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Create SessionContext from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            metadata=data.get("metadata", {}),
            turn_count=data.get("turn_count", 0),
            is_active=data.get("is_active", True),
        )


@dataclass
class UserContext:
    """
    Represents user-level context across multiple sessions.

    Attributes:
        user_id: Unique user identifier
        created_at: Timestamp when user was first seen
        last_seen: Timestamp of last user interaction
        total_sessions: Total number of sessions for this user
        preferences: User preferences and settings
    """

    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    total_sessions: int = 0
    preferences: Dict[str, Any] = field(default_factory=dict)

    def update_last_seen(self) -> None:
        """Update last seen timestamp to current time."""
        self.last_seen = datetime.utcnow()

    def increment_sessions(self) -> None:
        """Increment session count and update last seen."""
        self.total_sessions += 1
        self.update_last_seen()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "total_sessions": self.total_sessions,
            "preferences": self.preferences,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserContext":
        """Create UserContext from dictionary."""
        return cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            total_sessions=data.get("total_sessions", 0),
            preferences=data.get("preferences", {}),
        )
