"""
Comprehensive tests for session management system.

Tests cover the complete lifecycle of session management including:
- User context creation and persistence
- Session creation and lifecycle
- Turn tracking and activity updates
- Session cleanup and resource management
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from orchestrator.session.user_context import SessionContext, UserContext
from orchestrator.session.session_store import SessionStore
from orchestrator.session.session_manager import SessionManager


class TestUserContext:
    """Test UserContext dataclass functionality."""

    def test_user_context_creation(self):
        """Test basic UserContext creation with defaults."""
        user = UserContext(user_id="test_user")

        assert user.user_id == "test_user"
        assert user.total_sessions == 0
        assert user.preferences == {}
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.last_seen, datetime)

    def test_user_context_increment_sessions(self):
        """Test session count increment and last_seen update."""
        user = UserContext(user_id="test_user")
        initial_last_seen = user.last_seen

        user.increment_sessions()

        assert user.total_sessions == 1
        assert user.last_seen >= initial_last_seen

    def test_user_context_serialization(self):
        """Test UserContext to_dict and from_dict roundtrip."""
        original = UserContext(
            user_id="test_user",
            total_sessions=5,
            preferences={"theme": "dark", "language": "en"}
        )

        serialized = original.to_dict()
        deserialized = UserContext.from_dict(serialized)

        assert deserialized.user_id == original.user_id
        assert deserialized.total_sessions == original.total_sessions
        assert deserialized.preferences == original.preferences


class TestSessionContext:
    """Test SessionContext dataclass functionality."""

    def test_session_context_creation(self):
        """Test basic SessionContext creation with defaults."""
        session = SessionContext(user_id="test_user")

        assert session.user_id == "test_user"
        assert session.turn_count == 0
        assert session.is_active is True
        assert session.metadata == {}
        assert isinstance(session.session_id, str)
        assert len(session.session_id) > 0

    def test_session_context_increment_turn(self):
        """Test turn increment and activity update."""
        session = SessionContext(user_id="test_user")
        initial_activity = session.last_activity

        session.increment_turn()

        assert session.turn_count == 1
        assert session.last_activity >= initial_activity

    def test_session_context_deactivate(self):
        """Test session deactivation."""
        session = SessionContext(user_id="test_user")

        session.deactivate()

        assert session.is_active is False

    def test_session_context_serialization(self):
        """Test SessionContext to_dict and from_dict roundtrip."""
        original = SessionContext(
            user_id="test_user",
            metadata={"source": "cli", "version": "1.0"},
            turn_count=3
        )

        serialized = original.to_dict()
        deserialized = SessionContext.from_dict(serialized)

        assert deserialized.session_id == original.session_id
        assert deserialized.user_id == original.user_id
        assert deserialized.turn_count == original.turn_count
        assert deserialized.metadata == original.metadata


class TestSessionStore:
    """Test SessionStore persistence layer."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_sessions.db")
            yield db_path

    @pytest.fixture
    def store(self, temp_db):
        """Create SessionStore instance with temporary database."""
        store = SessionStore(db_path=temp_db)
        yield store
        store.close()

    def test_store_initialization(self, temp_db):
        """Test SessionStore schema initialization."""
        store = SessionStore(db_path=temp_db)

        assert os.path.exists(temp_db)
        assert store._conn is not None

        store.close()

    def test_save_and_get_user(self, store):
        """Test user persistence roundtrip."""
        user = UserContext(
            user_id="test_user",
            total_sessions=5,
            preferences={"theme": "dark"}
        )

        store.save_user(user)
        retrieved = store.get_user("test_user")

        assert retrieved is not None
        assert retrieved.user_id == user.user_id
        assert retrieved.total_sessions == user.total_sessions
        assert retrieved.preferences == user.preferences

    def test_save_and_get_session(self, store):
        """Test session persistence roundtrip."""
        session = SessionContext(
            user_id="test_user",
            metadata={"source": "test"},
            turn_count=3
        )

        store.save_session(session)
        retrieved = store.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        assert retrieved.user_id == session.user_id
        assert retrieved.turn_count == session.turn_count
        assert retrieved.metadata == session.metadata

    def test_get_user_sessions(self, store):
        """Test retrieving sessions for a user."""
        user_id = "test_user"

        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = SessionContext(user_id=user_id, turn_count=i)
            store.save_session(session)
            sessions.append(session)

        retrieved = store.get_user_sessions(user_id, limit=10)

        assert len(retrieved) == 3
        # Should be ordered by last_activity DESC
        assert all(s.user_id == user_id for s in retrieved)

    def test_get_user_sessions_active_only(self, store):
        """Test filtering for active sessions only."""
        user_id = "test_user"

        # Create active and inactive sessions
        active_session = SessionContext(user_id=user_id)
        store.save_session(active_session)

        inactive_session = SessionContext(user_id=user_id)
        inactive_session.deactivate()
        store.save_session(inactive_session)

        retrieved = store.get_user_sessions(user_id, active_only=True)

        assert len(retrieved) == 1
        assert retrieved[0].session_id == active_session.session_id
        assert retrieved[0].is_active is True

    def test_deactivate_session(self, store):
        """Test session deactivation in store."""
        session = SessionContext(user_id="test_user")
        store.save_session(session)

        result = store.deactivate_session(session.session_id)

        assert result is True

        retrieved = store.get_session(session.session_id)
        assert retrieved.is_active is False

    def test_cleanup_old_sessions(self, store):
        """Test automatic cleanup of old sessions."""
        # Create old inactive session
        old_session = SessionContext(user_id="test_user")
        old_session.deactivate()
        # Manually set old last_activity
        old_session.last_activity = datetime.utcnow() - timedelta(days=35)
        store.save_session(old_session)

        # Create recent session
        recent_session = SessionContext(user_id="test_user")
        store.save_session(recent_session)

        # Cleanup (max_session_age_days defaults to 30)
        cleaned = store.cleanup_old_sessions()

        assert cleaned == 1

        # Old session should be gone
        assert store.get_session(old_session.session_id) is None
        # Recent session should remain
        assert store.get_session(recent_session.session_id) is not None

    def test_thread_safety(self, store):
        """Test basic thread safety with RLock."""
        import threading

        errors = []

        def create_sessions():
            try:
                for i in range(10):
                    session = SessionContext(user_id=f"user_{threading.current_thread().name}")
                    store.save_session(session)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSessionManager:
    """Test SessionManager high-level API."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_sessions.db")
            yield db_path

    @pytest.fixture
    def manager(self, temp_db):
        """Create SessionManager instance with temporary database."""
        manager = SessionManager(db_path=temp_db)
        yield manager
        manager.close()

    def test_create_session(self, manager):
        """Test session creation through manager."""
        session = manager.create_session(
            user_id="test_user",
            metadata={"source": "test"}
        )

        assert session is not None
        assert session.user_id == "test_user"
        assert session.metadata["source"] == "test"
        assert manager.current_session == session
        assert manager.current_user is not None
        assert manager.current_user.user_id == "test_user"
        assert manager.current_user.total_sessions == 1

    def test_create_multiple_sessions_same_user(self, manager):
        """Test creating multiple sessions for same user."""
        session1 = manager.create_session(user_id="test_user")
        manager.end_session()

        session2 = manager.create_session(user_id="test_user")

        assert session1.session_id != session2.session_id
        assert manager.current_user.total_sessions == 2

    def test_resume_session(self, manager):
        """Test resuming an existing session."""
        # Create and end session
        original_session = manager.create_session(user_id="test_user")
        session_id = original_session.session_id
        manager.end_session()

        # Reactivate the session
        original_session.is_active = True
        manager.store.save_session(original_session)

        # Resume
        result = manager.resume_session(session_id)

        assert result is True
        assert manager.current_session is not None
        assert manager.current_session.session_id == session_id

    def test_resume_nonexistent_session(self, manager):
        """Test attempting to resume non-existent session."""
        result = manager.resume_session("nonexistent_id")

        assert result is False
        assert manager.current_session is None

    def test_record_turn(self, manager):
        """Test recording conversation turns."""
        session = manager.create_session(user_id="test_user")

        manager.record_turn()

        assert session.turn_count == 1

        manager.record_turn()

        assert session.turn_count == 2

    def test_record_turn_without_session(self, manager):
        """Test error handling when recording turn without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            manager.record_turn()

    def test_end_session(self, manager):
        """Test ending a session."""
        session = manager.create_session(user_id="test_user")
        session_id = session.session_id

        manager.end_session()

        assert manager.current_session is None
        assert manager.current_user is None

        # Verify session is marked inactive in store
        stored_session = manager.store.get_session(session_id)
        assert stored_session.is_active is False

    def test_get_session_info(self, manager):
        """Test getting session information."""
        # No active session
        info = manager.get_session_info()
        assert info["active"] is False

        # With active session
        session = manager.create_session(
            user_id="test_user",
            metadata={"source": "test"}
        )
        manager.record_turn()

        info = manager.get_session_info()
        assert info["active"] is True
        assert info["session_id"] == session.session_id
        assert info["user_id"] == "test_user"
        assert info["turn_count"] == 1

    def test_cleanup_old_sessions(self, manager):
        """Test cleanup through manager."""
        # Create old session
        old_session = manager.create_session(user_id="old_user")
        old_session.last_activity = datetime.utcnow() - timedelta(days=35)
        old_session.deactivate()
        manager.store.save_session(old_session)

        cleaned = manager.cleanup_old_sessions()

        assert cleaned >= 1

    def test_session_lifecycle_integration(self, manager):
        """Test complete session lifecycle."""
        # Create session
        session = manager.create_session(
            user_id="test_user",
            metadata={"source": "cli", "version": "1.0"}
        )
        assert manager.current_session is not None

        # Record multiple turns
        for _ in range(5):
            manager.record_turn()

        assert session.turn_count == 5

        # Get session info
        info = manager.get_session_info()
        assert info["turn_count"] == 5
        assert info["user_total_sessions"] == 1

        # End session
        session_id = session.session_id
        manager.end_session()

        assert manager.current_session is None

        # Verify persistence
        stored_session = manager.store.get_session(session_id)
        assert stored_session.turn_count == 5
        assert stored_session.is_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
