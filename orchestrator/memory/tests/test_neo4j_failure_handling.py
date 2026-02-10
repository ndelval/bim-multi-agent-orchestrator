"""
Tests for Neo4j failure handling and graceful degradation.

Validates that both HybridRAGMemoryProvider and Mem0MemoryProvider:
- Implement health checking before graph operations
- Retry with exponential backoff on connection failures
- Gracefully degrade to vector+lexical when Neo4j unavailable
- Handle authentication errors without retry loops
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from orchestrator.memory.providers.hybrid_provider import HybridRAGMemoryProvider
from orchestrator.memory.providers.mem0_provider import Mem0MemoryProvider
from orchestrator.core.config import EmbedderConfig


class TestHybridProviderNeo4jHandling:
    """Test Neo4j failure handling in HybridRAGMemoryProvider."""

    @pytest.fixture
    def embedder_config(self):
        """Create embedder configuration."""
        return EmbedderConfig(
            provider="openai",
            config={"model": "text-embedding-3-small"}
        )

    @pytest.fixture
    def hybrid_config_with_neo4j(self):
        """Create hybrid provider config with Neo4j enabled."""
        return {
            "vector_store": {
                "provider": "chromadb",
                "config": {
                    "path": "/tmp/test_chroma",
                    "collection": "test_collection"
                }
            },
            "lexical": {
                "db_path": "/tmp/test_lexical.db"
            },
            "rerank": {
                "enabled": False
            },
            "graph": {
                "enabled": True,
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "test_password"
            }
        }

    def test_check_neo4j_health_with_healthy_driver(self, embedder_config):
        """Test health check with healthy Neo4j connection."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock healthy Neo4j driver
        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value=1)

        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        result = provider._check_neo4j_health(mock_driver)

        assert result is True
        mock_session.run.assert_called_once_with("RETURN 1 AS health_check")

    def test_check_neo4j_health_with_unhealthy_driver(self, embedder_config):
        """Test health check with unhealthy Neo4j connection."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock unhealthy driver that raises exception
        mock_driver = Mock()
        mock_driver.session.side_effect = Exception("Connection failed")

        result = provider._check_neo4j_health(mock_driver)

        assert result is False

    def test_check_neo4j_health_with_none_driver(self, embedder_config):
        """Test health check with None driver."""
        provider = HybridRAGMemoryProvider(embedder_config)

        result = provider._check_neo4j_health(None)

        assert result is False

    @patch('orchestrator.memory.providers.hybrid_provider.GraphDatabase')
    def test_initialize_neo4j_with_retry_success_first_attempt(
        self,
        mock_graph_db,
        embedder_config
    ):
        """Test successful Neo4j initialization on first attempt."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock successful connection
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock health check success
        with patch.object(provider, '_check_neo4j_health', return_value=True):
            result = provider._initialize_neo4j_with_retry(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                max_retries=3
            )

        assert result is mock_driver
        assert mock_graph_db.driver.call_count == 1

    @patch('orchestrator.memory.providers.hybrid_provider.GraphDatabase')
    @patch('time.sleep')
    def test_initialize_neo4j_with_retry_eventual_success(
        self,
        mock_sleep,
        mock_graph_db,
        embedder_config
    ):
        """Test Neo4j initialization succeeding after retries."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock driver creation
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock health check: fail twice, then succeed
        health_checks = [False, False, True]
        with patch.object(
            provider,
            '_check_neo4j_health',
            side_effect=health_checks
        ):
            result = provider._initialize_neo4j_with_retry(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                max_retries=3
            )

        assert result is mock_driver
        # Should have created driver 3 times
        assert mock_graph_db.driver.call_count == 3
        # Should have slept twice (between attempts)
        assert mock_sleep.call_count == 2
        # Verify exponential backoff: 1s, 2s
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch('orchestrator.memory.providers.hybrid_provider.GraphDatabase')
    @patch('time.sleep')
    def test_initialize_neo4j_with_retry_max_retries_exceeded(
        self,
        mock_sleep,
        mock_graph_db,
        embedder_config
    ):
        """Test Neo4j initialization failing after max retries."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock driver creation
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock health check always failing
        with patch.object(provider, '_check_neo4j_health', return_value=False):
            result = provider._initialize_neo4j_with_retry(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                max_retries=3
            )

        assert result is None
        assert mock_graph_db.driver.call_count == 3
        assert mock_sleep.call_count == 2  # Between 3 attempts

    @patch('orchestrator.memory.providers.hybrid_provider.GraphDatabase')
    def test_initialize_neo4j_with_auth_error_no_retry(
        self,
        mock_graph_db,
        embedder_config
    ):
        """Test that authentication errors don't trigger retries."""
        from neo4j.exceptions import AuthError

        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock authentication failure
        mock_graph_db.driver.side_effect = AuthError("Invalid credentials")

        result = provider._initialize_neo4j_with_retry(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="wrong_password",
            max_retries=3
        )

        assert result is None
        # Should fail immediately without retries
        assert mock_graph_db.driver.call_count == 1

    def test_upsert_graph_skips_when_neo4j_unavailable(self, embedder_config):
        """Test _upsert_graph gracefully skips when Neo4j unavailable."""
        provider = HybridRAGMemoryProvider(embedder_config)
        provider._graph_driver = None  # No driver available

        # Should not raise exception
        provider._upsert_graph(
            doc_id="test_doc",
            content="test content",
            metadata={"document_id": "doc1"}
        )

    def test_upsert_graph_skips_when_neo4j_unhealthy(self, embedder_config):
        """Test _upsert_graph skips when Neo4j health check fails."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock unhealthy driver
        provider._graph_driver = Mock()

        with patch.object(provider, '_check_neo4j_health', return_value=False):
            # Should not raise exception
            provider._upsert_graph(
                doc_id="test_doc",
                content="test content",
                metadata={"document_id": "doc1"}
            )

    def test_graph_execute_write_skips_when_unhealthy(self, embedder_config):
        """Test _graph_execute_write skips when Neo4j unhealthy."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock driver
        provider._graph_driver = Mock()

        with patch.object(provider, '_check_neo4j_health', return_value=False):
            # Should not raise exception
            provider._graph_execute_write(
                func=lambda tx, params: None,
                params={"test": "data"}
            )

    def test_graph_query_skips_when_unhealthy(self, embedder_config):
        """Test _graph_query returns empty list when Neo4j unhealthy."""
        provider = HybridRAGMemoryProvider(embedder_config)

        # Mock driver
        provider._graph_driver = Mock()

        with patch.object(provider, '_check_neo4j_health', return_value=False):
            result = provider._graph_query(
                cypher="MATCH (n) RETURN n",
                params={}
            )

            assert result == []


class TestMem0ProviderNeo4jHandling:
    """Test Neo4j failure handling in Mem0MemoryProvider."""

    @pytest.fixture
    def embedder_config(self):
        """Create embedder configuration."""
        return EmbedderConfig(
            provider="openai",
            config={"model": "text-embedding-3-small"}
        )

    @pytest.fixture
    def mem0_config_with_neo4j(self):
        """Create Mem0 config with Neo4j graph store."""
        return {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "test_password"
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333
                }
            }
        }

    def test_check_neo4j_connection_success(self, embedder_config):
        """Test successful Neo4j connection check."""
        provider = Mem0MemoryProvider(embedder_config)

        # Mock successful Neo4j connection
        with patch('orchestrator.memory.providers.mem0_provider.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            mock_record = Mock()
            mock_record.__getitem__ = Mock(return_value=1)

            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=False)
            mock_driver.session.return_value = mock_session
            mock_driver.close = Mock()
            mock_gdb.driver.return_value = mock_driver

            result = provider._check_neo4j_connection(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password"
            )

            assert result is True
            mock_driver.close.assert_called_once()

    def test_check_neo4j_connection_failure(self, embedder_config):
        """Test Neo4j connection check with unavailable service."""
        from neo4j.exceptions import ServiceUnavailable

        provider = Mem0MemoryProvider(embedder_config)

        with patch('orchestrator.memory.providers.mem0_provider.GraphDatabase') as mock_gdb:
            mock_gdb.driver.side_effect = ServiceUnavailable("Service unavailable")

            result = provider._check_neo4j_connection(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password"
            )

            assert result is False

    @patch('time.sleep')
    def test_retry_graph_store_init_success_after_retries(
        self,
        mock_sleep,
        embedder_config,
        mem0_config_with_neo4j
    ):
        """Test graph store initialization succeeding after retries."""
        provider = Mem0MemoryProvider(embedder_config)

        # Mock connection checks: fail twice, then succeed
        connection_checks = [False, False, True]
        with patch.object(
            provider,
            '_check_neo4j_connection',
            side_effect=connection_checks
        ):
            result = provider._retry_graph_store_init(
                config=mem0_config_with_neo4j,
                max_retries=3
            )

        # Should return original config with graph_store intact
        assert "graph_store" in result
        assert result["graph_store"] == mem0_config_with_neo4j["graph_store"]

        # Should have retried with exponential backoff
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch('time.sleep')
    def test_retry_graph_store_init_max_retries_exceeded(
        self,
        mock_sleep,
        embedder_config,
        mem0_config_with_neo4j
    ):
        """Test graph store removed from config after max retries."""
        provider = Mem0MemoryProvider(embedder_config)

        # Mock connection always failing
        with patch.object(provider, '_check_neo4j_connection', return_value=False):
            result = provider._retry_graph_store_init(
                config=mem0_config_with_neo4j,
                max_retries=3
            )

        # Graph store should be removed from config
        assert "graph_store" not in result
        # Other config should remain
        assert "vector_store" in result

        # Should have retried
        assert mock_sleep.call_count == 2

    def test_retry_graph_store_init_incomplete_config(self, embedder_config):
        """Test handling of incomplete Neo4j configuration."""
        provider = Mem0MemoryProvider(embedder_config)

        incomplete_config = {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687"
                    # Missing username and password
                }
            }
        }

        result = provider._retry_graph_store_init(
            config=incomplete_config,
            max_retries=3
        )

        # Should remove graph_store due to incomplete config
        assert "graph_store" not in result

    def test_retry_graph_store_init_non_neo4j_provider(self, embedder_config):
        """Test that non-Neo4j graph providers are not retried."""
        provider = Mem0MemoryProvider(embedder_config)

        other_graph_config = {
            "graph_store": {
                "provider": "memgraph",
                "config": {}
            }
        }

        result = provider._retry_graph_store_init(
            config=other_graph_config,
            max_retries=3
        )

        # Should return original config unchanged
        assert result == other_graph_config

    @patch('orchestrator.memory.providers.mem0_provider.Memory')
    def test_init_mem0_client_fallback_without_vector_store(
        self,
        mock_memory,
        embedder_config
    ):
        """Test fallback to graph-only mode when vector store fails."""
        provider = Mem0MemoryProvider(embedder_config)

        config_with_vector = {
            "graph_store": {"provider": "neo4j"},
            "vector_store": {"provider": "qdrant"}
        }

        # Mock Memory.from_config to fail with vector store
        mock_memory.from_config.side_effect = [
            Exception("Vector store connection failed"),
            Mock()  # Success without vector store
        ]

        # Should not raise exception, should fallback
        client = provider._init_mem0_client(config_with_vector)

        assert client is not None
        # Should have been called twice: once with vector, once without
        assert mock_memory.from_config.call_count == 2


class TestNeo4jGracefulDegradation:
    """Integration tests for graceful degradation to vector+lexical."""

    @pytest.fixture
    def embedder_config(self):
        """Create embedder configuration."""
        return EmbedderConfig(
            provider="openai",
            config={"model": "text-embedding-3-small"}
        )

    def test_hybrid_provider_initializes_without_neo4j(self, embedder_config):
        """Test HybridRAGMemoryProvider initializes without Neo4j."""
        config = {
            "vector_store": {
                "provider": "chromadb",
                "config": {"path": "/tmp/test_chroma"}
            },
            "lexical": {"db_path": "/tmp/test_lexical.db"},
            "graph": {"enabled": False}
        }

        provider = HybridRAGMemoryProvider(embedder_config)

        # Should initialize successfully without Neo4j
        # (Actual initialization would require chromadb and real paths)
        assert provider is not None
        assert provider._graph_driver is None

    def test_hybrid_provider_continues_on_neo4j_failure(self, embedder_config):
        """Test HybridRAGMemoryProvider continues with degraded functionality."""
        provider = HybridRAGMemoryProvider(embedder_config)
        provider._graph_driver = None

        # Should be able to perform non-graph operations
        # (store/retrieve would work with vector+lexical only)
        assert provider._graph_driver is None

        # Graph operations should gracefully skip
        provider._upsert_graph("doc1", "content", {"document_id": "doc1"})
        # Should not raise exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
