"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client(async_test_client):
    """Get test client for FastAPI app."""
    return async_test_client


def test_root_endpoint(client):
    """Test GET / endpoint returns API information."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "architecture" in data
    assert "endpoints" in data
    assert data["name"] == "MultiModal Agentic RAG API"


def test_health_endpoint(client):
    """Test GET /health endpoint returns health status."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "architecture" in data
    assert "ingested_docs" in data
    assert data["status"] == "healthy"
    assert data["architecture"] == "2-agent LangGraph"


def test_health_endpoint_with_docs(client, temp_dir):
    """Test health endpoint with ingested documents."""
    # Create mock doc status file in rag_storage directory
    import json
    rag_storage_dir = temp_dir / "rag_storage"
    rag_storage_dir.mkdir()
    doc_status_file = rag_storage_dir / "kv_store_doc_status.json"
    mock_docs = {
        "doc1": {"status": "ingested"},
        "doc2": {"status": "ingested"},
        "doc3": {"status": "ingested"}
    }
    doc_status_file.write_text(json.dumps(mock_docs))

    # Change to temp_dir to pick up the mock file
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["ingested_docs"] == 3
    finally:
        os.chdir(original_cwd)


def test_query_endpoint_success(client):
    """Test POST /query with valid query."""
    # Mock the workflow
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "Is climate change real?",
            "response": "Yes, climate change is real.",
            "confidence": 0.9,
            "sources": [
                {
                    "text": "Climate change is real.",
                    "url": "http://example.com/1",
                    "score": 0.9
                }
            ],
            "entities": ["climate change"],
            "retrieved_count": 1
        }

        response = client.post(
            "/query",
            json={"query": "Is climate change real?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Is climate change real?"
        assert "response" in data
        assert "confidence" in data
        assert "sources" in data
        assert "entities" in data
        assert "retrieved_count" in data
        assert 0.0 <= data["confidence"] <= 1.0


def test_query_endpoint_validation_empty_query(client):
    """Test POST /query with empty query (should fail validation)."""
    response = client.post(
        "/query",
        json={"query": ""}
    )

    # Should fail validation (min_length=1)
    assert response.status_code == 422


def test_query_endpoint_validation_missing_query(client):
    """Test POST /query without query field (should fail validation)."""
    response = client.post(
        "/query",
        json={},
    )

    # Should fail validation
    assert response.status_code == 422


def test_query_endpoint_response_format(client):
    """Test that query endpoint returns correct response format."""
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "test",
            "response": "test response",
            "confidence": 0.8,
            "sources": [],
            "entities": [],
            "retrieved_count": 0
        }

        response = client.post(
            "/query",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all fields exist and have correct types
        assert isinstance(data["query"], str)
        assert isinstance(data["response"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["sources"], list)
        assert isinstance(data["entities"], list)
        assert isinstance(data["retrieved_count"], int)


def test_query_endpoint_with_session(client):
    """Test query endpoint with custom session ID."""
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "test",
            "response": "test response",
            "confidence": 0.8,
            "sources": [],
            "entities": [],
            "retrieved_count": 0
        }

        response = client.post(
            "/query",
            json={
                "query": "test",
                "session": "custom_session_123"
            }
        )

        assert response.status_code == 200


def test_query_endpoint_with_debug(client):
    """Test query endpoint with debug mode enabled."""
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "test",
            "response": "test response",
            "confidence": 0.8,
            "sources": [],
            "entities": [],
            "retrieved_count": 0
        }

        response = client.post(
            "/query",
            json={
                "query": "test",
                "debug": True
            }
        )

        assert response.status_code == 200


def test_query_endpoint_error_handling(client):
    """Test query endpoint handles errors gracefully."""
    # Mock query_with_agents to raise exception
    with patch("src.agents.workflow.query_with_agents", side_effect=Exception("Query failed")):
        response = client.post(
            "/query",
            json={"query": "test"}
        )

        # Should return 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


def test_query_endpoint_long_sources_truncated(client):
    """Test that long source text is truncated in response."""
    long_source_text = "A" * 1000  # Very long source text

    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "test",
            "response": "test response",
            "confidence": 0.8,
            "sources": [{"text": long_source_text, "url": "http://example.com"}],
            "entities": [],
            "retrieved_count": 1
        }

        response = client.post(
            "/query",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        # Source should be truncated to 500 chars
        assert len(data["sources"][0]["text"]) <= 500


def test_stats_endpoint(client):
    """Test GET /stats endpoint returns system statistics."""
    response = client.get("/stats")

    assert response.status_code == 200
    data = response.json()
    assert "ingested_docs" in data
    assert "architecture" in data
    assert "agents" in data
    assert isinstance(data["agents"], list)
    assert len(data["agents"]) == 2


def test_stats_endpoint_agents_list(client):
    """Test that stats endpoint lists all 2 agents."""
    response = client.get("/stats")

    assert response.status_code == 200
    data = response.json()

    expected_agents = [
        "Enhanced Retriever",
        "Enhanced Response Generator"
    ]

    for agent in expected_agents:
        assert agent in data["agents"]


def test_query_endpoint_sources_with_url_and_score(client):
    """Test query endpoint handles sources with URL and score."""
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "test",
            "response": "test response",
            "confidence": 0.8,
            "sources": [
                {
                    "text": "Test source",
                    "url": "http://example.com/test",
                    "score": 0.95
                }
            ],
            "entities": [],
            "retrieved_count": 1
        }

        response = client.post(
            "/query",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["url"] == "http://example.com/test"
        assert data["sources"][0]["score"] == 0.95


def test_query_endpoint_multiple_sources(client):
    """Test query endpoint with multiple sources."""
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": "test",
            "response": "test response",
            "confidence": 0.8,
            "sources": [
                {"text": f"Source {i}", "url": f"http://example.com/{i}", "score": 0.9 - i * 0.1}
                for i in range(5)
            ],
            "entities": [],
            "retrieved_count": 5
        }

        response = client.post(
            "/query",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 5
