"""Tests for Retriever agent (Agent 2)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from src.agents.retriever import retriever_agent, parse_lightrag_response, simple_retriever_fallback
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_retriever_agent_basic(agent_state_minimal):
    """Test basic retrieval functionality."""
    # Setup state with query
    agent_state_minimal["query"] = "Is climate change real?"
    agent_state_minimal["query_type"] = "text"

    # Mock simple_retriever
    mock_result = {
        "retrieved_docs": [
            {"text": "Climate change is real", "score": 0.9, "source": "source_1"}
        ],
        "retrieval_scores": [0.9],
        "retrieval_method": "mock"
    }

    with patch("src.agents.retriever.simple_retriever", return_value=mock_result):
        result = await retriever_agent(agent_state_minimal)

        # Verify results
        assert "retrieved_docs" in result
        assert "retrieval_scores" in result
        assert "retrieval_method" in result
        assert len(result["retrieved_docs"]) == 1
        assert result["retrieval_method"] == "mock"


@pytest.mark.asyncio
async def test_retriever_parse_lightrag_response():
    """Test LightRAG JSON response parsing."""
    # Mock LightRAG response
    mock_response = json.dumps({
        "results": [
            {"text": "Claim 1", "score": 0.9},
            {"text": "Claim 2", "score": 0.8}
        ]
    })

    docs = parse_lightrag_response(mock_response)

    # Verify parsing - function wraps dict in list
    assert isinstance(docs, list)
    assert len(docs) == 1  # The dict gets wrapped
    assert "results" in docs[0]
    assert docs[0]["results"][0]["text"] == "Claim 1"


@pytest.mark.asyncio
async def test_retriever_simple_fallback(agent_state_minimal):
    """Test simple retriever fallback mechanism."""
    agent_state_minimal["query"] = "climate change"

    # Mock the simple_retriever function
    mock_results = [
        {"text": "Climate change is real", "score": 0.9, "source": "source_1"}
    ]

    with patch("src.agents.retriever.simple_retriever", return_value={
        "retrieved_docs": mock_results,
        "retrieval_scores": [0.9],
        "retrieval_method": "keyword"
    }):
        result = await retriever_agent(agent_state_minimal)

        # Should use fallback
        assert result["retrieval_method"] == "keyword"
        assert len(result["retrieved_docs"]) > 0


@pytest.mark.asyncio
async def test_retriever_empty_results(agent_state_minimal):
    """Test handling of empty retrieval results."""
    agent_state_minimal["query"] = "obscure topic with no matches"

    # Mock empty results
    with patch("src.agents.retriever.simple_retriever", return_value={
        "retrieved_docs": [],
        "retrieval_scores": [],
        "retrieval_method": "keyword"
    }):
        result = await retriever_agent(agent_state_minimal)

        # Should return empty results
        assert result["retrieved_docs"] == []
        assert result["retrieval_scores"] == []


@pytest.mark.asyncio
async def test_retriever_malformed_response(agent_state_minimal):
    """Test handling of malformed LightRAG response."""
    agent_state_minimal["query"] = "test query"

    # Mock malformed JSON response
    malformed_json = "{invalid json"

    docs = parse_lightrag_response(malformed_json)

    # Should handle gracefully - wraps non-JSON string in dict
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["text"] == malformed_json


@pytest.mark.asyncio
async def test_retriever_with_query_type(agent_state_minimal):
    """Test retrieval respects query type."""
    agent_state_minimal["query"] = "test query"
    agent_state_minimal["query_type"] = "multimodal"

    with patch("src.agents.retriever.simple_retriever", return_value={
        "retrieved_docs": [{"text": "result", "score": 0.8}],
        "retrieval_scores": [0.8],
        "retrieval_method": "mock"
    }):
        result = await retriever_agent(agent_state_minimal)

        # Should still work regardless of query type
        assert "retrieved_docs" in result


@pytest.mark.asyncio
async def test_retriever_handles_top_k(agent_state_minimal):
    """Test retrieval respects top_k parameter."""
    agent_state_minimal["query"] = "test query"

    # Mock results with more than top_k
    mock_results = [
        {"text": f"Result {i}", "score": 0.9 - i * 0.1, "source": f"source_{i}"}
        for i in range(20)
    ]

    with patch("src.agents.retriever.simple_retriever", return_value={
        "retrieved_docs": mock_results[:10],
        "retrieval_scores": [0.9 - i * 0.1 for i in range(10)],
        "retrieval_method": "mock"
    }):
        result = await retriever_agent(agent_state_minimal)

        # Should limit results
        assert len(result["retrieved_docs"]) <= 10


@pytest.mark.asyncio
async def test_retriever_exception_handling(agent_state_minimal):
    """Test retriever handles exceptions gracefully."""
    agent_state_minimal["query"] = "test query"

    # Mock exception
    with patch("src.agents.retriever.simple_retriever", side_effect=Exception("Retrieval failed")):
        result = await retriever_agent(agent_state_minimal)

        # Should return safe defaults
        assert "retrieved_docs" in result
        assert "retrieval_method" in result


def test_parse_lightrag_response_empty():
    """Test parsing empty LightRAG response."""
    empty_response = json.dumps({"results": []})

    docs = parse_lightrag_response(empty_response)

    # Function wraps the dict in a list
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert "results" in docs[0]
    assert docs[0]["results"] == []


def test_parse_lightrag_response_missing_fields():
    """Test parsing response with missing fields."""
    incomplete_response = json.dumps({
        "results": [
            {"text": "Claim without score"}
        ]
    })

    docs = parse_lightrag_response(incomplete_response)

    # Should handle missing fields
    assert len(docs) == 1
    assert "text" in docs[0]
