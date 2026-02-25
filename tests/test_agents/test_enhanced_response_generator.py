"""Tests for Enhanced Response Generator agent (combines Evidence Aggregator + Response Generator)."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

from src.agents.enhanced_response_generator import enhanced_response_generator_agent
from src.agents.base_state import BaseAgentState

@pytest.fixture
def agent_state_with_docs() -> BaseAgentState:
    return {
        "query": "What are the latest advancements in quantum computing?",
        "query_image": None,
        "retrieval_mode": "hybrid",
        "memory_context": None,
        "query_type": "text",
        "entities": ["quantum computing"],
        "query_embedding": [0.1] * 2048,
        "retrieved_docs": [
            {
                "text": "Quantum supremacy was achieved using superconducting qubits.",
                "score": 0.95,
                "metadata": {"source": "quantum_paper_1.pdf", "chunk_id": "1"}
            },
            {
                "text": "Error correction remains a significant challenge for scaling quantum computers.",
                "score": 0.88,
                "metadata": {"source": "quantum_review.pdf", "chunk_id": "2"}
            }
        ],
        "retrieval_scores": [0.95, 0.88],
        "retrieval_method": "hybrid",
        "reranked_docs": [],
        "evidence_summary": "",
        "top_results": [],
        "response": "",
        "sources": [],
        "messages": [],
        "verification_status": None,
        "verification_feedback": None
    }

@pytest.fixture
def agent_state_minimal() -> BaseAgentState:
    return {
        "query": "Test query",
        "query_image": None,
        "retrieval_mode": "hybrid",
        "memory_context": None,
        "query_type": "text",
        "entities": [],
        "query_embedding": [],
        "retrieved_docs": [],
        "retrieval_scores": [],
        "retrieval_method": "hybrid",
        "reranked_docs": [],
        "evidence_summary": "",
        "top_results": [],
        "response": "",
        "sources": [],
        "messages": [],
        "verification_status": None,
        "verification_feedback": None
    }


@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.ainvoke.return_value = Mock(content="This is a generated response.")
    return mock


@pytest.mark.asyncio
async def test_enhanced_response_generator_basic(agent_state_with_docs, mock_llm):
    """Test enhanced response generator with basic state."""
    
    mock_registry = MagicMock()
    mock_registry.close = AsyncMock()
    # Mock reranker to just return the same docs
    mock_registry.execute = AsyncMock(return_value=json.dumps({"reranked_docs": agent_state_with_docs["retrieved_docs"]}))
    
    with patch("src.agents.enhanced_response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(
            get_local_llm=Mock(return_value=mock_llm),
            get_llm_with_fallback=Mock(return_value=mock_llm)
        )
        
        with patch("src.agents.enhanced_response_generator.ToolRegistry", return_value=mock_registry):
            # Execute agent
            result = await enhanced_response_generator_agent(agent_state_with_docs)

            # Verify results
            assert "response" in result
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert "sources" in result
            assert isinstance(result["sources"], list)
            assert len(result["sources"]) > 0


@pytest.mark.asyncio
async def test_enhanced_response_generator_empty_docs(agent_state_minimal):
    """Test enhanced response generator handles empty retrieved docs."""
    # Execute agent
    result = await enhanced_response_generator_agent(agent_state_minimal)

    # Should return safe fallback
    assert "response" in result
    assert "couldn't find" in result["response"].lower()
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_enhanced_response_generator_handles_errors(agent_state_with_docs):
    """Test enhanced response generator handles LLM errors gracefully."""
    # Mock LLM to fail
    with patch("src.agents.enhanced_response_generator.get_model_selector") as mock_get_model:
        mock_get_model.side_effect = Exception("Model selector failed")

        # Execute agent - should not raise
        result = await enhanced_response_generator_agent(agent_state_with_docs)

        # Should return error response
        assert "response" in result
        assert "error" in result["response"].lower()

from unittest.mock import MagicMock

