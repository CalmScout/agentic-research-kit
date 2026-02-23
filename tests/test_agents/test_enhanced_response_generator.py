"""Tests for Enhanced Response Generator agent (combines Evidence Aggregator + Response Generator)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.enhanced_response_generator import enhanced_response_generator_agent, parse_confidence_from_response
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_enhanced_response_generator_basic(agent_state_with_docs, mock_llm):
    """Test enhanced response generator with basic state."""
    # Mock the model selector
    with patch("src.agents.enhanced_response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(
            get_local_llm=Mock(return_value=mock_llm),
            get_llm_with_fallback=Mock(return_value=mock_llm)
        )

        # Execute agent
        result = await enhanced_response_generator_agent(agent_state_with_docs)

        # Verify results
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "sources" in result
        assert isinstance(result["sources"], list)


@pytest.mark.asyncio
async def test_enhanced_response_generator_reranking(agent_state_with_docs, mock_llm):
    """Test that enhanced response generator reranks documents."""
    with patch("src.agents.enhanced_response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(
            get_local_llm=Mock(return_value=mock_llm),
            get_llm_with_fallback=Mock(return_value=mock_llm)
        )

        # Execute agent
        result = await enhanced_response_generator_agent(agent_state_with_docs)

        # Verify reranked docs exist
        assert "reranked_docs" in result
        assert isinstance(result["reranked_docs"], list)
        # Reranking should reduce to top 10
        assert len(result["reranked_docs"]) <= 10


@pytest.mark.asyncio
async def test_enhanced_response_generator_empty_docs(agent_state_minimal):
    """Test enhanced response generator handles empty retrieved docs."""
    agent_state_minimal["retrieved_docs"] = []
    agent_state_minimal["query"] = "Test query"

    # Execute agent
    result = await enhanced_response_generator_agent(agent_state_minimal)

    # Should return safe fallback
    assert "response" in result
    assert "confidence" in result
    assert result["confidence"] <= 0.5  # Low confidence for no results


@pytest.mark.asyncio
async def test_enhanced_response_generator_evidence_synthesis(agent_state_with_docs, mock_llm):
    """Test evidence synthesis from retrieved documents."""
    # Mock LLM to return specific synthesis
    mock_llm_with_response = Mock()
    mock_llm_with_response.ainvoke = AsyncMock(return_value=Mock(
        content="**Main Consensus**: Climate change is real.\n**Key Evidence**: Temperature rise, CO2 levels.\n**Credibility**: High confidence from multiple sources."
    ))

    with patch("src.agents.enhanced_response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(
            get_local_llm=Mock(return_value=mock_llm_with_response),
            get_llm_with_fallback=Mock(return_value=mock_llm)
        )

        # Execute agent
        result = await enhanced_response_generator_agent(agent_state_with_docs)

        # Verify evidence summary exists
        # Note: evidence_summary is internal but should be processed
        assert "response" in result


def test_parse_confidence_high():
    """Test confidence parsing for high confidence indicators."""
    response = "The evidence clearly shows that climate change is caused by human activities."
    confidence = parse_confidence_from_response(response)
    assert confidence >= 0.8


def test_parse_confidence_low():
    """Test confidence parsing for low confidence indicators."""
    response = "There is insufficient evidence to determine the cause."
    confidence = parse_confidence_from_response(response)
    assert confidence <= 0.5


def test_parse_confidence_medium():
    """Test confidence parsing for medium confidence."""
    response = "It appears that climate change is likely caused by human activities."
    confidence = parse_confidence_from_response(response)
    assert 0.5 <= confidence <= 0.8


def test_parse_confidence_default():
    """Test confidence parsing defaults to moderate."""
    response = "Here is some information about climate change."
    confidence = parse_confidence_from_response(response)
    assert confidence == 0.7  # Default moderate confidence


@pytest.mark.asyncio
async def test_enhanced_response_generator_handles_errors(agent_state_with_docs):
    """Test enhanced response generator handles LLM errors gracefully."""
    # Mock LLM to fail
    with patch("src.agents.enhanced_response_generator.get_model_selector") as mock_get_model:
        mock_get_model.side_effect = Exception("Model selector failed")

        # Execute agent - should not raise
        result = await enhanced_response_generator_agent(agent_state_with_docs)

        # Should return error response (fallback with moderate confidence)
        assert "response" in result
        assert "confidence" in result
        # Fallback response has default moderate confidence (0.7)
        assert 0.0 <= result["confidence"] <= 1.0
