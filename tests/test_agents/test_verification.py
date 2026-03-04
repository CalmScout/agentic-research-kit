from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base_state import BaseAgentState
from src.agents.verification import verification_agent


@pytest.mark.asyncio
async def test_verification_agent_verified():
    """Test verification agent when response is fully supported."""
    state: BaseAgentState = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "sources": [{"text": "France's capital city is Paris.", "metadata": {"source": "test"}}]
    }

    mock_llm_response = MagicMock()
    mock_llm_response.content = '{"is_verified": true, "feedback": "All claims supported.", "corrected_response": "The capital of France is Paris."}'

    # Patch get_model_selector to return our mock
    with patch("src.agents.verification.get_model_selector") as mock_get_selector:
        mock_selector = MagicMock()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_selector.get_llm_with_fallback.return_value = mock_llm
        mock_get_selector.return_value = mock_selector

        result = await verification_agent(state)

        assert result["verification_status"] == "verified"
        assert result["response"] == "The capital of France is Paris."
        assert "All claims supported" in result["verification_feedback"]

@pytest.mark.asyncio
async def test_verification_agent_corrected():
    """Test verification agent when response contains a hallucination."""
    state: BaseAgentState = {
        "query": "What is the capital of France and its population?",
        "response": "The capital of France is Paris and it has 50 million people.",
        "sources": [{"text": "France's capital city is Paris.", "metadata": {"source": "test"}}]
    }

    # The LLM identifies the population claim as unsupported
    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        '{"is_verified": false, '
        '"feedback": "Population of 50 million is not mentioned in sources.", '
        '"corrected_response": "The capital of France is Paris."}'
    )

    # Patch get_model_selector to return our mock
    with patch("src.agents.verification.get_model_selector") as mock_get_selector:
        mock_selector = MagicMock()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_selector.get_llm_with_fallback.return_value = mock_llm
        mock_get_selector.return_value = mock_selector

        result = await verification_agent(state)

        assert result["verification_status"] == "corrected"
        assert result["response"] == "The capital of France is Paris."
        assert "50 million" in result["verification_feedback"]

@pytest.mark.asyncio
async def test_verification_agent_skipped():
    """Test verification agent when no sources or response are present."""
    # No sources
    state_no_sources: BaseAgentState = {
        "query": "Test",
        "response": "Test response",
        "sources": [],
        "iteration_count": 0
    }
    result = await verification_agent(state_no_sources)
    assert result["verification_status"] == "refine"
    assert "Initial retrieval yielded no results" in result["verification_feedback"]
    # No response
    state_no_response: BaseAgentState = {
        "query": "Test",
        "response": "",
        "sources": [{"text": "source"}],
        "iteration_count": 0
    }
    result = await verification_agent(state_no_response)
    assert result["verification_status"] == "refine"
