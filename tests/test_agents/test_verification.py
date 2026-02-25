import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.agents.verification import verification_agent
from src.agents.base_state import BaseAgentState

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

    with patch("src.agents.model_selector.ModelSelector.get_llm_with_fallback") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_get_llm.return_value = mock_llm

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

    with patch("src.agents.model_selector.ModelSelector.get_llm_with_fallback") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_get_llm.return_value = mock_llm

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
        "sources": []
    }
    result = await verification_agent(state_no_sources)
    assert result["verification_status"] == "skipped"

    # No response
    state_no_response: BaseAgentState = {
        "query": "Test",
        "response": "",
        "sources": [{"text": "source"}]
    }
    result = await verification_agent(state_no_response)
    assert result["verification_status"] == "skipped"
