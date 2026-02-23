"""Tests for Response Generator agent (Agent 4)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.response_generator import (
    response_generator_agent,
    format_sources_for_prompt,
    parse_confidence_from_response,
    format_response_for_display
)
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_response_generator_basic(agent_state_minimal, sample_claims):
    """Test basic response generation."""
    # Setup state with evidence
    agent_state_minimal["query"] = "Is climate change real?"
    agent_state_minimal["evidence_summary"] = "Evidence shows climate change is real."
    agent_state_minimal["top_claims"] = sample_claims[:2]

    # Mock LLM
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(
        return_value=Mock(content="Based on the evidence, climate change is definitely real.")
    )

    with patch("src.agents.response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(get_llm_with_fallback=Mock(return_value=mock_llm))

        result = await response_generator_agent(agent_state_minimal)

        # Verify results
        assert "response" in result
        assert "confidence" in result
        assert "sources" in result
        assert len(result["response"]) > 0
        assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_response_generator_format_sources(sample_claims):
    """Test source formatting for prompt."""
    formatted = format_sources_for_prompt(sample_claims)

    # Should format sources nicely
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "climate change" in formatted.lower()
    assert "Source:" in formatted


@pytest.mark.asyncio
async def test_response_generator_parse_confidence_high():
    """Test parsing high confidence response."""
    response = "Based on clear evidence, climate change is definitely real. This is conclusively established."

    confidence = parse_confidence_from_response(response)

    # Should detect high confidence
    assert confidence >= 0.8


@pytest.mark.asyncio
async def test_response_generator_parse_confidence_medium():
    """Test parsing medium confidence response."""
    response = "Evidence suggests climate change is likely real. It appears to be supported by data."

    confidence = parse_confidence_from_response(response)

    # Should detect medium-high confidence
    assert 0.6 <= confidence < 0.8


@pytest.mark.asyncio
async def test_response_generator_parse_confidence_low():
    """Test parsing low confidence response."""
    response = "Climate change might be real, but there is insufficient evidence to confirm."

    confidence = parse_confidence_from_response(response)

    # Should detect low confidence
    assert confidence < 0.5


@pytest.mark.asyncio
async def test_response_generator_llm_failure(agent_state_minimal, sample_claims):
    """Test fallback when LLM fails."""
    agent_state_minimal["query"] = "Is climate change real?"
    agent_state_minimal["evidence_summary"] = "Evidence shows climate change is real."
    agent_state_minimal["top_claims"] = sample_claims[:2]

    # Mock LLM that fails
    mock_llm_fail = Mock()
    mock_llm_fail.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))

    with patch("src.agents.response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(get_llm_with_fallback=Mock(return_value=mock_llm_fail))

        result = await response_generator_agent(agent_state_minimal)

        # Should return fallback response
        assert "response" in result
        assert "confidence" in result
        assert result["confidence"] == 0.5  # Fallback confidence


@pytest.mark.asyncio
async def test_response_generator_format_display(agent_state_minimal):
    """Test formatting response for display."""
    response_text = "Climate change is real."
    confidence = 0.85
    sources = [
        {"text": "Source 1", "url": "http://example.com/1", "score": 0.9}
    ]

    formatted = format_response_for_display(response_text, confidence, sources)

    # Should format nicely
    assert "Climate change is real." in formatted
    assert "High" in formatted  # Confidence level
    assert "85%" in formatted or "0.85" in formatted
    assert "Source 1" in formatted


def test_response_generator_format_confidence_levels():
    """Test different confidence levels are formatted correctly."""
    # High confidence
    formatted_high = format_response_for_display("Response", 0.9, [])
    assert "High" in formatted_high

    # Medium confidence
    formatted_med = format_response_for_display("Response", 0.6, [])
    assert "Medium" in formatted_med

    # Low confidence
    formatted_low = format_response_for_display("Response", 0.3, [])
    assert "Low" in formatted_low


@pytest.mark.asyncio
async def test_response_generator_with_empty_evidence(agent_state_minimal):
    """Test response generation with minimal evidence."""
    agent_state_minimal["query"] = "What is this?"
    agent_state_minimal["evidence_summary"] = "No evidence found."
    agent_state_minimal["top_claims"] = []

    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(
        return_value=Mock(content="I could not find relevant information.")
    )

    with patch("src.agents.response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(get_llm_with_fallback=Mock(return_value=mock_llm))

        result = await response_generator_agent(agent_state_minimal)

        # Should still generate response
        assert "response" in result
        assert len(result["response"]) > 0


@pytest.mark.asyncio
async def test_response_generator_sources_preserved(agent_state_minimal, sample_claims):
    """Test that sources are preserved in response."""
    agent_state_minimal["query"] = "test"
    agent_state_minimal["evidence_summary"] = "test summary"
    agent_state_minimal["top_claims"] = sample_claims[:3]

    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Test response."))

    with patch("src.agents.response_generator.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(get_llm_with_fallback=Mock(return_value=mock_llm))

        result = await response_generator_agent(agent_state_minimal)

        # Sources should match top_claims
        assert len(result["sources"]) == len(sample_claims[:3])
        assert result["sources"][0]["text"] == sample_claims[0]["text"]


def test_parse_confidence_keywords():
    """Test confidence parsing with various keywords."""
    # Test various high confidence indicators
    high_responses = [
        "This is definitely correct.",
        "Clearly shows evidence.",
        "Strongly supports the claim.",
        "Conclusively established.",
    ]
    for response in high_responses:
        conf = parse_confidence_from_response(response)
        assert conf >= 0.8, f"Failed for: {response}"

    # Test low confidence indicators
    low_responses = [
        "Might be true.",
        "Could be correct.",
        "Possibly accurate.",
        "Insufficient evidence.",
    ]
    for response in low_responses:
        conf = parse_confidence_from_response(response)
        assert conf < 0.7, f"Failed for: {response}"


def test_format_sources_empty():
    """Test formatting empty sources."""
    formatted = format_sources_for_prompt([])

    assert formatted == "No sources available."


def test_format_sources_long_text():
    """Test that long source text is truncated."""
    long_source = [{"text": "A" * 200, "score": 0.9, "source": "test"}]

    formatted = format_sources_for_prompt(long_source)

    # Should truncate to 150 chars
    assert len(formatted) < 200
