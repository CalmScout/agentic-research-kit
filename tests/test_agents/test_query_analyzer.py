"""Tests for Query Analyzer agent (Agent 1)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.query_analyzer import query_analyzer_agent, extract_entities_simple
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_query_analyzer_text_query(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test query analyzer with text query."""
    # Setup state with query
    agent_state_minimal["query"] = "Is climate change caused by humans?"

    # Mock the model selector and embedder
    with patch("src.agents.query_analyzer.get_model_selector") as mock_get_model:
        with patch("src.agents.query_analyzer.embedder", mock_embedding_model):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Verify results
            assert "query_type" in result
            assert result["query_type"] == "text"
            assert "entities" in result
            assert isinstance(result["entities"], list)
            assert "query_embedding" in result
            assert len(result["query_embedding"]) == 2048


@pytest.mark.asyncio
async def test_query_analyzer_multimodal_query(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test query analyzer with multimodal query (image)."""
    # Setup state with image
    agent_state_minimal["query"] = "What is shown in this image?"
    agent_state_minimal["query_image"] = "/path/to/image.jpg"

    # Mock the model selector and embedder
    with patch("src.agents.query_analyzer.get_model_selector") as mock_get_model:
        with patch("src.agents.query_analyzer.embedder", mock_embedding_model):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Verify query type is multimodal
            assert result["query_type"] == "multimodal"
            assert "query_embedding" in result


@pytest.mark.asyncio
async def test_query_analyzer_entity_extraction(agent_state_minimal):
    """Test entity extraction from query."""
    agent_state_minimal["query"] = "Is climate change caused by humans?"

    # Mock LLM to return entities
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=Mock(content="climate change, humans, global warming"))

    with patch("src.agents.query_analyzer.get_model_selector") as mock_get_model:
        with patch("src.agents.query_analyzer.embedder"):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Verify entities extracted
            assert "entities" in result
            assert len(result["entities"]) == 3
            assert "climate change" in result["entities"]
            assert "humans" in result["entities"]


@pytest.mark.asyncio
async def test_query_analyzer_embedding_generation(agent_state_minimal, mock_embedding_model):
    """Test embedding generation."""
    agent_state_minimal["query"] = "Test query"

    with patch("src.agents.query_analyzer.get_model_selector") as mock_get_model:
        with patch("src.agents.query_analyzer.embedder", mock_embedding_model):
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="none"))
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Verify embedding dimension
            assert "query_embedding" in result
            assert len(result["query_embedding"]) == 2048
            assert all(isinstance(x, float) for x in result["query_embedding"])


@pytest.mark.asyncio
async def test_query_analyzer_llm_failure(agent_state_minimal, mock_embedding_model):
    """Test fallback when LLM fails."""
    agent_state_minimal["query"] = "Test query"

    # Mock LLM that raises exception
    mock_llm_fail = Mock()
    mock_llm_fail.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))

    with patch("src.agents.query_analyzer.get_model_selector") as mock_get_model:
        with patch("src.agents.query_analyzer.embedder", mock_embedding_model):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm_fail))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Should still return result with empty entities
            assert "entities" in result
            assert result["entities"] == []
            assert "query_type" in result


@pytest.mark.asyncio
async def test_query_analyzer_embedding_failure(agent_state_minimal):
    """Test fallback when embedding generation fails."""
    agent_state_minimal["query"] = "Test query"

    # Mock embedder that raises exception
    mock_embedder_fail = Mock()
    mock_embedder_fail.embed_text = Mock(side_effect=Exception("Embedding failed"))

    with patch("src.agents.query_analyzer.get_model_selector"):
        with patch("src.agents.query_analyzer.embedder", mock_embedder_fail):
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="test"))
            mock_get_model = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Should return zero embedding as fallback
            assert "query_embedding" in result
            assert result["query_embedding"] == [0.0] * 2048


def test_extract_entities_simple():
    """Test rule-based entity extraction fallback."""
    query = "Did Biden visit the White House to discuss inflation?"

    entities = extract_entities_simple(query)

    # Should extract capitalized words (excluding stopwords)
    assert isinstance(entities, list)
    assert "Biden" in entities
    # "The", "White House" might be excluded depending on stopword filtering


def test_extract_entities_simple_with_stopwords():
    """Test that stopwords are filtered out."""
    query = "What did The President say about This issue?"

    entities = extract_entities_simple(query)

    # Should filter out common stopwords
    assert "The" not in entities
    assert "This" not in entities
    # Should keep content words
    assert "President" in entities


def test_extract_entities_simple_no_entities():
    """Test with query containing no entities."""
    query = "what is happening today"

    entities = extract_entities_simple(query)

    # Should return empty list or minimal results
    assert isinstance(entities, list)


@pytest.mark.asyncio
async def test_query_analyzer_none_entities(agent_state_minimal):
    """Test handling of 'none' response from LLM."""
    agent_state_minimal["query"] = "Test query"

    # Mock LLM to return "none"
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=Mock(content="none"))

    with patch("src.agents.query_analyzer.get_model_selector") as mock_get_model:
        with patch("src.agents.query_analyzer.embedder"):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Execute agent
            result = await query_analyzer_agent(agent_state_minimal)

            # Verify entities is empty list
            assert "entities" in result
            assert result["entities"] == []
