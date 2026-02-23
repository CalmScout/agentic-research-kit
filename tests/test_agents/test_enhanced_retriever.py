"""Tests for Enhanced Retriever agent (combines Query Analyzer + Retriever)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.enhanced_retriever import enhanced_retriever_agent
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_enhanced_retriever_text_query(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test enhanced retriever with text query."""
    # Setup state with query
    agent_state_minimal["query"] = "Is climate change caused by humans?"

    # Mock the model selector (used in EntityExtractorTool) and embedder
    with patch("src.agents.tools.rag_tools.entity_extractor.get_model_selector") as mock_get_model:
        with patch("src.agents.enhanced_retriever.embedder", mock_embedding_model):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            # Mock simple_retriever to return documents
            mock_docs = [
                {"text": "Climate change is caused by human activities", "score": 0.9, "source": "test1.pdf"},
                {"text": "Greenhouse gases contribute to global warming", "score": 0.8, "source": "test2.pdf"}
            ]

            with patch("src.agents.tools.rag_tools.simple_retriever.simple_retriever", AsyncMock(return_value={
                "retrieved_docs": mock_docs,
                "retrieval_scores": [0.9, 0.8],
                "retrieval_method": "keyword"
            })):
                # Execute agent
                result = await enhanced_retriever_agent(agent_state_minimal)

                # Verify results
                assert "query_type" in result
                assert result["query_type"] == "text"
                assert "entities" in result
                assert isinstance(result["entities"], list)
                assert "query_embedding" in result
                assert len(result["query_embedding"]) == 2048
                assert "retrieved_docs" in result
                assert len(result["retrieved_docs"]) > 0


@pytest.mark.asyncio
async def test_enhanced_retriever_multimodal_query(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test enhanced retriever with multimodal query (image)."""
    # Setup state with image
    agent_state_minimal["query"] = "What is shown in this image?"
    agent_state_minimal["query_image"] = "/path/to/image.jpg"

    # Mock the model selector (used in EntityExtractorTool) and embedder
    with patch("src.agents.tools.rag_tools.entity_extractor.get_model_selector") as mock_get_model:
        with patch("src.agents.enhanced_retriever.embedder", mock_embedding_model):
            mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

            with patch("src.agents.tools.rag_tools.simple_retriever.simple_retriever", AsyncMock(return_value={
                "retrieved_docs": [],
                "retrieval_scores": [],
                "retrieval_method": "keyword"
            })):
                # Execute agent
                result = await enhanced_retriever_agent(agent_state_minimal)

                # Verify query type is multimodal
                assert result["query_type"] == "multimodal"
                assert "query_embedding" in result


@pytest.mark.asyncio
async def test_enhanced_retriever_entity_extraction(agent_state_minimal):
    """Test entity extraction from query."""
    agent_state_minimal["query"] = "Is climate change caused by humans?"

    # Mock LLM to return entities
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=Mock(content="climate change, humans, global warming"))

    with patch("src.agents.tools.rag_tools.entity_extractor.get_model_selector") as mock_get_model:
        mock_get_model.return_value = Mock(get_local_llm=Mock(return_value=mock_llm))

        with patch("src.agents.enhanced_retriever.embedder", Mock()):
            with patch("src.agents.tools.rag_tools.simple_retriever.simple_retriever", AsyncMock(return_value={
                "retrieved_docs": [],
                "retrieval_scores": [],
                "retrieval_method": "keyword"
            })):
                # Execute agent
                result = await enhanced_retriever_agent(agent_state_minimal)

                # Verify entities were extracted (converted from JSON)
                assert "entities" in result
                assert isinstance(result["entities"], list)


@pytest.mark.asyncio
async def test_enhanced_retriever_handles_errors(agent_state_minimal, mock_embedding_model):
    """Test enhanced retriever handles errors gracefully."""
    agent_state_minimal["query"] = "Test query"

    with patch("src.agents.enhanced_retriever.embedder", mock_embedding_model):
        # Mock retrieval to fail
        with patch("src.agents.tools.rag_tools.simple_retriever.simple_retriever", AsyncMock(side_effect=Exception("Retrieval failed"))):
            # Execute agent
            result = await enhanced_retriever_agent(agent_state_minimal)

            # Should return safe defaults
            assert "retrieved_docs" in result
            assert "retrieval_method" in result
