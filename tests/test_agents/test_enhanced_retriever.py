"""Tests for Enhanced Retriever agent (combines Query Analyzer + Retriever)."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

from src.agents.enhanced_retriever import enhanced_retriever_agent
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_enhanced_retriever_text_query(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test enhanced retriever with text query."""
    # Setup state with query
    agent_state_minimal["query"] = "Is climate change caused by humans?"

    # Mock ToolRegistry.execute
    with patch("src.agents.enhanced_retriever.ToolRegistry.execute") as mock_execute:
        # Entity extractor returns JSON list of entities
        # Hybrid retriever returns JSON with documents
        def side_effect(name, params):
            if name == "entity_extractor":
                return json.dumps(["climate change", "humans"])
            elif name == "hybrid_retriever":
                return json.dumps({
                    "retrieved_docs": [
                        {"text": "Climate change is caused by human activities", "score": 0.9, "source": "test1.pdf"},
                        {"text": "Greenhouse gases contribute to global warming", "score": 0.8, "source": "test2.pdf"}
                    ],
                    "retrieval_scores": [0.9, 0.8],
                    "retrieval_method": "hybrid"
                })
            return "[]"

        mock_execute.side_effect = side_effect

        with patch("src.agents.enhanced_retriever.embedder", mock_embedding_model):
            # Execute agent
            result = await enhanced_retriever_agent(agent_state_minimal)

            # Verify results
            assert "query_type" in result
            assert result["query_type"] == "text"
            assert "entities" in result
            assert result["entities"] == ["climate change", "humans"]
            assert "query_embedding" in result
            assert len(result["query_embedding"]) == 2048
            assert "retrieved_docs" in result
            assert len(result["retrieved_docs"]) == 2


@pytest.mark.asyncio
async def test_enhanced_retriever_multimodal_query(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test enhanced retriever with multimodal query (image)."""
    # Setup state with image
    agent_state_minimal["query"] = "What is shown in this image?"
    agent_state_minimal["query_image"] = "/path/to/image.jpg"

    # Mock ToolRegistry.execute
    with patch("src.agents.enhanced_retriever.ToolRegistry.execute") as mock_execute:
        def side_effect(name, params):
            if name == "entity_extractor":
                return json.dumps([])
            return json.dumps({
                "retrieved_docs": [],
                "retrieval_scores": [],
                "retrieval_method": "hybrid"
            })
        
        mock_execute.side_effect = side_effect

        with patch("src.agents.enhanced_retriever.embedder", mock_embedding_model):
            # Execute agent
            result = await enhanced_retriever_agent(agent_state_minimal)

            # Verify query type is multimodal
            assert result["query_type"] == "multimodal"
            assert "query_embedding" in result


@pytest.mark.asyncio
async def test_enhanced_retriever_entity_extraction(agent_state_minimal):
    """Test entity extraction from query."""
    agent_state_minimal["query"] = "Is climate change caused by humans?"

    # Mock ToolRegistry.execute
    with patch("src.agents.enhanced_retriever.ToolRegistry.execute") as mock_execute:
        def side_effect(name, params):
            if name == "entity_extractor":
                return json.dumps(["climate change", "humans"])
            return json.dumps({
                "retrieved_docs": [],
                "retrieval_scores": [],
                "retrieval_method": "hybrid"
            })
        
        mock_execute.side_effect = side_effect

        with patch("src.agents.enhanced_retriever.embedder", Mock()):
            # Execute agent
            result = await enhanced_retriever_agent(agent_state_minimal)

            # Verify entities were extracted
            assert "entities" in result
            assert result["entities"] == ["climate change", "humans"]


@pytest.mark.asyncio
async def test_enhanced_retriever_handles_errors(agent_state_minimal, mock_embedding_model):
    """Test enhanced retriever handles errors gracefully."""
    agent_state_minimal["query"] = "Test query"

    with patch("src.agents.enhanced_retriever.embedder", mock_embedding_model):
        # Mock ToolRegistry.execute to fail
        with patch("src.agents.enhanced_retriever.ToolRegistry.execute", side_effect=Exception("Tool execution failed")):
            # Execute agent
            result = await enhanced_retriever_agent(agent_state_minimal)

            # Should return safe defaults
            assert "retrieved_docs" in result
            assert "retrieval_method" in result
