"""Tests for multi-agent workflow orchestration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import os
import json
from pathlib import Path

from src.agents.workflow import (
    create_multi_agent_workflow,
    query_with_agents,
    query_with_agents_sync,
    _initialize_phoenix
)
from src.agents.base_state import BaseAgentState


@pytest.mark.asyncio
async def test_workflow_creation():
    """Test LangGraph workflow compilation."""
    workflow = create_multi_agent_workflow()

    # Verify workflow is created
    assert workflow is not None
    assert hasattr(workflow, "ainvoke")
    assert hasattr(workflow, "stream")


@pytest.mark.asyncio
async def test_workflow_end_to_end(agent_state_minimal, mock_llm, mock_embedding_model):
    """Test full workflow with mocked agents."""
    query = "Is climate change real?"

    # Mock the agent functions directly in the workflow module
    with patch("src.agents.workflow.enhanced_retriever_agent") as mock_retriever:
        with patch("src.agents.workflow.enhanced_response_generator_agent") as mock_generator:
            with patch("src.agents.workflow.verification_agent") as mock_verifier:
                
                # Setup mock returns
                mock_retriever.return_value = {
                    "query_type": "text",
                    "entities": ["climate change"],
                    "retrieved_docs": [{"text": "test", "score": 0.9}],
                    "retrieval_scores": [0.9],
                    "retrieval_method": "mock"
                }
                
                mock_generator.return_value = {
                    "response": "Final response",
                    "sources": [{"text": "test", "score": 0.9}],
                    "top_results": [{"text": "test", "score": 0.9}],
                    "confidence": 0.9
                }
                
                mock_verifier.return_value = {
                    "response": "Final response",
                    "verification_status": "verified",
                    "verification_feedback": "All good"
                }

                result = await query_with_agents(query)

                # Verify result structure
                assert "query" in result
                assert "response" in result
                assert "sources" in result
                assert "retrieved_count" in result
                assert result["response"] == "Final response"
                assert mock_retriever.called
                assert mock_generator.called
                assert mock_verifier.called


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow handles errors gracefully."""
    query = "test query"

    # Mock workflow to raise exception
    with patch("src.agents.workflow.create_multi_agent_workflow", side_effect=Exception("Workflow failed")):
        result = await query_with_agents(query)

        # Should return error response
        assert "error" in result or "response" in result
        assert result.get("confidence") is None


@pytest.mark.asyncio
async def test_workflow_phoenix_disabled():
    """Test workflow works without Phoenix observability."""
    # Ensure Phoenix is disabled
    os.environ["PHOENIX_ENABLED"] = "false"

    # Re-initialize to test without Phoenix
    import importlib
    import src.agents.workflow
    importlib.reload(src.agents.workflow)

    workflow = create_multi_agent_workflow()

    # Should still create workflow
    assert workflow is not None


def test_query_with_agents_sync():
    """Test synchronous wrapper for async function."""
    query = "test query"

    # Mock the async function
    with patch("src.agents.workflow.query_with_agents") as mock_query:
        mock_query.return_value = {
            "query": query,
            "response": "test response",
            "confidence": 0.8,
            "sources": [],
            "retrieved_count": 0
        }

        result = query_with_agents_sync(query)

        # Verify result
        assert result["query"] == query
        assert result["response"] == "test response"


@pytest.mark.asyncio
async def test_workflow_state_propagation():
    """Test that state propagates correctly through agents."""
    query = "test query"

    with patch("src.agents.workflow.enhanced_retriever_agent") as mock_retriever:
        with patch("src.agents.workflow.enhanced_response_generator_agent") as mock_generator:
            with patch("src.agents.workflow.verification_agent") as mock_verifier:
                
                mock_retriever.return_value = {"entities": ["test"], "retrieved_docs": [{"text": "doc1"}]}
                mock_generator.return_value = {"response": "resp", "sources": []}
                mock_verifier.return_value = {"verification_status": "verified"}

                result = await query_with_agents(query)

                # Verify state is propagated and added to result
                assert "query" in result
                assert "entities" in result
                assert result["entities"] == ["test"]
                assert result["retrieved_count"] == 1


@pytest.mark.asyncio
async def test_workflow_with_debug_mode():
    """Test workflow with debug logging enabled."""
    query = "test query"

    with patch("src.agents.workflow.create_multi_agent_workflow") as mock_workflow:
        mock_workflow.return_value = Mock()
        mock_workflow.return_value.ainvoke = AsyncMock(return_value={
            "query": query,
            "response": "test",
            "confidence": 0.8,
            "sources": [],
            "entities": [],
            "retrieved_docs": []
        })

        # Test with debug=True
        result = await query_with_agents(query, debug=True)

        # Should complete successfully
        assert result["query"] == query


@pytest.mark.asyncio
async def test_workflow_with_multimodal_query():
    """Test workflow with image query."""
    query = "What is this?"
    image_path = "/path/to/image.jpg"

    with patch("src.agents.workflow.create_multi_agent_workflow") as mock_workflow:
        mock_workflow.return_value = Mock()
        mock_workflow.return_value.ainvoke = AsyncMock(return_value={
            "query": query,
            "response": "test response",
            "confidence": 0.8,
            "sources": [],
            "entities": [],
            "retrieved_docs": []
        })

        result = await query_with_agents(query, query_image=image_path)

        # Should handle image queries
        assert result["query"] == query


@pytest.mark.asyncio
async def test_workflow_metadata_added():
    """Test that metadata is added to result."""
    query = "test query"

    with patch("src.agents.workflow.create_multi_agent_workflow") as mock_workflow:
        mock_workflow.return_value = Mock()
        mock_workflow.return_value.ainvoke = AsyncMock(return_value={
            "query": query,
            "response": "test",
            "confidence": 0.8,
            "sources": [{"text": "source1"}, {"text": "source2"}],
            "entities": ["entity1", "entity2"],
            "retrieved_docs": [{"text": "doc1"}, {"text": "doc2"}]
        })

        result = await query_with_agents(query)

        # Verify metadata
        assert result["retrieved_count"] == 2
        assert len(result["entities"]) == 2


def test_initialize_phoenix_disabled():
    """Test Phoenix initialization when disabled."""
    os.environ["PHOENIX_ENABLED"] = "false"

    # Should not raise exception
    _initialize_phoenix()

    # Phoenix should remain disabled
    from src.agents.workflow import _phoenix_initialized
    assert _phoenix_initialized == False


@pytest.mark.asyncio
async def test_workflow_empty_sources():
    """Test workflow with no sources found."""
    query = "obscure query with no matches"

    with patch("src.agents.workflow.create_multi_agent_workflow") as mock_workflow:
        mock_workflow.return_value = Mock()
        mock_workflow.return_value.ainvoke = AsyncMock(return_value={
            "query": query,
            "response": "No relevant information found.",
            "sources": [],
            "entities": [],
            "retrieved_docs": []
        })

        result = await query_with_agents(query)

        # Should handle empty sources
        assert result["sources"] == []
        assert result["retrieved_count"] == 0
