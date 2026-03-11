from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.workflow import (
    create_multi_agent_workflow,
    query_with_agents,
    query_with_agents_sync,
)


@pytest.mark.asyncio
async def test_create_workflow():
    """Test that workflow can be created successfully."""
    workflow = create_multi_agent_workflow()
    assert workflow is not None


@pytest.mark.asyncio
async def test_workflow_end_to_end():
    """Test full workflow with mocked nodes."""
    query = "Is climate change real?"

    # Mock all nodes
    with (
        patch("src.agents.workflow.skill_injector_node") as mock_skills,
        patch("src.agents.workflow.research_coordinator_node") as mock_coord,
        patch("src.agents.workflow.rag_search_node") as mock_rag,
        patch("src.agents.workflow.web_search_node") as mock_web,
        patch("src.agents.workflow.enhanced_response_generator_agent") as mock_gen,
        patch("src.agents.workflow.verification_agent") as mock_ver,
        patch("src.agents.workflow.MemoryStore") as mock_mem_class
    ):
        # Setup returns
        mock_skills.return_value = {"skill_instructions": ""}
        mock_coord.return_value = {"retrieved_docs": []}
        mock_rag.return_value = {"retrieved_docs": [{"content": "rag"}], "retrieval_method": "rag"}
        mock_web.return_value = {"retrieved_docs": [{"content": "rag"}, {"content": "web"}], "retrieval_method": "rag+web"}
        mock_gen.return_value = {"response": "Final answer", "sources": []}
        mock_ver.return_value = {"verification_status": "verified"}

        mock_mem = MagicMock()
        mock_mem.get_research_context.return_value = ""
        mock_mem_class.return_value = mock_mem

        result = await query_with_agents(query)

        # Verify
        assert result["response"] == "Final answer"
        assert mock_skills.called
        assert mock_coord.called
        assert mock_rag.called
        assert mock_web.called
        assert mock_gen.called
        assert mock_ver.called


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow handles errors gracefully."""
    query = "test query"

    # Mock workflow to raise exception
    with patch("src.agents.workflow.create_multi_agent_workflow", side_effect=Exception("Workflow failed")):
        result = await query_with_agents(query)

        # Should return error response
        assert "error" in result
        assert "Workflow failed" in result["error"]


def test_query_with_agents_sync():
    """Test synchronous wrapper for async function."""
    query = "test query"

    # Mock the async function
    with patch("src.agents.workflow.query_with_agents", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = {
            "query": query,
            "response": "test response",
            "sources": [],
            "retrieved_count": 0
        }

        result = query_with_agents_sync(query)

        # Verify result
        assert result["query"] == query
        assert result["response"] == "test response"
