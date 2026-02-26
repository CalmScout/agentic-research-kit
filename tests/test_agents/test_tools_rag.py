import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from src.agents.tools.rag_tools.hybrid_retriever import HybridRetrieverTool

@pytest.fixture
def tool():
    return HybridRetrieverTool()

@pytest.mark.asyncio
async def test_hybrid_retriever_tool_success(tool):
    mock_isolated_rag = MagicMock()
    mock_isolated_rag.aquery_sync.return_value = {
        "status": "success",
        "data": {
            "chunks": [
                {"content": "chunk 1", "score": 0.9, "chunk_id": "c1"},
                {"content": "chunk 2", "score": 0.8, "chunk_id": "c2"}
            ]
        }
    }
    
    with (
        patch.object(HybridRetrieverTool, "_initialize_isolated_rag"),
        patch.object(tool, "isolated_rag", mock_isolated_rag)
    ):
        tool._initialized = True
        
        result_json = await tool.execute(query="test query", top_k=2)
        result = json.loads(result_json)
        
        assert result["retrieval_method"] == "hybrid"
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0]["text"] == "chunk 1"
        assert result["retrieval_scores"][0] == 0.9

@pytest.mark.asyncio
async def test_hybrid_retriever_tool_fallback(tool):
    mock_isolated_rag = MagicMock()
    mock_isolated_rag.aquery_sync.side_effect = Exception("LightRAG failed")
    
    mock_fallback_result = {
        "retrieved_docs": [{"text": "fallback doc", "score": 0.5, "metadata": {}}],
        "retrieval_method": "keyword"
    }
    
    with (
        patch.object(HybridRetrieverTool, "_initialize_isolated_rag"),
        patch.object(tool, "isolated_rag", mock_isolated_rag),
        patch("src.agents.tools.rag_tools.hybrid_retriever.simple_retriever", new_callable=AsyncMock) as mock_simple
    ):
        
        tool._initialized = True
        mock_simple.return_value = mock_fallback_result
        
        result_json = await tool.execute(query="test query")
        result = json.loads(result_json)
        
        assert result["retrieval_method"] == "keyword"
        assert result["retrieved_docs"][0]["text"] == "fallback doc"

def test_hybrid_retriever_tool_validate_params(tool):
    assert len(tool.validate_params({"query": "test"})) == 0
    assert "Missing required parameter: 'query'" in tool.validate_params({})
    assert "Parameter 'top_k' must be an integer" in tool.validate_params({"query": "t", "top_k": "invalid"})
