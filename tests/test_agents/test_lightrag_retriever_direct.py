from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from src.agents.direct_lightrag_retriever import (
    DirectLightRAGRetriever,
    _patched_get_storage_class,
    get_direct_lightrag_retriever,
    direct_hf_embedding_wrapper,
    direct_hf_llm_wrapper
)
from src.agents.lancedb_storage import (
    LanceDBDocStatusStorage,
    LanceDBKVStorage,
    LanceDBVectorDBStorage,
)

@pytest.fixture
def retriever():
    return DirectLightRAGRetriever(working_dir="./test_rag_storage", device="cpu")

def test_patched_get_storage_class():
    mock_self = MagicMock()
    assert _patched_get_storage_class(mock_self, "LanceDBKVStorage") == LanceDBKVStorage
    assert _patched_get_storage_class(mock_self, "LanceDBDocStatusStorage") == LanceDBDocStatusStorage
    assert _patched_get_storage_class(mock_self, "LanceDBVectorDBStorage") == LanceDBVectorDBStorage
    with patch("src.agents.direct_lightrag_retriever._original_get_storage_class") as mock_orig:
        mock_orig.return_value = "original_class"
        assert _patched_get_storage_class(mock_self, "JsonKVStorage") == "original_class"

@pytest.mark.asyncio
async def test_direct_hf_embedding_wrapper():
    """Test the standalone embedding wrapper."""
    mock_model = MagicMock()
    mock_model.aembed_documents = AsyncMock(return_value=[[0.1] * 1024])
    
    with patch("src.agents.direct_lightrag_retriever.OpenAIEmbeddings", return_value=mock_model):
        # Reset singleton for test
        import src.agents.direct_lightrag_retriever as dlr
        dlr._wrapper_embeddings = None
        
        result = await direct_hf_embedding_wrapper(["text"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1024)
        mock_model.aembed_documents.assert_called_once()

@pytest.mark.asyncio
async def test_direct_hf_llm_wrapper():
    """Test the standalone LLM wrapper."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"result": "ok"}'))
    
    with patch("src.agents.direct_lightrag_retriever.ThinkingProcessStripper", return_value=mock_llm):
        # Reset singleton for test
        import src.agents.direct_lightrag_retriever as dlr
        dlr._wrapper_llm = None
        
        result = await direct_hf_llm_wrapper("prompt")
        assert result == '{"result": "ok"}'
        mock_llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_success(retriever):
    mock_rag = MagicMock()
    mock_rag.aquery_data = AsyncMock(return_value={
        "status": "success",
        "data": {
            "chunks": [
                {"content": "chunk 1", "score": 0.95, "file_path": "f1.txt", "chunk_id": "c1"},
                {"content": "chunk 2", "score": 0.90, "file_path": "f2.txt", "chunk_id": "c2"}
            ]
        }
    })

    with patch.object(retriever, "get_rag", return_value=mock_rag):
        result = await retriever.retrieve("query", top_k=2, mode="naive")

        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0]["text"] == "chunk 1"
        assert result["retrieved_docs"][0]["score"] == 0.95
        assert result["retrieval_method"] == "hybrid_naive"

@pytest.mark.asyncio
async def test_retrieve_error(retriever):
    mock_rag = MagicMock()
    mock_rag.aquery_data = AsyncMock(side_effect=Exception("RAG failure"))

    with patch.object(retriever, "get_rag", return_value=mock_rag):
        result = await retriever.retrieve("query")
        assert "error" in result
        assert "RAG failure" in result["error"]

def test_get_direct_lightrag_retriever():
    r1 = get_direct_lightrag_retriever()
    r2 = get_direct_lightrag_retriever()
    assert r1 is r2
    assert isinstance(r1, DirectLightRAGRetriever)
