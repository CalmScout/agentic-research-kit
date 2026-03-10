import json
from unittest.mock import mock_open, patch

import pytest

from src.agents.simple_retriever import simple_retriever


@pytest.mark.asyncio
async def test_simple_retriever_success():
    docs_data = {
        "doc1": {"content": "This is a test document about climate change.", "file_path": "path1.txt"},
        "doc2": {"content": "Another document about artificial intelligence.", "file_path": "path2.txt"}
    }

    vdb_data = {"matrix": [], "embedding_dim": 2048}
    entity_data = {}

    # Mock multiple open calls
    def side_effect(path):
        if "kv_store_full_docs.json" in path:
            return mock_open(read_data=json.dumps(docs_data))()
        if "vdb_chunks.json" in path:
            return mock_open(read_data=json.dumps(vdb_data))()
        if "kv_store_entity_chunks.json" in path:
            return mock_open(read_data=json.dumps(entity_data))()
        return mock_open()()

    with patch("builtins.open", side_effect=side_effect), \
         patch("os.path.exists", return_value=True):
        result = await simple_retriever("climate change", top_k=5)

        assert result["retrieval_method"] == "keyword"
        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0]["doc_id"] == "doc1"
        assert result["retrieval_scores"][0] > 0

@pytest.mark.asyncio
async def test_simple_retriever_no_match():
    docs_data = {"doc1": {"content": "abc", "file_path": "p1"}}

    with patch("builtins.open", mock_open(read_data=json.dumps(docs_data))), \
         patch("os.path.exists", return_value=True):
        # This will fail for subsequent opens if not handled, but let's assume it returns empty for them
        result = await simple_retriever("xyz")
        assert len(result["retrieved_docs"]) == 0
