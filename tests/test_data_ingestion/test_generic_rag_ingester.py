from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.generic_rag_ingester import GenericRAGIngester

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.embed_query.return_value = [0.0] * 1536
    return model

@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.ainsert = AsyncMock()
    return rag

@patch("src.data_ingestion.generic_rag_ingester.LightRAG")
@patch("langchain_openai.OpenAIEmbeddings")
def test_ingester_init(mock_embeddings_class, mock_lightrag, mock_embedding_model):
    mock_embeddings_class.return_value = mock_embedding_model
    mock_lightrag.return_value = MagicMock()

    ingester = GenericRAGIngester(working_dir="./test_rag")

    assert ingester.working_dir == "./test_rag"
    mock_embeddings_class.assert_called()
    mock_lightrag.assert_called()

@pytest.mark.asyncio
@patch("src.data_ingestion.generic_rag_ingester.LightRAG")
@patch("langchain_openai.OpenAIEmbeddings")
async def test_ingest_df(mock_embeddings_class, mock_lightrag, mock_embedding_model):
    mock_embeddings_class.return_value = mock_embedding_model
    rag_instance = MagicMock()
    rag_instance.initialize_storages = AsyncMock()
    rag_instance.ainsert = AsyncMock()
    mock_lightrag.return_value = rag_instance

    ingester = GenericRAGIngester()
    df = pd.DataFrame({"content": ["text1", "text2"], "title": ["t1", "t2"]})

    stats = await ingester.ingest_df(df)

    assert stats["total_items"] == 2
    assert rag_instance.initialize_storages.called
    assert rag_instance.ainsert.call_count == 2

def test_format_content():
    # We need to mock the init to avoid model loading
    with patch("langchain_openai.OpenAIEmbeddings"), \
         patch("src.data_ingestion.generic_rag_ingester.LightRAG"):
        ingester = GenericRAGIngester(content_template="Title: {title}\nBody: {content}")
        row = pd.Series({"title": "My Title", "content": "My Content"})

        formatted = ingester._format_content(row)
        assert formatted == "Title: My Title\nBody: My Content"

def test_extract_metadata():
    with patch("langchain_openai.OpenAIEmbeddings"), \
         patch("src.data_ingestion.generic_rag_ingester.LightRAG"):
        ingester = GenericRAGIngester(metadata_fields=["category", "author"])
        row = pd.Series({"category": "science", "author": "John Doe", "extra": "val"})

        metadata = ingester._extract_metadata(row, idx=0, content_id="doc_0")
        assert metadata["category"] == "science"
        assert metadata["author"] == "John Doe"
        assert "extra" not in metadata
        assert metadata["doc_id"] == 0
