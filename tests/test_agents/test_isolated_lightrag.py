import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.agents.isolated_lightrag import IsolatedLightRAG, create_isolated_lightrag

@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.aquery = AsyncMock(return_value="query result")
    rag.aquery_data = AsyncMock(return_value={"data": "structured result"})
    rag.asearch = AsyncMock(return_value="search result")
    rag.ainsert = AsyncMock()
    return rag

@pytest.fixture
def rag_factory(mock_rag):
    return lambda: mock_rag

def test_isolated_lightrag_init():
    factory = lambda: MagicMock()
    isolated = IsolatedLightRAG(factory, max_workers=2, timeout=30.0)
    assert isolated.rag_factory == factory
    assert isolated.max_workers == 2
    assert isolated.timeout == 30.0
    isolated.close()

def test_aquery_sync(mock_rag, rag_factory):
    isolated = IsolatedLightRAG(rag_factory)
    try:
        result = isolated.aquery_sync("test query", mode="local")
        assert result == "query result"
        mock_rag.aquery.assert_called_once()
    finally:
        isolated.close()

def test_aquery_sync_structured(mock_rag, rag_factory):
    isolated = IsolatedLightRAG(rag_factory)
    try:
        result = isolated.aquery_sync("test query", only_need_context=True)
        assert result == {"data": "structured result"}
        mock_rag.aquery_data.assert_called_once()
    finally:
        isolated.close()

def test_asearch_sync(mock_rag, rag_factory):
    isolated = IsolatedLightRAG(rag_factory)
    try:
        result = isolated.asearch_sync("test search")
        assert result == "search result"
        mock_rag.asearch.assert_called_once()
    finally:
        isolated.close()

def test_ainsert_sync(mock_rag, rag_factory):
    isolated = IsolatedLightRAG(rag_factory)
    try:
        isolated.ainsert_sync("new text")
        mock_rag.ainsert.assert_called_once_with("new text")
    finally:
        isolated.close()

def test_create_isolated_lightrag(rag_factory):
    isolated = create_isolated_lightrag(rag_factory)
    assert isinstance(isolated, IsolatedLightRAG)
    isolated.close()

def test_context_manager(rag_factory):
    with IsolatedLightRAG(rag_factory) as isolated:
        assert isinstance(isolated, IsolatedLightRAG)
    # Should be closed after with block
