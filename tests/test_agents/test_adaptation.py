"""Tests for adapted features from nanobot."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.enhanced_retriever import enhanced_retriever_agent
from src.agents.memory.store import MemoryStore
from src.agents.model_selector import ModelSelector
from src.agents.providers import find_by_name
from src.utils.config import Settings


@pytest.mark.asyncio
async def test_provider_registry_and_selector():
    """Test that ModelSelector correctly uses the ProviderRegistry."""
    # Patch environment before initializing Settings
    import os
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
        settings = Settings()
        selector = ModelSelector(settings)

        # Test finding providers
        deepseek_spec = find_by_name("deepseek")
        assert deepseek_spec is not None
        assert deepseek_spec.display_name == "DeepSeek"

        # Test initializing an API provider (mocked)
        with patch("src.agents.model_selector.ChatOpenAI") as mock_chat:
            llm = selector.get_llm_for_provider("deepseek")
            assert llm is not None
            mock_chat.assert_called_once()
            args, kwargs = mock_chat.call_args
            assert kwargs["api_key"].get_secret_value() == "test-key"
            assert kwargs["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_memory_consolidation():
    """Test LLM-based memory consolidation."""
    with patch("tempfile.TemporaryDirectory") as tmp_dir:
        workspace = Path(tmp_dir)
        memory = MemoryStore(workspace)

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "The research confirms that X causes Y.",
            "research_memory_update": "# Research Findings\n\n- Fact 1: X causes Y."
        })
        mock_llm.ainvoke.return_value = mock_response

        session_queries = [
            {"query": "Does X cause Y?", "response": "Yes, research shows...", "confidence": 0.9}
        ]

        summary = await memory.consolidate_session(session_queries, llm=mock_llm)

        assert summary == "The research confirms that X causes Y."
        assert "Fact 1: X causes Y" in memory.read_long_term()


@pytest.mark.asyncio
async def test_enhanced_retriever_web_fallback():
    """Test that EnhancedRetriever triggers web search when local results are low."""
    state = {
        "query": "Who won the super bowl 2026?",
        "query_image": None,
    }

    # Mock settings with Brave key
    mock_settings = MagicMock()
    mock_settings.mcp_servers = []
    mock_settings.brave_api_key = "fake-brave-key"
    mock_settings.retrieval_top_k = 5

    # Mock registry and tools
    with patch("src.agents.enhanced_retriever.get_settings", return_value=mock_settings):
        with patch("src.agents.enhanced_retriever.ToolRegistry") as mock_registry_class:
            mock_registry = mock_registry_class.return_value
            mock_registry.close = AsyncMock() # Fix: make close an AsyncMock

            # Entity extractor returns empty list
            mock_registry.execute.side_effect = [
                AsyncMock(return_value="[]")(), # entity_extractor
                AsyncMock(return_value=json.dumps({
                    "retrieved_docs": [], # ZERO local docs
                    "retrieval_scores": [],
                    "retrieval_method": "hybrid"
                }))(), # hybrid_retriever
                AsyncMock(return_value="1. The team X won. http://news.com")(), # web_search
            ]

            with patch("src.agents.enhanced_retriever.embedder") as mock_embedder:
                mock_embedder.embed_text.return_value = [0.1] * 2048

                result = await enhanced_retriever_agent(state)

                # Should have triggered web search (3rd call to execute)
                assert mock_registry.execute.call_count == 3
                assert "web" in result["retrieval_method"]
                assert len(result["retrieved_docs"]) > 0
                assert result["retrieved_docs"][0]["metadata"]["type"] == "web_search"
