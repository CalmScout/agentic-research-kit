"""Test configuration and shared fixtures for pytest."""

import pytest
import asyncio
import json
import tempfile
from typing import List, Dict, Any, AsyncGenerator, Optional
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------------
# Mock Models
# -----------------------------------------------------------------------------
class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, fixed_value: float = 0.0):
        """Initialize with fixed value for embeddings.

        Args:
            fixed_value: Value to use for all embedding dimensions
        """
        self.fixed_value = fixed_value

    def embed_text(self, text: str) -> List[float]:
        """Return fixed embedding vector for text."""
        # 2048-dim vector (Qwen3-VL-Embedding-2B)
        return [self.fixed_value] * 2048

    def embed_multimodal(self, text: str, image_path: str) -> List[float]:
        """Return fixed embedding for multimodal content."""
        return [self.fixed_value] * 2048


class MockLLM:
    """Mock LLM for testing."""

    def __init__(
        self,
        response: str = "Mock response",
        responses: Optional[List[str]] = None,
        fail_on_call: bool = False
    ):
        """Initialize mock LLM.

        Args:
            response: Default response string
            responses: List of responses to cycle through
            fail_on_call: If True, raise exception on invoke
        """
        self.response = response
        self.responses = responses or []
        self.call_count = 0
        self.fail_on_call = fail_on_call

    async def ainvoke(self, messages, **kwargs):
        """Mock async invoke."""
        if self.fail_on_call:
            raise Exception("Mock LLM configured to fail")

        from langchain_core.messages import AIMessage

        # Return next response from list if available
        if self.responses and self.call_count < len(self.responses):
            response_text = self.responses[self.call_count]
            self.call_count += 1
            return AIMessage(content=response_text)

        # Otherwise return default response
        return AIMessage(content=self.response)


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self.results = results or [
            {
                "text": "Mock result 1: Climate change is real.",
                "score": 0.9,
                "source": "mock_source_1",
                "doc_id": "doc_1",
            },
            {
                "text": "Mock result 2: Evidence shows temperature rising.",
                "score": 0.8,
                "source": "mock_source_2",
                "doc_id": "doc_2",
            },
        ]

    async def retrieve(self, query: str, top_k: int = 50) -> Dict[str, Any]:
        """Mock retrieval."""
        return {
            "retrieved_docs": self.results[:top_k],
            "retrieval_scores": [r["score"] for r in self.results[:top_k]],
            "retrieval_method": "mock",
        }


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self, reranked_results: Optional[List[Dict[str, Any]]] = None):
        """Initialize mock reranker.

        Args:
            reranked_results: Pre-reranked results to return
        """
        self.reranked_results = reranked_results or [
            {
                "text": "Reranked result 1: High confidence climate data.",
                "score": 0.95,
                "source": "source_1",
                "doc_id": "doc_1",
            },
            {
                "text": "Reranked result 2: Supporting evidence.",
                "score": 0.85,
                "source": "source_2",
                "doc_id": "doc_2",
            },
        ]

    async def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Mock reranking.

        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top results to return

        Returns:
            Reranked documents
        """
        return self.reranked_results[:top_k]


# -----------------------------------------------------------------------------
# Sample Test Data
# -----------------------------------------------------------------------------
SAMPLE_QUERY = "Is climate change real?"

SAMPLE_ENTITIES = ["climate change", "temperature"]

SAMPLE_IMAGE_PATH = "/path/to/test/image.jpg"

SAMPLE_CLAIMS = [
    {
        "text": "Climate change is real and caused by human activity.",
        "score": 0.9,
        "source": "https://example.com/claim1",
        "doc_id": "doc_1",
    },
    {
        "text": "Global temperatures have risen 1.5°C since pre-industrial times.",
        "score": 0.85,
        "source": "https://example.com/claim2",
        "doc_id": "doc_2",
    },
    {
        "text": "Ice caps are melting at unprecedented rates.",
        "score": 0.8,
        "source": "https://example.com/claim3",
        "doc_id": "doc_3",
    },
]

SAMPLE_RETRIEVAL = {
    "retrieved_docs": [
        {
            "text": "Mock result 1: Climate change is real.",
            "score": 0.9,
            "source": "mock_source_1",
        },
        {
            "text": "Mock result 2: Evidence shows temperature rising.",
            "score": 0.8,
            "source": "mock_source_2",
        },
    ],
    "retrieval_scores": [0.9, 0.8],
    "retrieval_method": "mock",
}

SAMPLE_EVIDENCE_SUMMARY = "Evidence shows climate change is real."

SAMPLE_TOP_CLAIMS = [
    {
        "text": "Climate change is real and caused by human activity.",
        "score": 0.9,
        "source": "https://example.com/claim1",
    }
]

SAMPLE_RESPONSE = """Based on the available evidence, climate change is real."""

SAMPLE_CONFIDENCE = 0.8


# -----------------------------------------------------------------------------
# Async Test Client
# -----------------------------------------------------------------------------
@pytest.fixture
async def async_test_client():
    """Async test client for FastAPI."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)


# -----------------------------------------------------------------------------
# Model Mocks
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def mock_llm():
    """Mock LLM."""
    return MockLLM()


@pytest.fixture
def mock_retriever():
    """Mock retriever."""
    return MockRetriever()


# -----------------------------------------------------------------------------
# Sample Data
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return SAMPLE_QUERY


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return SAMPLE_ENTITIES


@pytest.fixture
def sample_retrieval():
    """Sample retrieval result for testing."""
    return SAMPLE_RETRIEVAL


@pytest.fixture
def sample_evidence_summary():
    """Sample evidence summary for testing."""
    return SAMPLE_EVIDENCE_SUMMARY


@pytest.fixture
def sample_top_claims():
    """Sample top claims for testing."""
    return SAMPLE_TOP_CLAIMS


@pytest.fixture
def sample_response():
    """Sample response for testing."""
    return SAMPLE_RESPONSE


@pytest.fixture
def sample_confidence():
    """Sample confidence score."""
    return SAMPLE_CONFIDENCE


@pytest.fixture
def sample_image_path():
    """Sample image path for testing."""
    return SAMPLE_IMAGE_PATH


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return SAMPLE_CLAIMS


# -----------------------------------------------------------------------------
# Agent State Fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def agent_state():
    """Create a sample AgentState for testing."""
    from src.agents.base_state import BaseAgentState

    return {
        "query": SAMPLE_QUERY,
        "query_image": None,
        "memory_context": None,
        "query_type": "text",
        "entities": SAMPLE_ENTITIES,
        "query_embedding": [0.0] * 2048,
        "retrieved_docs": SAMPLE_CLAIMS,
        "retrieval_scores": [0.9, 0.85, 0.8],
        "retrieval_method": "mock",
        "reranked_docs": SAMPLE_CLAIMS[:2],
        "evidence_summary": SAMPLE_EVIDENCE_SUMMARY,
        "top_results": SAMPLE_CLAIMS[:2],  # Changed from top_claims
        "response": SAMPLE_RESPONSE,
        "confidence": SAMPLE_CONFIDENCE,
        "sources": SAMPLE_CLAIMS[:2],
        "messages": [],
    }


@pytest.fixture
def agent_state_minimal():
    """Create minimal AgentState (beginning of workflow)."""
    from src.agents.base_state import BaseAgentState

    return {
        "query": SAMPLE_QUERY,
        "query_image": None,
        "memory_context": None,
        "query_type": "",
        "entities": [],
        "query_embedding": [],
        "retrieved_docs": [],
        "retrieval_scores": [],
        "retrieval_method": "",
        "reranked_docs": [],
        "evidence_summary": "",
        "top_results": [],  # Changed from top_claims
        "response": "",
        "confidence": 0.0,
        "sources": [],
        "messages": [],
    }


@pytest.fixture
def agent_state_with_docs():
    """Create AgentState with retrieved documents (for testing response generator)."""
    from src.agents.base_state import BaseAgentState

    return {
        "query": SAMPLE_QUERY,
        "query_image": None,
        "memory_context": None,
        "query_type": "text",
        "entities": SAMPLE_ENTITIES,
        "query_embedding": [0.0] * 2048,
        "retrieved_docs": SAMPLE_CLAIMS,
        "retrieval_scores": [0.9, 0.85, 0.8],
        "retrieval_method": "mock",
        "reranked_docs": [],
        "evidence_summary": "",
        "top_results": [],
        "response": "",
        "confidence": 0.0,
        "sources": [],
        "messages": [],
    }


# -----------------------------------------------------------------------------
# Additional Mock Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_reranker():
    """Mock reranker."""
    return MockReranker()


@pytest.fixture
def mock_llm_with_fail():
    """Mock LLM that fails on invoke."""
    return MockLLM(fail_on_call=True)


@pytest.fixture
def mock_llm_with_responses():
    """Mock LLM with predefined responses."""
    return MockLLM(responses=["First response", "Second response", "Third response"])


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_json_file(temp_dir):
    """Create a mock JSON file for testing."""
    json_path = temp_dir / "test.json"
    test_data = {"key": "value", "nested": {"item": 123}}
    json_path.write_text(json.dumps(test_data))
    return json_path


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def test_config(mock_embedding_model, mock_llm, mock_retriever):
    """Configure test with mocked models.

    This fixture is autouse=True to ensure all tests use mocked models
    and prevent real LLM loading which causes slowness.
    """
    with patch("src.utils.vision_embedding.get_embedding_model", return_value=mock_embedding_model), \
         patch("src.utils.vision_embedding.get_qwen2_llm", return_value=mock_llm), \
         patch("src.utils.vision_embedding.get_vision_model", return_value=mock_llm), \
         patch("src.utils.vision_embedding.get_text_llm", return_value=mock_llm), \
         patch("src.utils.vision_embedding.get_unified_model", return_value=mock_llm), \
         patch("src.utils.vision_embedding.Qwen3VLEmbedding", return_value=mock_embedding_model), \
         patch("src.utils.vision_embedding.Qwen3Embedding", return_value=mock_embedding_model), \
         patch("src.utils.vision_embedding.Qwen3VisionEmbedder", return_value=mock_llm), \
         patch("src.utils.vision_embedding.Qwen2TextLLM", return_value=mock_llm), \
         patch("src.utils.vision_embedding.UnifiedQwen3VL", return_value=mock_llm):
        
        try:
            # Monkey-patch the model singletons if they exist
            from src.agents import embeddings, model_selector, simple_retriever

            # Save original singletons (if they exist)
            original_embedding = getattr(embeddings, 'embedder', None)
            original_model_selector = getattr(model_selector, '_selector', None)
            original_retriever = getattr(simple_retriever, '_retriever_instance', None)

            # Create a mock selector that returns the mock LLM
            mock_selector = Mock()
            mock_selector.get_local_llm.return_value = mock_llm
            mock_selector.get_llm_with_fallback.return_value = mock_llm
            mock_selector.get_llm_for_provider.return_value = mock_llm

            # Patch if possible
            if hasattr(embeddings, 'embedder'):
                embeddings.embedder = mock_embedding_model
            if hasattr(model_selector, '_selector'):
                model_selector._selector = mock_selector
            if hasattr(simple_retriever, '_retriever_instance'):
                simple_retriever._retriever_instance = mock_retriever

            yield mock_embedding_model, mock_llm, mock_retriever

            # Restore original singletons
            if original_embedding is not None:
                embeddings.embedder = original_embedding
            if original_model_selector is not None:
                model_selector._selector = original_model_selector
            if original_retriever is not None:
                simple_retriever._retriever_instance = original_retriever

        except Exception:
            # If patching fails, just yield the mocks
            yield mock_embedding_model, mock_llm, mock_retriever
