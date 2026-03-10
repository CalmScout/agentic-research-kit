from unittest.mock import MagicMock, patch
from src.agents.reranker import Qwen3VLReranker, get_reranker

def test_reranker_basic():
    # Mock httpx.post to fail so we test fallback
    with patch("httpx.post", side_effect=Exception("API down")):
        reranker = Qwen3VLReranker(top_k=2)
        docs = [
            {"text": "doc 1", "score": 0.5},
            {"text": "doc 2", "score": 0.9},
            {"text": "doc 3", "score": 0.7}
        ]

        result = reranker.rerank("query", docs)
        assert len(result) == 2
        assert result[0]["text"] == "doc 2"
        assert result[1]["text"] == "doc 3"

def test_reranker_api_success():
    # Mock httpx.post to return specific scores
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"index": 0, "score": 0.1},
        {"index": 1, "score": 0.9},
        {"index": 2, "score": 0.5}
    ]
    
    with patch("httpx.post", return_value=mock_response):
        reranker = Qwen3VLReranker(top_k=2)
        docs = [
            {"text": "doc 1", "score": 0.5},
            {"text": "doc 2", "score": 0.9},
            {"text": "doc 3", "score": 0.7}
        ]

        result = reranker.rerank("query", docs)
        assert len(result) == 2
        assert result[0]["text"] == "doc 2"
        assert result[1]["text"] == "doc 3"

def test_reranker_with_scores():
    # Mock httpx.post to fail
    with patch("httpx.post", side_effect=Exception("API down")):
        reranker = Qwen3VLReranker(top_k=2)
        docs = [
            {"text": "doc 1", "score": 0.5},
            {"text": "doc 2", "score": 0.9}
        ]

        docs_out, scores = reranker.rerank_with_scores("query", docs)
        assert len(docs_out) == 2
        assert scores[0] == 0.9
        assert scores[1] == 0.5

def test_get_reranker():
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2
    assert isinstance(r1, Qwen3VLReranker)
