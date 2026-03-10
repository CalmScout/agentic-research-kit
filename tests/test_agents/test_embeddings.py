from unittest.mock import MagicMock, patch

import pytest
from src.agents.embeddings import EmbeddingService

@pytest.fixture
def service():
    return EmbeddingService(device="cpu")

def test_embed_text(service):
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

    with patch.object(service, "_get_model", return_value=mock_model):
        result = service.embed_text("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_model.embed_query.assert_called_with("hello")

def test_embed_image(service):
    # Now it returns a vector of zeros
    result = service.embed_image("test.jpg")
    assert result == [0.0] * 1024

def test_embed_multimodal(service):
    with (
        patch.object(service, "embed_text", return_value=[1.0, 1.0])
    ):
        result = service.embed_multimodal("text", "image.jpg")
        assert result == [1.0, 1.0]

def test_embed_batch(service):
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1], [0.2]]
    with patch.object(service, "_get_model", return_value=mock_model):
        result = service.embed_batch(["a", "b"])
        assert result == [[0.1], [0.2]]
        mock_model.embed_documents.assert_called_once_with(["a", "b"])
