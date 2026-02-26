import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.agents.embeddings import EmbeddingService

@pytest.fixture
def service():
    return EmbeddingService(device="cpu")

def test_embed_text(service):
    mock_model = MagicMock()
    mock_model.embed_text.return_value = np.array([0.1, 0.2, 0.3])
    
    with patch("src.agents.embeddings.get_embedding_model", return_value=mock_model):
        result = service.embed_text("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_model.embed_text.assert_called_with("hello")

def test_embed_image(service):
    mock_model = MagicMock()
    mock_model.embed_image.return_value = np.array([0.4, 0.5, 0.6])
    
    with patch("src.agents.embeddings.get_embedding_model", return_value=mock_model):
        result = service.embed_image("test.jpg")
        assert result == [0.4, 0.5, 0.6]
        mock_model.embed_image.assert_called_with("test.jpg")

def test_embed_multimodal(service):
    with (
        patch.object(service, "embed_text", return_value=[1.0, 1.0]),
        patch.object(service, "embed_image", return_value=[0.0, 0.0])
    ):
        result = service.embed_multimodal("text", "image.jpg")
        assert result == [0.5, 0.5]

def test_embed_batch(service):
    with patch.object(service, "embed_text") as mock_embed:
        mock_embed.side_effect = [[0.1], [0.2]]
        result = service.embed_batch(["a", "b"])
        assert result == [[0.1], [0.2]]
        assert mock_embed.call_count == 2
