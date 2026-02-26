import pytest
import numpy as np
import torch
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from src.utils.vision_embedding import (
    Qwen3VisionEmbedder,
    Qwen3TextLLM,
    Qwen3VLTextLLM,
    UnifiedQwen3VL,
    Qwen3VLEmbedding,
    Qwen3Embedding,
    Qwen2TextLLM,
    Phi35TextLLM,
    reset_models
)
from PIL import Image
import io

@pytest.fixture(autouse=True)
def cleanup_models():
    reset_models()
    yield
    reset_models()

def create_mock_image():
    file = io.BytesIO()
    image = Image.new('RGB', (100, 100))
    image.save(file, 'png')
    file.seek(0)
    return image

@pytest.mark.asyncio
async def test_qwen3_vision_embedder_analyze_image():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        # Mock generate output
        mock_outputs = torch.tensor([[1, 2, 3, 4]])
        mock_model.generate.return_value = mock_outputs
        mock_transformers.AutoModelForImageTextToText.from_pretrained.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "templated text"
        
        # Mock inputs which should have .to(device)
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda k: torch.tensor([[1, 2]]) if k == "input_ids" else None
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.tensor([[[[0.1]]]])}
        
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = ["Analyzed response"]
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        
        with patch("torch.cuda.is_available", return_value=False):
            embedder = Qwen3VisionEmbedder(model_name="test", device="cpu")
            img = create_mock_image()
            result = embedder.analyze_image(img, "test prompt")
            
            assert result == "Analyzed response"
            mock_model.generate.assert_called()

@pytest.mark.asyncio
async def test_qwen3_text_llm_generate():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_outputs = torch.tensor([[1, 2, 3]])
        mock_model.generate.return_value = mock_outputs
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda k: torch.tensor([[1, 2]]) if k == "input_ids" else None
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "Generated text"
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        
        with patch("torch.cuda.is_available", return_value=False):
            llm = Qwen3TextLLM(model_name="test", device="cpu")
            result = llm.generate("hello")
            
            assert result == "Generated text"
            mock_model.generate.assert_called()

@pytest.mark.asyncio
async def test_qwen3_embedding_embed_text():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        # Mock last_hidden_state
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 5, 4096)
        mock_model.return_value = mock_outputs
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        mock_tokenizer.return_value = mock_inputs
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        
        with patch("torch.cuda.is_available", return_value=False):
            model = Qwen3Embedding(model_name="test", device="cpu")
            result = model.embed_text("test text")
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (4096,)

def test_vision_embedder_cleanup():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoModelForImageTextToText.from_pretrained.return_value = mock_model
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        
        with patch("torch.cuda.is_available", return_value=False):
            embedder = Qwen3VisionEmbedder(model_name="test", device="cpu")
            embedder.cleanup()
            assert not hasattr(embedder, "model")
            assert not hasattr(embedder, "processor")
