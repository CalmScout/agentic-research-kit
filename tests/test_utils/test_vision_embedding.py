from unittest.mock import MagicMock, patch

import pytest

from src.utils.vision_embedding import (
    Phi35TextLLM,
    Qwen2TextLLM,
    Qwen3Embedding,
    Qwen3TextLLM,
    Qwen3VisionEmbedder,
    Qwen3VLEmbedding,
    Qwen3VLTextLLM,
    UnifiedQwen3VL,
    get_quantization_config,
    get_vision_model,
    reset_models,
)


@pytest.fixture(autouse=True)
def cleanup_models():
    reset_models()
    yield
    reset_models()

def test_get_quantization_config_none():
    with patch("os.getenv", return_value="false"):
        config = get_quantization_config()
        assert config is None

def test_get_quantization_config_4bit():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("os.getenv") as mock_env:
            mock_env.side_effect = lambda k, d=None: "true" if k == "USE_4BIT_QUANTIZATION" else "false"
            config = get_quantization_config()
            assert config is not None
            assert mock_transformers.BitsAndBytesConfig.called

def test_qwen3_vision_embedder_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            embedder = Qwen3VisionEmbedder(model_name="test-model", device="cpu")
            assert embedder.model_name == "test-model"
            assert mock_transformers.AutoModelForImageTextToText.from_pretrained.called

def test_qwen3_text_llm_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            llm = Qwen3TextLLM(model_name="test-text", device="cpu")
            assert llm.model_name == "test-text"
            assert mock_transformers.AutoModelForCausalLM.from_pretrained.called

def test_qwen3_vl_text_llm_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            llm = Qwen3VLTextLLM(model_name="test-vl-text", device="cpu")
            assert llm.model_name == "test-vl-text"
            assert mock_transformers.AutoModelForImageTextToText.from_pretrained.called

def test_unified_qwen3_vl_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            model = UnifiedQwen3VL(model_name="test-unified", device="cpu")
            assert model.model_name == "test-unified"
            assert mock_transformers.AutoModelForImageTextToText.from_pretrained.called

def test_qwen3_vl_embedding_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            model = Qwen3VLEmbedding(model_name="test-vl-embed", device="cpu")
            assert model.model_name == "test-vl-embed"
            assert mock_transformers.AutoModel.from_pretrained.called

def test_qwen3_embedding_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            model = Qwen3Embedding(model_name="test-embed", device="cpu")
            assert model.model_name == "test-embed"
            assert mock_transformers.AutoModel.from_pretrained.called

def test_qwen2_text_llm_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            llm = Qwen2TextLLM(model_name="test-qwen2", device="cpu")
            assert llm.model_name == "test-qwen2"
            assert mock_transformers.AutoModelForCausalLM.from_pretrained.called

def test_phi35_text_llm_init():
    mock_transformers = MagicMock()
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            llm = Phi35TextLLM(model_name="test-phi35", device="cpu")
            assert llm.model_name == "test-phi35"
            assert mock_transformers.AutoModelForCausalLM.from_pretrained.called

def test_get_vision_model_singleton():
    with patch("src.utils.vision_embedding.Qwen3VisionEmbedder") as mock_class:
        model1 = get_vision_model(model_name="test")
        model2 = get_vision_model(model_name="test")
        assert model1 is model2
        assert mock_class.called
        assert mock_class.call_count == 1
