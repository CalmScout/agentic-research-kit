import logging
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def get_quantization_config():
    """Get bitsandbytes quantization config if enabled in environment."""
    use_4bit = os.getenv("USE_4BIT_QUANTIZATION", "false").lower() == "true"
    use_8bit = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"

    if use_4bit:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # Enable fallback to CPU for small parts
        )
    return None


class Qwen3VisionEmbedder:
    """
    GPU-accelerated Qwen3-VL model for multimodal vision-language tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        batch_size: int = 4,
        quantization: str | None = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.batch_size = batch_size
        self.model_name = model_name

        logger.info(f"Loading {model_name} on {self.device}...")

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            quant_config = get_quantization_config()

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                quantization_config=quant_config,
            )

            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )

            logger.info(f"✓ {model_name} loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def analyze_image(
        self, image: str | Path | Image.Image, prompt: str, context: str = "", max_tokens: int = 512
    ) -> str:
        try:
            if isinstance(image, (str, Path)):
                img: Image.Image = Image.open(image)
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            full_prompt = f"{prompt}"
            if context:
                full_prompt += f"\n\nContext:\n{context}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False, temperature=1.0
                )

            response = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {e}"

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"✓ Unloaded {self.model_name} from GPU")

    def _clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def batch_analyze(
        self,
        images: list[str | Path | Image.Image],
        prompts: list[str],
        contexts: list[str] | None = None,
        max_tokens: int = 512,
    ) -> list[str]:
        if contexts is None:
            contexts = [""] * len(images)

        results = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]
            batch_contexts = contexts[i : i + self.batch_size]

            for img, prompt, ctx in zip(batch_images, batch_prompts, batch_contexts, strict=False):
                result = self.analyze_image(img, prompt, ctx, max_tokens)
                results.append(result)

            self._clear_cache()
        return results


class Qwen3TextLLM:
    """
    GPU-accelerated Qwen3-8B model for text generation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        torch_dtype: str = "float16",
        quantization: str | None = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.model_name = model_name

        logger.info(f"Loading {model_name} on {self.device}...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            quant_config = get_quantization_config()

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                quantization_config=quant_config,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            logger.info(f"✓ {model_name} loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"


class Qwen3VLTextLLM:
    """
    GPU-accelerated Qwen3-VL model for text generation (text-only mode).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        quantization: str | None = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.model_name = model_name

        logger.info(f"Loading {model_name} on {self.device} for text generation...")

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            quant_config = get_quantization_config()

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                quantization_config=quant_config,
            )

            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )

            logger.info(f"✓ {model_name} loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            if system_prompt:
                messages.insert(
                    0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                )

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                )

            response = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"


class UnifiedQwen3VL:
    """
    Unified Qwen3-VL model that handles both text and multimodal tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        batch_size: int = 4,
        quantization: str | None = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.batch_size = batch_size
        self.model_name = model_name

        logger.info(f"Loading unified Qwen3-VL model {model_name} on {self.device}...")

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            quant_config = get_quantization_config()

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                quantization_config=quant_config,
            )

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            logger.info(f"✓ Unified {model_name} loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def analyze_image(
        self, image: str | Path | Image.Image, prompt: str, context: str = "", max_tokens: int = 512
    ) -> str:
        try:
            if isinstance(image, (str, Path)):
                img: Image.Image = Image.open(image)
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            full_prompt = f"{prompt}"
            if context:
                full_prompt += f"\n\nContext:\n{context}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False, temperature=1.0
                )

            response = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {e}"

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            if system_prompt:
                messages.insert(
                    0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                )

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                )

            response = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"✓ Unloaded {self.model_name} from GPU")


class Qwen3VLEmbedding:
    """
    Multimodal embedding model for text + images.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.model_name = model_name

        logger.info(f"Loading multimodal embedding model {model_name} on {self.device}...")

        try:
            from transformers import AutoModel, AutoTokenizer

            # Note: Qwen3-VL-Embedding-2B architecture is incompatible with bitsandbytes quantization
            # Since it's only 2B parameters (~4GB), we load it in full float16/bfloat16
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                # quantization_config=None (explicitly skip)
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            logger.info(f"✓ Multimodal embedding model {model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embedding_np = embedding.cpu().numpy().flatten()

            return cast(np.ndarray, embedding_np.astype(np.float32))

        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.zeros(2048, dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        try:
            from PIL import Image

            image = Image.open(image_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                    ],
                }
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            inputs = self.tokenizer(
                text=[text], images=[image], return_tensors="pt", padding=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embedding_np = embedding.cpu().numpy().flatten()

            return cast(np.ndarray, embedding_np.astype(np.float32))

        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return np.zeros(2048, dtype=np.float32)

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"✓ Unloaded {self.model_name} from GPU")


class Qwen3Embedding:
    """
    GPU-accelerated Qwen3-Embedding model for vector embeddings.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.model_name = model_name

        logger.info(f"Loading {model_name} on {self.device}...")

        try:
            from transformers import AutoModel, AutoTokenizer

            quant_config = get_quantization_config()

            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                quantization_config=quant_config,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            logger.info(f"✓ {model_name} loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embedding_np = embedding.cpu().numpy().flatten()

            return cast(np.ndarray, embedding_np.astype(np.float32))

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(4096, dtype=np.float32)

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return np.vstack(embeddings).astype(np.float32)


# Singleton pattern for model instances
_vision_model: Qwen3VisionEmbedder | None = None
_text_llm: Qwen3TextLLM | Qwen3VLTextLLM | None = None
_embedding_model: Any | None = None
_unified_model: UnifiedQwen3VL | None = None
_qwen2_llm = None
_phi35_llm = None


def reset_models():
    """Reset all model singletons (primarily for testing)."""
    global _vision_model, _text_llm, _embedding_model, _unified_model, _qwen2_llm, _phi35_llm
    _vision_model = None
    _text_llm = None
    _embedding_model = None
    _unified_model = None
    _qwen2_llm = None
    _phi35_llm = None


def get_unified_model(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "cuda",
    torch_dtype: str = "float16",
    batch_size: int = 4,
) -> UnifiedQwen3VL:
    """Get or create unified model singleton."""
    global _unified_model

    if _unified_model is None:
        _unified_model = UnifiedQwen3VL(
            model_name=model_name, device=device, torch_dtype=torch_dtype, batch_size=batch_size
        )

    return _unified_model


def get_vision_model(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda",
    torch_dtype: str = "float16",
) -> Qwen3VisionEmbedder | UnifiedQwen3VL:
    """Get or create vision model singleton (with unified model support)."""
    global _vision_model, _unified_model

    text_model_name = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen3-8B")
    if model_name == text_model_name and ("VL" in model_name or "vl" in model_name.lower()):
        return get_unified_model(model_name=model_name, device=device, torch_dtype=torch_dtype)

    if _vision_model is None:
        _vision_model = Qwen3VisionEmbedder(
            model_name=model_name, device=device, torch_dtype=torch_dtype
        )

    return _vision_model


def get_text_llm(
    model_name: str = "Qwen/Qwen3-8B", device: str = "cuda", torch_dtype: str = "float16"
) -> Qwen3TextLLM | Qwen3VLTextLLM | UnifiedQwen3VL:
    """Get or create text LLM singleton (with unified model support)."""
    global _text_llm, _unified_model

    vision_model_name = os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    if model_name == vision_model_name and ("VL" in model_name or "vl" in model_name.lower()):
        return get_unified_model(model_name=model_name, device=device, torch_dtype=torch_dtype)

    if _text_llm is None:
        if "VL" in model_name or "vl" in model_name.lower():
            _text_llm = Qwen3VLTextLLM(
                model_name=model_name, device=device, torch_dtype=torch_dtype
            )
        else:
            _text_llm = Qwen3TextLLM(model_name=model_name, device=device, torch_dtype=torch_dtype)

    return _text_llm


def get_embedding_model(
    model_name: str = "Qwen/Qwen3-Embedding-8B", device: str = "cuda", torch_dtype: str = "float16"
) -> Qwen3Embedding | Qwen3VLEmbedding:
    """Get or create embedding model singleton."""
    global _embedding_model

    if _embedding_model is None:
        if "VL" in model_name or "vl" in model_name.lower():
            _embedding_model = Qwen3VLEmbedding(
                model_name=model_name, device=device, torch_dtype=torch_dtype
            )
        else:
            _embedding_model = Qwen3Embedding(
                model_name=model_name, device=device, torch_dtype=torch_dtype
            )

    return _embedding_model


class Qwen2TextLLM:
    """Qwen2 for fast local entity extraction.

    This class provides a fast, lightweight LLM for entity extraction tasks.
    Qwen2-1.5B-Instruct is chosen for its speed, small footprint (~1GB VRAM), and
    compatibility with KV caching (unlike Phi-3.5 which has DynamicCache issues).

    Note: We use Qwen2.5-1.5B-Instruct for better instruction following and multilingual support.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        max_new_tokens: int = 512,
    ):
        """
        Initialize Qwen2.5 model.

        Args:
            model_name: HuggingFace model name (default: Qwen2.5-1.5B-Instruct)
            device: Target device ("cuda" or "cpu")
            torch_dtype: Data type for model weights
            max_new_tokens: Maximum tokens to generate
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        logger.info(f"Loading Qwen2.5 model {model_name} on {self.device}...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load model (Qwen2.5-1.5B is small enough to fit without quantization)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✓ Qwen2.5 model loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text response.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            # Format messages (Qwen2 uses chat template)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)

            # Generate with KV cache enabled (works perfectly with Qwen2!)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # KV cache works with Qwen2!
                )

            # Decode output (skip input tokens)
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"


# Global singleton for Qwen2
_qwen2_llm = None


def get_qwen2_llm(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    torch_dtype: str = "float16",
) -> Qwen2TextLLM:
    """
    Get or create Qwen2.5 LLM singleton.

    Args:
        model_name: HuggingFace model name
        device: Target device ("cuda" or "cpu")
        torch_dtype: Data type for model weights

    Returns:
        Qwen2TextLLM instance
    """
    global _qwen2_llm

    if _qwen2_llm is None:
        _qwen2_llm = Qwen2TextLLM(model_name=model_name, device=device, torch_dtype=torch_dtype)

    return _qwen2_llm


class Phi35TextLLM:
    """
    Local Phi-3.5-3.8B-Instruct model for fast text generation and entity extraction.

    This is a text-only model optimized for:
    - Entity extraction during ingestion (much faster than API)
    - Response generation for queries
    - Reasoning about claims

    Benefits over API:
    - No rate limits
    - No network latency
    - No API costs
    - Works offline
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        max_new_tokens: int = 512,
        load_in_8bit: bool = True,  # Enable 8-bit quantization by default for 12GB GPU
    ):
        """
        Initialize Phi-3.5 model.

        Args:
            model_name: HuggingFace model name (default: Phi-3.5-mini-instruct)
            device: Target device ("cuda" or "cpu")
            torch_dtype: Data type for model weights
            max_new_tokens: Maximum tokens to generate
            load_in_8bit: Use 8-bit quantization (reduces VRAM from ~4.5GB to ~2.5GB)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit and device == "cuda"

        logger.info(
            f"Loading Phi-3.5 model {model_name} on {self.device} (8-bit: {self.load_in_8bit})..."
        )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load model with 8-bit quantization if enabled
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=self.load_in_8bit,
                trust_remote_code=True,
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✓ Phi-3.5 model loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text response.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            # Format messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disabled due to DynamicCache compatibility issue with Phi-3.5
                    # Note: Without KV cache, generation is slower but more stable
                )

            # Decode output (skip input tokens)
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return cast(str, response)

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"


# Global singleton for Phi-3.5
_phi35_llm = None


def get_phi35_llm(
    model_name: str = "microsoft/Phi-3.5-mini-instruct",
    device: str = "cuda",
    torch_dtype: str = "float16",
    load_in_8bit: bool = True,
) -> Phi35TextLLM:
    """
    Get or create Phi-3.5 LLM singleton.

    Args:
        model_name: HuggingFace model name
        device: Target device ("cuda" or "cpu")
        torch_dtype: Data type for model weights
        load_in_8bit: Use 8-bit quantization (recommended for 12GB GPU)

    Returns:
        Phi35TextLLM instance
    """
    global _phi35_llm

    if _phi35_llm is None:
        _phi35_llm = Phi35TextLLM(
            model_name=model_name, device=device, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit
        )

    return _phi35_llm
