"""Qwen3-VL embedding wrapper for multi-agent system.

Provides singleton access to Qwen3-VL-Embedding-2B for efficient memory usage.
Reuses the existing Qwen3VLEmbedding from src.utils.vision_embedding.
"""

from typing import List, Union, Optional
from loguru import logger


# Singleton instance for memory efficiency
_embedding_model = None


def get_embedding_model():
    """Get singleton embedding model instance.

    Returns:
        Qwen3VLEmbedding: Shared embedding model instance
    """
    global _embedding_model

    if _embedding_model is None:
        from src.utils.vision_embedding import Qwen3VLEmbedding

        logger.info("Initializing Qwen3-VL-Embedding-2B model...")
        _embedding_model = Qwen3VLEmbedding(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            device="cuda",
            torch_dtype="float16",
        )
        logger.info("✓ Embedding model initialized")

    return _embedding_model


class Qwen3VLEmbeddingWrapper:
    """Wrapper for Qwen3-VL-Embedding-2B with singleton pattern.

    This wrapper provides a clean interface for agents to generate embeddings
    while ensuring the model is loaded only once in memory.
    """

    def __init__(self):
        """Initialize wrapper (lazy loads model on first use)."""
        self._model = None

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text query.

        Args:
            text: Text query to embed

        Returns:
            List[float]: 2048-dimensional embedding vector
        """
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("embed_text") as span:
                span.set_attribute("text_length", len(text))
                embedding = self.model.embed_text(text)
                logger.debug(f"Generated text embedding (dim={len(embedding)})")
                return embedding
        except ImportError:
            embedding = self.model.embed_text(text)
            logger.debug(f"Generated text embedding (dim={len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for image.

        Args:
            image_path: Path to image file

        Returns:
            List[float]: 2048-dimensional embedding vector
        """
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("embed_image") as span:
                span.set_attribute("image_path", image_path)
                embedding = self.model.embed_image(image_path)
                logger.debug(f"Generated image embedding (dim={len(embedding)})")
                return embedding
        except ImportError:
            embedding = self.model.embed_image(image_path)
            logger.debug(f"Generated image embedding (dim={len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed image: {e}")
            raise

    def embed_multimodal(
        self, text: str, image_path: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for text + image (multimodal).

        Args:
            text: Text query
            image_path: Optional image path

        Returns:
            List[float]: 2048-dimensional embedding vector
        """
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("embed_multimodal") as span:
                span.set_attribute("text_length", len(text))
                if image_path:
                    span.set_attribute("image_path", image_path)
                    embedding = self.model.embed_multimodal(text, image_path)
                    logger.debug(f"Generated multimodal embedding (dim={len(embedding)})")
                else:
                    embedding = self.embed_text(text)
                return embedding
        except ImportError:
            if image_path:
                embedding = self.model.embed_multimodal(text, image_path)
                logger.debug(f"Generated multimodal embedding (dim={len(embedding)})")
            else:
                embedding = self.embed_text(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed multimodal: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("embed_batch") as span:
                span.set_attribute("batch_size", len(texts))
                embeddings = [self.embed_text(text) for text in texts]
                logger.debug(f"Generated batch embeddings (count={len(texts)})")
                return embeddings
        except ImportError:
            embeddings = [self.embed_text(text) for text in texts]
            logger.debug(f"Generated batch embeddings (count={len(texts)})")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise


# Convenience singleton instance
embedder = Qwen3VLEmbeddingWrapper()
