"""Embedding service for generating multimodal vector representations.

Uses GPU-accelerated Qwen3-VL-Embedding-2B for unified text/image vector space.
"""

import logging
from typing import cast

from src.utils.config import get_settings
from src.utils.vision_embedding import get_embedding_model

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Multimodal embedding service using local Qwen3-VL-Embedding-2B."""

    def __init__(self, device: str = "cuda"):
        """Initialize embedding service.

        Args:
            device: Target device ("cuda" or "cpu")
        """
        self.settings = get_settings()
        self.device = device
        self._model = None

    def _get_model(self):
        """Lazy load embedding model singleton."""
        if self._model is None:
            self._model = get_embedding_model(
                model_name=self.settings.embedding_model, device=self.device
            )
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a text string.

        Args:
            text: Input text

        Returns:
            List[float]: 2048D embedding vector
        """
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode

            tracer = trace.get_tracer(__name__)
            # Use OpenInference semantic conventions for Phoenix
            with tracer.start_as_current_span(
                "embed_text",
                attributes={
                    "input.value": text[:1000] + "..." if len(text) > 1000 else text,
                    "input.mime_type": "text/plain",
                    "openinference.span.kind": "EMBEDDING",
                    "embedding.model_name": self.settings.embedding_model,
                },
            ) as span:
                model = self._get_model()
                embedding = model.embed_text(text)
                result = cast(list[float], embedding.tolist())

                # We don't log the full vector to avoid overhead, just dimensionality
                span.set_attribute("output.value", f"vector(dim={len(result)})")
                span.set_status(StatusCode.OK)
                return result
        except ImportError:
            model = self._get_model()
            embedding = model.embed_text(text)
            return cast(list[float], embedding.tolist())
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            try:
                from opentelemetry.trace import StatusCode

                span = trace.get_current_span()
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
            except Exception:
                pass
            raise

    def embed_image(self, image_path: str) -> list[float]:
        """Generate vector embedding for an image.

        Args:
            image_path: Path to image file

        Returns:
            List[float]: 2048D embedding vector
        """
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "embed_image",
                attributes={
                    "input.value": image_path,
                    "openinference.span.kind": "EMBEDDING",
                    "embedding.model_name": self.settings.embedding_model,
                },
            ) as span:
                model = self._get_model()
                embedding = model.embed_image(image_path)
                result = cast(list[float], embedding.tolist())

                span.set_attribute("output.value", f"vector(dim={len(result)})")
                span.set_status(StatusCode.OK)
                return result
        except ImportError:
            model = self._get_model()
            embedding = model.embed_image(image_path)
            return cast(list[float], embedding.tolist())
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            try:
                from opentelemetry.trace import StatusCode

                span = trace.get_current_span()
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
            except Exception:
                pass
            raise

    def embed_multimodal(self, text: str, image_path: str | None = None) -> list[float]:
        """Generate joint text+image embedding.

        Args:
            text: Text component
            image_path: Optional image path

        Returns:
            List[float]: Joint embedding vector
        """
        if not image_path:
            return self.embed_text(text)

        # Qwen3-VL-Embedding-2B supports joint inputs
        # For now we use image embedding as primary for multimodal queries
        # (Alternatively, average text + image vectors)
        image_vector = self.embed_image(image_path)
        text_vector = self.embed_text(text)

        # Average fusion for simple multimodal representation
        import numpy as np

        avg_vector = (np.array(image_vector) + np.array(text_vector)) / 2.0
        return cast(list[float], avg_vector.tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "embed_batch",
                attributes={
                    "input.value": f"batch_size={len(texts)}",
                    "openinference.span.kind": "EMBEDDING",
                    "embedding.model_name": self.settings.embedding_model,
                },
            ) as span:
                model = self._get_model()
                # Handle model batching if supported
                if hasattr(model, "embed_text_batch"):
                    embeddings = model.embed_text_batch(texts)
                    result = cast(list[list[float]], embeddings.tolist())
                    span.set_attribute("output.value", f"batch_vectors(count={len(result)})")
                    span.set_status(StatusCode.OK)
                    return result
                else:
                    embeddings = [self.embed_text(text) for text in texts]
                    result = cast(list[list[float]], embeddings)
                    span.set_attribute("output.value", f"batch_vectors(count={len(result)})")
                    span.set_status(StatusCode.OK)
                    return result
        except ImportError:
            model = self._get_model()
            if hasattr(model, "embed_text_batch"):
                embeddings = model.embed_text_batch(texts)
                return cast(list[list[float]], embeddings.tolist())
            else:
                embeddings = [self.embed_text(text) for text in texts]
                return cast(list[list[float]], embeddings)
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            try:
                from opentelemetry.trace import StatusCode

                span = trace.get_current_span()
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
            except Exception:
                pass
            raise


# Singleton instance
embedder = EmbeddingService()
