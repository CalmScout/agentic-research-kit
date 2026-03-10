"""Embedding service for generating multimodal vector representations.

Uses GPU-accelerated BAAI/bge-large-en-v1.5 for text vector space via TEI.
"""

import logging
import os

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Multimodal embedding service using local TEI."""

    def __init__(self, device: str = "cuda"):
        """Initialize embedding service.

        Args:
            device: Target device ("cuda" or "cpu")
        """
        self.settings = get_settings()
        self.device = device
        self._model: OpenAIEmbeddings | None = None
        self._dim = 1024  # Default for BAAI/bge-large-en-v1.5

    def _get_model(self) -> OpenAIEmbeddings:
        """Lazy load embedding model singleton pointing to TEI."""
        if self._model is None:
            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
            base_url = os.getenv("EMBEDDING_API_URL", "http://localhost:8082/v1")

            self._model = OpenAIEmbeddings(
                model=model_name,
                base_url=base_url,
                api_key=SecretStr("EMPTY"),
                timeout=120.0,
            )

            # Detect actual dimension by asking the API
            try:
                import httpx

                # TEI info endpoint is at the base URL (without /v1)
                info_url = base_url.replace("/v1", "") + "/info"
                response = httpx.get(info_url, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    # TEI returns model info including dimension
                    self._dim = data.get("max_sequence_length", 1024)  # Placeholder if not found
                    # Better: actually use a test query if /info doesn't have it
                    test_vec = self._model.embed_query("test")
                    self._dim = len(test_vec)
                    logger.info(f"✓ Detected embedding dimension from {model_name}: {self._dim}")
                else:
                    test_vec = self._model.embed_query("test")
                    self._dim = len(test_vec)
                    logger.info(f"✓ Detected embedding dimension via test query: {self._dim}")
            except Exception as e:
                logger.warning(f"Could not detect embedding dimension: {e}. Defaulting to 1024.")
                self._dim = 1024

        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # Ensure model is initialized to detect dimension
        if self._model is None:
            self._get_model()
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a text string."""
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
                    "embedding.model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                },
            ) as span:
                model = self._get_model()
                # embed_query is synchronous
                try:
                    result = model.embed_query(text)
                except AttributeError as e:
                    if "'str' object has no attribute 'data'" in str(e):
                        raise ConnectionError(
                            "Failed to connect to TEI Embedding API at http://localhost:8082/v1. "
                            "Ensure the 'tei' docker container is running."
                        ) from e
                    raise
                except Exception as e:
                    if "Connection" in str(e) or "ConnectError" in str(type(e)):
                        raise ConnectionError(
                            "Failed to connect to TEI Embedding API at http://localhost:8082/v1. "
                            "Ensure the 'tei' docker container is running."
                        ) from e
                    raise

                span.set_attribute("output.value", f"vector(dim={len(result)})")
                span.set_status(StatusCode.OK)
                return result
        except ImportError:
            model = self._get_model()
            return model.embed_query(text)
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

        Currently a fallback placeholder as standard TEI text endpoint doesn't natively support images.
        """
        # Ensure model is initialized to have _dim
        self._get_model()
        logger.warning(
            "Image embedding via TEI text model not natively supported. Returning zero vector."
        )
        return [0.0] * self._dim

    def embed_multimodal(self, text: str, image_path: str | None = None) -> list[float]:
        """Generate joint text+image embedding."""
        if not image_path:
            return self.embed_text(text)

        # In a real multimodal TEI setup, we would send both. For now fallback to text.
        return self.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "embed_batch",
                attributes={
                    "input.value": f"batch_size={len(texts)}",
                    "openinference.span.kind": "EMBEDDING",
                    "embedding.model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                },
            ) as span:
                model = self._get_model()
                try:
                    result = model.embed_documents(texts)
                except AttributeError as e:
                    if "'str' object has no attribute 'data'" in str(e):
                        raise ConnectionError(
                            "Failed to connect to TEI Embedding API at http://localhost:8082/v1. "
                            "Ensure the 'tei' docker container is running."
                        ) from e
                    raise
                except Exception as e:
                    if "Connection" in str(e) or "ConnectError" in str(type(e)):
                        raise ConnectionError(
                            "Failed to connect to TEI Embedding API at http://localhost:8082/v1. "
                            "Ensure the 'tei' docker container is running."
                        ) from e
                    raise
                span.set_attribute("output.value", f"batch_vectors(count={len(result)})")
                span.set_status(StatusCode.OK)
                return result
        except ImportError:
            model = self._get_model()
            return model.embed_documents(texts)
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
