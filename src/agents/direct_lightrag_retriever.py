"""Direct LightRAG retriever using local embedding models.

This module provides direct access to LightRAG without using the HTTP server.
It uses local Qwen3-VL embedding models for query embedding and retrieval.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES
from lightrag.utils import EmbeddingFunc, lazy_external_import

from src.agents.lancedb_storage import (
    LanceDBDocStatusStorage,
    LanceDBKVStorage,
    LanceDBVectorDBStorage,
)
from src.utils.vision_embedding import Qwen3VLEmbedding, get_qwen2_llm

# Register custom LanceDB storages names
STORAGES["LanceDBKVStorage"] = "LanceDBKVStorage"
STORAGES["LanceDBDocStatusStorage"] = "LanceDBDocStatusStorage"
STORAGES["LanceDBVectorDBStorage"] = "LanceDBVectorDBStorage"

# Add to allowed implementations to bypass validation errors
STORAGE_IMPLEMENTATIONS["KV_STORAGE"]["implementations"].append("LanceDBKVStorage")
STORAGE_IMPLEMENTATIONS["DOC_STATUS_STORAGE"]["implementations"].append("LanceDBDocStatusStorage")
STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"].append("LanceDBVectorDBStorage")

# Monkey-patch LightRAG to support our custom classes directly
_original_get_storage_class = LightRAG._get_storage_class


def _patched_get_storage_class(self, storage_name: str) -> Any:
    if storage_name == "LanceDBKVStorage":
        return LanceDBKVStorage
    elif storage_name == "LanceDBDocStatusStorage":
        return LanceDBDocStatusStorage
    elif storage_name == "LanceDBVectorDBStorage":
        return LanceDBVectorDBStorage

    # Handle standard LightRAG storages or others
    try:
        return cast(Callable[..., Any], _original_get_storage_class(self, storage_name))
    except Exception:
        # Fallback for dynamic import if not handled by original
        if storage_name in STORAGES:
            import_path = STORAGES[storage_name]
            if not isinstance(import_path, str):
                return import_path
            return cast(Callable[..., Any], lazy_external_import(import_path, storage_name))
        raise


LightRAG._get_storage_class = _patched_get_storage_class

logger = logging.getLogger(__name__)


class DirectLightRAGRetriever:
    """Direct LightRAG retriever using local embedding models.

    This avoids the HTTP server and works entirely with local models.
    """

    def __init__(self, working_dir: str = "./rag_storage", device: str = "cuda"):
        """Initialize direct LightRAG retriever.

        Args:
            working_dir: Path to LightRAG storage directory
            device: Device to use for embeddings ("cuda" or "cpu")
        """
        self.working_dir = Path(working_dir)
        self.device = device
        self._rag: LightRAG | None = None
        self._embedding_model: Qwen3VLEmbedding | None = None

    def _get_embedding_model(self) -> Qwen3VLEmbedding:
        """Get or create embedding model instance (lazy loading)."""
        if self._embedding_model is None:
            logger.info("Loading local embedding model: Qwen/Qwen3-VL-Embedding-2B")
            self._embedding_model = Qwen3VLEmbedding(
                model_name="Qwen/Qwen3-VL-Embedding-2B",
                device=self.device,
                torch_dtype="float16",  # Use float16 instead of "auto"
            )
            # Test embedding to get dimension
            test_emb = self._embedding_model.embed_text("test")
            logger.info(f"✓ Embedding model loaded (dimension: {test_emb.shape[0]})")

        return self._embedding_model

    async def _embed_with_local_model(self, texts: list) -> np.ndarray:
        """Generate embeddings using local Qwen3-VL model.

        Args:
            texts: List of text strings to embed

        Returns:
            2D Numpy array with shape (num_texts, embedding_dim)
        """
        embedding_model = self._get_embedding_model()
        
        # Optimization: LightRAG's internal workers can cause multiple accesses
        # We ensure thread-safe single-threaded execution here
        embeddings = []

        for text in texts:
            try:
                # Direct call to singleton which is already instrumented for Phoenix
                embedding = embedding_model.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Embedding error for text '{text[:50]}...': {e}")
                # Return zero vector on error
                embeddings.append(np.zeros(2048, dtype=np.float32))

        # Convert list of 1D arrays to 2D numpy array
        return np.vstack(embeddings).astype(np.float32)

    def _create_llm_model_func(self) -> Callable:
        """Create async LLM function compatible with LightRAG.

        Returns an async function that LightRAG can call for entity extraction
        and knowledge graph operations during queries.
        """

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[Any] | None = None,
            **kwargs: Any,
        ) -> str:
            """LLM function for LightRAG entity extraction."""
            try:
                # Lazy load LLM on first call (uses singleton from vision_embedding.py)
                if not hasattr(self, "_llm"):
                    logger.info("Initializing Qwen2.5-1.5B for LightRAG LLM operations...")
                    self._llm = get_qwen2_llm(
                        model_name="Qwen/Qwen2.5-1.5B-Instruct", device=self.device
                    )
                    logger.info("✓ LightRAG LLM initialized")

                # Generate response using sync call in async context
                response = self._llm.generate(
                    prompt=prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=512
                )
                return response

            except Exception as e:
                logger.error(f"LLM function error: {e}", exc_info=True)
                return ""

        return llm_model_func

    def get_rag(self) -> LightRAG:
        """Get or create LightRAG instance.

        Returns:
            LightRAG instance configured with local embedding model
        """
        if self._rag is None:
            logger.info(
                f"Initializing LightRAG with local models (working_dir: {self.working_dir})"
            )

            # CRITICAL: Force LightRAG to use a single background worker for embeddings
            # This prevents memory explosion when multiple entities are embedded
            os.environ["EMBEDDING_FUNC_MAX_ASYNC"] = "1"

            # Create embedding function. 
            embedding_func = EmbeddingFunc(
                embedding_dim=2048,  # Qwen3-VL-Embedding-2B dimension
                max_token_size=8192,
                func=self._embed_with_local_model,
            )

            # Create LLM function for entity extraction
            llm_model_func = self._create_llm_model_func()

            # Initialize LightRAG
            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                kv_storage="LanceDBKVStorage",
                doc_status_storage="LanceDBDocStatusStorage",
                vector_storage="LanceDBVectorDBStorage",
                embedding_batch_num=1,  # Force sequential embedding to save RAM
                embedding_func_max_async=1,  # CRITICAL: Fix for the "8 workers" issue
            )

            logger.info("✓ LightRAG initialized successfully (EMBEDDING_FUNC_MAX_ASYNC=1)")

        return self._rag

    async def retrieve(self, query: str, top_k: int = 50, mode: str = "naive") -> dict[str, Any]:
        """Retrieve documents using LightRAG."""
        rag = self.get_rag()

        logger.info(f"Querying LightRAG directly (mode={mode}, top_k={top_k})")

        try:
            result = await rag.aquery_data(query, param=QueryParam(mode=mode))

            documents = []
            scores = []

            if isinstance(result, dict):
                data = result.get("data", {})
                chunks = data.get("chunks", [])

                for i, item in enumerate(chunks):
                    if isinstance(item, dict):
                        doc = {
                            "text": item.get("content", item.get("text", str(item))),
                            "score": item.get("score", 1.0 - (i * 0.01)),
                            "metadata": {
                                "source": "lightrag_direct",
                                "mode": mode,
                                "file_path": item.get("file_path", ""),
                                "chunk_id": item.get("chunk_id", ""),
                            },
                        }
                        documents.append(doc)
                        scores.append(doc["score"])

            logger.info(
                f"✓ Retrieved {len(documents)} documents from LightRAG (direct, mode={mode})"
            )

            return {
                "retrieved_docs": documents,
                "retrieval_scores": scores,
                "retrieval_method": f"hybrid_{mode}",
            }

        except Exception as e:
            logger.error(f"Direct LightRAG retrieval failed: {e}", exc_info=True)
            return cast(dict[str, Any], {"error": str(e)})


# Singleton instance
_retriever: DirectLightRAGRetriever | None = None


def get_direct_lightrag_retriever() -> DirectLightRAGRetriever:
    """Get singleton direct LightRAG retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = DirectLightRAGRetriever()
    return _retriever
