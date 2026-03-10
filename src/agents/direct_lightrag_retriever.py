"""Direct LightRAG retriever using API embedding models.

This module provides direct access to LightRAG without using the HTTP server.
It uses API endpoints (TEI and vLLM) for embedding and LLM operations.
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
from src.agents.model_selector import ThinkingProcessStripper
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

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

# --- Standalone Wrapper Functions for LightRAG (Avoids deepcopy/pickle issues) ---

# Global singletons for the wrappers to avoid recreation overhead
_wrapper_llm = None
_wrapper_embeddings = None

async def direct_hf_llm_wrapper(prompt: str, system_prompt: str | None = None, **kwargs) -> str:
    """Standalone LLM wrapper for LightRAG using singleton client and json_repair."""
    global _wrapper_llm
    if _wrapper_llm is None:
        model_name = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen3.5-4B")
        _wrapper_llm = ThinkingProcessStripper(
            model=model_name,
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
            temperature=0.0,
            timeout=120.0,
            max_tokens=1024 
        )
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    res = await _wrapper_llm.ainvoke(messages)
    content = str(res.content)
    
    # If the prompt looks like it's asking for JSON (common in LightRAG), 
    # use json_repair to ensure it's clean for their internal parser.
    if any(keyword in prompt.lower() for keyword in ["json", "format:", "{"]):
        try:
            import json_repair
            return json_repair.repair_json(content)
        except ImportError:
            pass
            
    return content

async def direct_hf_embedding_wrapper(texts: list[str], **kwargs) -> np.ndarray:
    """Standalone embedding wrapper for LightRAG using singleton client."""
    global _wrapper_embeddings
    if _wrapper_embeddings is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        _wrapper_embeddings = OpenAIEmbeddings(
            model=model_name,
            base_url="http://localhost:8082/v1",
            api_key="EMPTY",
            timeout=120.0,
        )
    
    res = await _wrapper_embeddings.aembed_documents(texts)
    return np.array(res, dtype=np.float32)

# --- End Wrapper Functions ---

class DirectLightRAGRetriever:
    """Direct LightRAG retriever using API embedding models.

    This avoids the HTTP server and works entirely with local API models.
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
        self._embedding_dim: int = 1024  # Default for BGE large

    def get_rag(self) -> LightRAG:
        """Get or create LightRAG instance."""
        if self._rag is None:
            logger.info(
                f"Initializing LightRAG with API models (working_dir: {self.working_dir})"
            )

            # Detect dimension if not already set (one-time check)
            if self._embedding_dim == 1024:
                try:
                    import httpx
                    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
                    base_url = os.getenv("EMBEDDING_API_URL", "http://localhost:8082/v1")
                    logger.debug(f"Detecting embedding dimension for {model_name}...")
                    
                    # Try info endpoint or test query
                    test_embeddings = OpenAIEmbeddings(
                        model=model_name,
                        base_url=base_url,
                        api_key="EMPTY",
                        timeout=10.0
                    )
                    test_vec = test_embeddings.embed_query("test")
                    self._embedding_dim = len(test_vec)
                    logger.info(f"✓ Detected LightRAG embedding dimension: {self._embedding_dim}")
                except Exception as e:
                    logger.warning(f"Could not detect embedding dimension for LightRAG: {e}. Defaulting to 1024.")
                    self._embedding_dim = 1024

            # Create embedding function using standalone wrapper.
            embedding_func = EmbeddingFunc(
                embedding_dim=self._embedding_dim,
                max_token_size=8192,
                func=direct_hf_embedding_wrapper,
            )

            # Initialize LightRAG with standalone wrappers
            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=direct_hf_llm_wrapper,
                embedding_func=embedding_func,
                kv_storage="LanceDBKVStorage",
                doc_status_storage="LanceDBDocStatusStorage",
                vector_storage="LanceDBVectorDBStorage",
                embedding_batch_num=1,
                embedding_func_max_async=1,
                llm_model_max_async=1,  # CRITICAL: Force exactly 1 LLM task at a time
            )

            logger.info("✓ LightRAG initialized successfully")

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
