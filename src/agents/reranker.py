"""Qwen3-VL Reranker wrapper for improving retrieval relevance.

Provides cross-modal reranking to improve the quality of retrieved documents.
Can be skipped if time-constrained (fallback: simple score-based ranking).
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Qwen3VLReranker:
    """Reranker using Qwen3-VL-Reranker-2B for cross-modal ranking.

    This reranker improves retrieval quality by:
    - Cross-modal scoring (text query vs text+image documents)
    - Relevance-based reordering
    - Top-K selection

    Note: This is optional. If reranker fails, system falls back to
    score-based ranking from LightRAG.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        device: str = "cuda",
        top_k: int = 10,
    ):
        """Initialize reranker.

        Args:
            model_name: HuggingFace model name
            device: Target device ("cuda" or "cpu")
            top_k: Default number of top documents to return
        """
        self.model_name = model_name
        self.device = device if device == "cuda" else "cpu"
        self.top_k = top_k
        self._model = None
        self._initialized = False

    def _load_model(self):
        """Lazy load reranker model."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading {self.model_name}...")
            # TODO: Implement actual model loading
            # For now, we'll use a simpler approach:
            # Just re-sort based on existing scores
            logger.warning(
                "Reranker model loading not implemented, using score-based fallback"
            )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            # Fall back to score-based ranking
            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.

        Args:
            query: User's query
            documents: List of retrieved documents with metadata
            top_k: Number of top documents to return

        Returns:
            List[Dict]: Reranked documents (top-k)
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        top_k = top_k or self.top_k

        # Ensure model is loaded
        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"Reranker failed, using original order: {e}")
            return documents[:top_k]

        # Fallback: Sort by existing retrieval scores
        try:
            # Extract scores
            scored_docs = []
            for doc in documents:
                score = doc.get("score", 0.0)
                scored_docs.append((score, doc))

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Return top-k documents
            reranked = [doc for score, doc in scored_docs[:top_k]]

            logger.info(f"Reranked {len(documents)} docs → {len(reranked)} top docs")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original documents up to top_k
            return documents[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Rerank documents and return both docs and scores.

        Args:
            query: User's query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Tuple[List[Dict], List[float]]: (reranked docs, scores)
        """
        reranked_docs = self.rerank(query, documents, top_k)

        # Extract scores
        scores = [doc.get("score", 0.0) for doc in reranked_docs]

        return reranked_docs, scores


# Singleton instance
_reranker: Optional[Qwen3VLReranker] = None


def get_reranker() -> Qwen3VLReranker:
    """Get singleton reranker instance.

    Returns:
        Qwen3VLReranker: Shared reranker instance
    """
    global _reranker
    if _reranker is None:
        _reranker = Qwen3VLReranker()
    return _reranker
