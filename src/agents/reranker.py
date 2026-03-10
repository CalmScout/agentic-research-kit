"""Qwen3-VL Reranker wrapper for improving retrieval relevance.

Provides cross-modal reranking to improve the quality of retrieved documents.
Can be skipped if time-constrained (fallback: simple score-based ranking).
"""

import logging
from typing import Any

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
            logger.warning("Reranker model loading not implemented, using score-based fallback")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            # Fall back to score-based ranking
            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents based on query relevance using TEI Reranker API.

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

        try:
            import httpx
            import os
            
            # Using local TEI Reranker endpoint (standard port 8081)
            reranker_url = os.getenv("RERANKER_URL", "http://localhost:8081/rerank")
            
            # Fast check for reranker liveness
            try:
                # Prepare payload for TEI /rerank endpoint
                # Format: {"query": "...", "texts": ["...", "..."]}
                texts = [doc.get("text", "") for doc in documents]
                
                payload = {
                    "query": query,
                    "texts": texts,
                    "truncate": True
                }
                
                response = httpx.post(reranker_url, json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    results = response.json()
                    # TEI returns: [{"index": 0, "score": 0.99}, ...]
                    
                    # Merge scores back into documents
                    for res in results:
                        idx = res["index"]
                        score = res["score"]
                        documents[idx]["rerank_score"] = score
                    
                    # Sort by new rerank_score
                    reranked = sorted(documents, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                    
                    logger.info(f"✓ Reranked {len(documents)} docs via TEI API")
                    return reranked[:top_k]
                else:
                    logger.warning(f"Reranker API returned {response.status_code}: {response.text}")
                    raise RuntimeError("API failure")
                    
            except Exception as e:
                logger.warning(f"Reranker API not available: {e}. Falling back to score-based ranking.")
                # Fallback to score-based ranking below
                pass

            # Fallback: Sort by existing retrieval scores
            # Extract scores
            scored_docs = []
            for doc in documents:
                score = doc.get("score", 0.0)
                scored_docs.append((score, doc))

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Return top-k documents
            reranked = [doc for score, doc in scored_docs[:top_k]]

            logger.info(f"Reranked {len(documents)} docs → {len(reranked)} top docs (Score fallback)")
            return reranked

        except Exception as e:
            logger.error(f"Critical error in reranker: {e}")
            # Return original documents up to top_k
            return documents[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[float]]:
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
_reranker: Qwen3VLReranker | None = None


def get_reranker() -> Qwen3VLReranker:
    """Get singleton reranker instance.

    Returns:
        Qwen3VLReranker: Shared reranker instance
    """
    global _reranker
    if _reranker is None:
        _reranker = Qwen3VLReranker()
    return _reranker
