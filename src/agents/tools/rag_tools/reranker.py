"""Reranker tool for improving retrieval relevance.

Provides reranking of retrieved documents using score-based or model-based reranking.
"""

import json
from typing import Any, List

from src.agents.tools.base import Tool
from src.agents.reranker import get_reranker

from loguru import logger


class RerankerTool(Tool):
    """Tool for reranking retrieved documents by relevance."""

    @property
    def name(self) -> str:
        return "reranker"

    @property
    def description(self) -> str:
        return "Rerank retrieved documents to improve relevance (top 50 → top 10)"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "docs": {
                    "type": "array",
                    "description": "List of retrieved documents to rerank",
                    "items": {"type": "object"}
                },
                "query": {
                    "type": "string",
                    "description": "Original query for relevance scoring"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top documents to return",
                    "default": 10
                }
            },
            "required": ["docs", "query"]
        }

    async def execute(self, docs: List[dict], query: str, top_k: int = 10, **kwargs) -> str:
        """Rerank documents by relevance to the query.

        Args:
            docs: List of documents to rerank
            query: Original query
            top_k: Number of top documents to return

        Returns:
            JSON string with reranked documents
        """
        try:
            # Try to use the reranker if available
            reranker = get_reranker()

            if reranker is None:
                # Fallback: simple score-based reranking
                logger.debug("Reranker not available, using score-based fallback")

                # Sort by existing score
                sorted_docs = sorted(
                    docs,
                    key=lambda x: x.get("score", 0.0),
                    reverse=True
                )
                reranked = sorted_docs[:top_k]

            else:
                # Use Qwen3-VL reranker (when available)
                # For now, use the same fallback as above
                sorted_docs = sorted(
                    docs,
                    key=lambda x: x.get("score", 0.0),
                    reverse=True
                )
                reranked = sorted_docs[:top_k]

            # Extract scores
            reranked_scores = [doc.get("score", 0.0) for doc in reranked]

            logger.info(f"Reranked {len(docs)} → {len(reranked)} documents")

            return json.dumps({
                "reranked_docs": reranked,
                "reranked_scores": reranked_scores,
                "num_docs": len(reranked)
            })

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original docs
            return json.dumps({
                "reranked_docs": docs[:top_k],
                "reranked_scores": [d.get("score", 0.0) for d in docs[:top_k]],
                "num_docs": min(len(docs), top_k),
                "error": str(e)
            })
