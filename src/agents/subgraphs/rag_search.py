"""
RAG Search Subgraph for specialized internal knowledge retrieval.
Part of the hardware-aware fractal subagent pattern.
"""

import json
from typing import Any, TypedDict

from src.agents.tools.rag_tools.hybrid_retriever import HybridRetrieverTool
from src.utils.config import get_settings
from src.utils.logger import logger


class RAGSearchState(TypedDict):
    """Internal state for the RAG search subgraph."""

    query: str
    retrieval_mode: str
    results: list[dict]
    method: str
    error: str | None


async def rag_search_node(state: dict) -> dict[str, Any]:
    """
    Sub-node that performs hybrid RAG retrieval (Vector + KG).
    """
    query = state["query"]
    mode = state.get("retrieval_mode", "hybrid")
    existing_docs = state.get("retrieved_docs", [])
    settings = get_settings()

    logger.info(f"RAGSearch Subgraph: Querying internal knowledge (mode={mode})")

    retriever = HybridRetrieverTool()

    try:
        retrieval_result = await retriever.execute(
            query=query, top_k=settings.retrieval_top_k, mode=mode
        )

        data = json.loads(retrieval_result)
        retrieved_docs = data.get("retrieved_docs", [])
        method = data.get("retrieval_method", mode)

        # Format for consistency
        new_results = []
        for doc in retrieved_docs:
            new_results.append(
                {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.5),
                }
            )

        logger.info(f"RAGSearch Subgraph: Retrieved {len(new_results)} documents")

        return {"retrieved_docs": existing_docs + new_results, "retrieval_method": method}

    except Exception as e:
        logger.error(f"RAGSearch Subgraph failed: {e}")
        return {"retrieved_docs": existing_docs, "retrieval_method": "error"}
