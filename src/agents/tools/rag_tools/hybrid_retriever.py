"""Hybrid retrieval tool using thread-isolated LightRAG.

This tool provides LightRAG hybrid retrieval (vector + BM25 + knowledge graph)
through thread-based isolation, avoiding async context conflicts with LangGraph.
"""

import json
import logging
from typing import Any

from src.agents.direct_lightrag_retriever import get_direct_lightrag_retriever
from src.agents.isolated_lightrag import create_isolated_lightrag
from src.agents.simple_retriever import simple_retriever

from ..base import Tool

logger = logging.getLogger(__name__)


class HybridRetrieverTool(Tool):
    """Hybrid retrieval with automatic fallback to keyword search.

    Uses thread-isolated LightRAG to avoid async context conflicts
    with LangGraph's event loop. Falls back to keyword search on errors.

    Thread isolation provides:
    - Zero API costs (uses local Qwen3-VL model)
    - Full hybrid search (vector + BM25 + knowledge graph)
    - Clean async context separation

    Example:
        >>> tool = HybridRetrieverTool()
        >>> result = await tool.execute(query="climate change", top_k=50)
        >>> data = json.loads(result)
        >>> print(data["retrieval_method"])  # "hybrid" or "keyword"
    """

    def __init__(self):
        """Initialize hybrid retriever tool."""
        self.isolated_rag = None
        self.direct_retriever = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "hybrid_retriever"

    @property
    def description(self) -> str:
        """Tool description for LLM function calling."""
        return (
            "Retrieve documents using LightRAG hybrid search "
            "(vector + BM25 + knowledge graph) with automatic keyword fallback. "
            "Uses thread isolation to avoid async conflicts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve",
                    "default": 50,
                },
                "mode": {
                    "type": "string",
                    "description": "Retrieval mode (naive, local, global, hybrid)",
                    "enum": ["naive", "local", "global", "hybrid"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        }

    def _initialize_isolated_rag(self) -> None:
        """Initialize thread-isolated LightRAG instance.

        This is done lazily on first use to avoid loading the embedding model
        until it's actually needed.
        """
        if self._initialized:
            return

        try:
            logger.info("Initializing thread-isolated LightRAG...")

            # CRITICAL: Use singleton retriever to avoid memory stacking and reloading models
            self.direct_retriever = get_direct_lightrag_retriever()

            # Use a factory function to initialize LightRAG within the worker thread.
            def rag_factory():
                return self.direct_retriever.get_rag()

            # Wrap in thread isolation using the singleton getter
            self.isolated_rag = create_isolated_lightrag(
                rag_factory,
                timeout=180.0,  # Increased timeout for complex queries
            )

            self._initialized = True
            logger.info("✓ Thread-isolated LightRAG initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize thread-isolated LightRAG: {e}", exc_info=True)
            # Don't set _initialized, allowing retry on next call
            raise

    async def execute(self, **kwargs: Any) -> str:
        """Execute hybrid retrieval with automatic fallback.

        Args:
            **kwargs: Must contain 'query', optional 'top_k', 'mode'

        Returns:
            JSON string with retrieved documents and metadata
        """
        query = kwargs.get("query", "")
        if not query:
            return json.dumps({"error": "Missing required parameter 'query'"}, ensure_ascii=False)
        top_k = kwargs.get("top_k", 50)
        mode = kwargs.get("mode", "hybrid")

        try:
            # Lazy initialization on first use
            if not self._initialized:
                self._initialize_isolated_rag()

            # Use thread-isolated hybrid retrieval
            logger.info(f"Querying LightRAG (mode={mode}, top_k={top_k}): '{query[:50]}...'")

            # Call aquery_sync which runs in isolated thread
            # and uses aquery_data for structured results when only_need_context=True
            result = self.isolated_rag.aquery_sync(
                query, mode=mode, only_need_context=True  # naive, local, global, hybrid
            )

            # Parse LightRAG response from aquery_data
            # Returns: {"status": "success", "data": {"chunks": [...], "entities": [...], "relationships": [...]}}
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected LightRAG response type: {type(result)}")

            data = result.get("data", {})
            chunks = data.get("chunks", [])

            if not chunks:
                # Fallback to checking if it returned a string with context (old behavior)
                context_text = result.get("context", "")
                if isinstance(context_text, str) and context_text:
                    # Parse context_text split by double newlines
                    chunks = [
                        {"content": c.strip()} for c in context_text.split("\n\n") if c.strip()
                    ]

            # If still no chunks, trigger the fallback mechanism gracefully
            if not chunks:
                logger.info("LightRAG returned no results, falling back to keyword search")
                result_dict = await simple_retriever(query, top_k)
                return json.dumps(result_dict)

            # Format chunks into documents
            documents = []
            scores = []

            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")
                if content:
                    # Calculate score: earlier chunks get higher scores if not provided
                    score = chunk.get("score", 1.0 - (i * 0.01))
                    if score < 0.1:
                        score = 0.1

                    documents.append(
                        {
                            "text": content,
                            "score": score,
                            "metadata": {
                                "source": "lightrag_hybrid",
                                "mode": "hybrid",
                                "chunk_index": i,
                                "file_path": chunk.get("file_path", ""),
                                "chunk_id": chunk.get("chunk_id", ""),
                            },
                        }
                    )
                    scores.append(score)

            # Limit to top_k documents
            documents = documents[:top_k]
            scores = scores[:top_k]

            logger.info(f"✓ Hybrid retrieval successful: {len(documents)} docs retrieved")

            return json.dumps(
                {
                    "retrieved_docs": documents,
                    "retrieval_scores": scores,
                    "retrieval_method": "hybrid",
                }
            )

        except Exception as e:
            logger.debug(f"Hybrid retrieval encountered an issue: {e}")
            logger.info("Falling back to keyword search...")

            # Graceful fallback to keyword search
            try:
                result_dict = await simple_retriever(query, top_k)
                # Ensure we return a JSON string
                return json.dumps(result_dict)
            except Exception as fallback_error:
                logger.error(f"Keyword fallback also failed: {fallback_error}")
                raise RuntimeError(
                    f"Both hybrid and keyword retrieval failed. "
                    f"Hybrid error: {e}, Keyword error: {fallback_error}"
                ) from fallback_error

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate tool parameters.

        Args:
            params: Parameter dict to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if "query" not in params:
            errors.append("Missing required parameter: 'query'")

        if "query" in params:
            query = params["query"]
            if not isinstance(query, str):
                errors.append("Parameter 'query' must be a string")
            elif not query.strip():
                errors.append("Parameter 'query' cannot be empty")

        if "top_k" in params:
            top_k = params["top_k"]
            if not isinstance(top_k, int):
                errors.append("Parameter 'top_k' must be an integer")
            elif top_k < 1:
                errors.append("Parameter 'top_k' must be at least 1")
            elif top_k > 200:
                errors.append("Parameter 'top_k' cannot exceed 200")

        return errors

    async def close(self) -> None:
        """Close tool and clean up resources."""
        # We NO LONGER close the isolated_rag here because it is a shared singleton
        # required for subsequent iterations of the ReAct loop.
        if self.isolated_rag:
            logger.debug("Releasing HybridRetrieverTool (keeping background RAG warm)")
            self.isolated_rag = None
            self._initialized = False
