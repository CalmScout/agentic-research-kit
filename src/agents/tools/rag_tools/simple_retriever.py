"""Simple retrieval tool.

Provides basic retrieval from RAG storage using keyword matching.
"""

import json
from typing import Any

from src.agents.tools.base import Tool
from src.agents.simple_retriever import simple_retriever

from loguru import logger


class SimpleRetrieverTool(Tool):
    """Tool for retrieving documents using simple keyword matching."""

    @property
    def name(self) -> str:
        return "simple_retriever"

    @property
    def description(self) -> str:
        return "Retrieve relevant documents using keyword matching"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve",
                    "default": 10
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, top_k: int = 10, **kwargs) -> str:
        """Retrieve documents for the given query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            JSON string with retrieved documents and metadata
        """
        try:
            result = await simple_retriever(query, top_k=top_k)

            # Return the full result as JSON
            return json.dumps({
                "retrieved_docs": result.get("retrieved_docs", []),
                "retrieval_scores": result.get("retrieval_scores", []),
                "retrieval_method": result.get("retrieval_method", "keyword"),
                "num_docs": len(result.get("retrieved_docs", []))
            })

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return json.dumps({
                "retrieved_docs": [],
                "retrieval_scores": [],
                "retrieval_method": "keyword",
                "num_docs": 0,
                "error": str(e)
            })
