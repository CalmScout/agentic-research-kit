"""Simple direct retriever for testing.

Provides basic retrieval from RAG-Anything without complex async issues.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def simple_retriever(query: str, top_k: int = 10) -> dict[str, Any]:
    """Simple retriever that directly accesses RAG-Anything storage.

    Args:
        query: User query
        top_k: Number of documents to retrieve

    Returns:
        Dict with retrieved_docs, retrieval_scores, retrieval_method
    """
    logger.info(f"Simple retriever: '{query[:50]}...' (top_k={top_k})")

    try:
        # Load documents from kv_store_full_docs.json
        with open("./rag_storage/kv_store_full_docs.json") as f:
            docs = json.load(f)

        # Load embeddings from vdb_chunks.json
        with open("./rag_storage/vdb_chunks.json") as f:
            vdb_data = json.load(f)
            vdb_data.get("matrix", [])
            vdb_data.get("embedding_dim", 2048)

        # Load entity chunks for knowledge graph traversal
        with open("./rag_storage/kv_store_entity_chunks.json") as f:
            json.load(f)

        # Get doc IDs
        doc_ids = list(docs.keys())

        logger.info(f"Loaded {len(doc_ids)} documents from storage")

        # Simple keyword matching for now (will be replaced by vector search)
        query_lower = query.lower()

        # Score each document by keyword matches
        scored_docs = []
        for doc_id in doc_ids:
            doc_content = docs[doc_id].get("content", "")
            doc_text = doc_content.lower()

            # Calculate keyword score
            words_in_query = set(query_lower.split())
            words_in_doc = set(doc_text.split())
            overlap = len(words_in_query & words_in_doc)
            score = overlap / max(len(words_in_query), 1)

            if score > 0:
                scored_docs.append(
                    {
                        "doc_id": doc_id,
                        "score": score,
                        "text": doc_content[:500],  # First 500 chars
                        "url": docs[doc_id].get("file_path", "Unknown"),
                        "entities": [],  # Could add entity extraction here
                    }
                )

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Take top_k documents
        retrieved_docs = scored_docs[:top_k]

        # Add retrieval scores
        retrieval_scores = [doc["score"] for doc in retrieved_docs]

        logger.info(f"Retrieved {len(retrieved_docs)} documents using keyword matching")

        return {
            "retrieved_docs": retrieved_docs,
            "retrieval_scores": retrieval_scores,
            "retrieval_method": "keyword",
        }

    except Exception as e:
        logger.error(f"Simple retriever failed: {e}")
        return {
            "retrieved_docs": [],
            "retrieval_scores": [],
            "retrieval_method": "keyword",
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        result = await simple_retriever("climate change", top_k=5)
        print("Retrieved docs:", len(result["retrieved_docs"]))
        print("Top result:", result["retrieved_docs"][0]["text"][:200])

    asyncio.run(test())
