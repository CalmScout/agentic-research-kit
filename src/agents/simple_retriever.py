"""Simple direct retriever for testing.

Provides basic retrieval from RAG-Anything without complex async issues.
"""

import json
import logging
import os
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
        # Check if files exist before trying to load
        full_docs_path = "./rag_storage/kv_store_full_docs.json"
        if not os.path.exists(full_docs_path):
            logger.warning(f"Storage file {full_docs_path} not found. Returning empty results.")
            return {"retrieved_docs": [], "retrieval_scores": [], "retrieval_method": "keyword"}

        with open(full_docs_path) as f:
            docs = json.load(f)

        # Load embeddings from vdb_chunks.json if it exists
        vdb_path = "./rag_storage/vdb_chunks.json"
        if os.path.exists(vdb_path):
            with open(vdb_path) as f:
                vdb_data = json.load(f)
                vdb_data.get("matrix", [])
                from src.agents.embeddings import embedder
                vdb_data.get("embedding_dim", embedder._dim)

        # Load entity chunks for knowledge graph traversal
        with open("./rag_storage/kv_store_entity_chunks.json") as f:
            json.load(f)

        # Get doc IDs
        doc_ids = list(docs.keys())

        logger.info(f"Loaded {len(doc_ids)} documents from storage")

        # Simple keyword matching for now (will be replaced by vector search)
        query_lower = query.lower()

        # Basic stop words to filter out
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "at",
            "by",
            "from",
            "for",
            "with",
            "in",
            "on",
            "to",
            "of",
            "up",
            "out",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "shall",
            "will",
            "should",
            "would",
            "may",
            "might",
            "must",
            "can",
            "could",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "this",
            "that",
            "these",
            "those",
            "am",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        # Score each document by keyword matches
        scored_docs = []
        for doc_id in doc_ids:
            doc_content = docs[doc_id].get("content", "")
            doc_text = doc_content.lower()

            # Calculate keyword score (ignoring stop words)
            query_words = [w for w in query_lower.split() if w not in stop_words]
            if not query_words:
                # If only stop words in query, use them all
                query_words = query_lower.split()

            words_in_query = set(query_words)
            words_in_doc = set(doc_text.split())
            overlap_words = words_in_query & words_in_doc
            overlap_count = len(overlap_words)

            # Calculate score
            score = overlap_count / max(len(words_in_query), 1)

            # STRICTOR FILTERING:
            # 1. Must match at least 2 non-stop-words (unless the query is very short)
            # 2. Must meet a higher threshold (0.3)
            # 3. Give a bonus to matches that aren't just common words like 'research'
            is_significant_match = overlap_count >= 2 or len(words_in_query) <= 2

            if score > 0.3 and is_significant_match:
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
