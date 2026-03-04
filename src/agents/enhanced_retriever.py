"""Enhanced Retriever Agent (combines Query Analyzer + Retriever).

This agent handles:
- Query preprocessing (type detection, entity extraction, embeddings)
- Document retrieval (using tool registry pattern)
- Web research (if local RAG results are insufficient)
- Proactive background research (for important entities)
- Parallel execution where possible

Agent 1 in the simplified 2-agent workflow.
"""

import json
from contextlib import AsyncExitStack
from typing import Any

from src.agents.base_state import BaseAgentState
from src.agents.embeddings import embedder
from src.agents.tools.mcp import MCPServerConfig, connect_mcp_servers
from src.agents.tools.rag_tools.deep_dive import EntityDeepDiveTool
from src.agents.tools.rag_tools.entity_extractor import EntityExtractorTool
from src.agents.tools.rag_tools.hybrid_retriever import HybridRetrieverTool
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.web import WebFetchTool, WebSearchTool
from src.utils.config import get_settings
from src.utils.logger import logger


async def enhanced_retriever_agent(state: BaseAgentState) -> dict[str, Any]:
    """Enhanced Retriever Agent (Query Analyzer + Retriever).

    Combines query analysis and retrieval in a single agent using tool registry.

    Responsibilities:
    1. Detect query type (text vs multimodal)
    2. Extract entities using local LLM
    3. Retrieve relevant documents (Local RAG + Web Search if needed)
    4. Trigger proactive background research
    5. Calculate retrieval scores

    Args:
        state: Current agent state

    Returns:
        Dict with keys: query_type, entities, retrieved_docs, retrieval_scores, retrieval_method
    """
    query = state["query"]
    query_image = state.get("query_image")
    retrieval_mode = state.get("retrieval_mode", "hybrid")
    settings = get_settings()

    logger.info(f"Enhanced Retriever: Processing query '{query[:50]}...' (mode={retrieval_mode})")

    async with AsyncExitStack() as stack:
        try:
            # -------------------------------------------------------------
            # Initialize tool registry
            # -------------------------------------------------------------
            registry = ToolRegistry()
            stack.push_async_callback(registry.close)  # Ensure registry is closed
            registry.register(EntityExtractorTool())
            registry.register(HybridRetrieverTool())
            registry.register(WebSearchTool(api_key=settings.brave_api_key))
            registry.register(WebFetchTool())
            registry.register(EntityDeepDiveTool())

            # -------------------------------------------------------------
            # Step 0: Connect to MCP servers if configured
            # -------------------------------------------------------------
            if settings.mcp_servers:
                logger.info(f"Connecting to {len(settings.mcp_servers)} MCP servers...")
                mcp_configs = [MCPServerConfig(**cfg) for cfg in settings.mcp_servers]
                await connect_mcp_servers(mcp_configs, registry, stack)

            # -------------------------------------------------------------
            # Step 1: Detect query type and analyze verification feedback
            # -------------------------------------------------------------
            query_type = "multimodal" if query_image else "text"

            # If this is a refinement loop, use feedback to enhance the query
            verification_feedback = state.get("verification_feedback")
            active_query = query
            if verification_feedback and state.get("verification_status") == "refine":
                logger.info("Refinement Loop: Enhancing query based on verification feedback")
                # Extract the core gap from feedback to keep the query focused
                active_query = f"{query} (Missing: {verification_feedback[:200]})"

            logger.debug(f"Query type: {query_type}")

            # -------------------------------------------------------------
            # Step 2: Extract entities via tool
            # -------------------------------------------------------------
            entities_result = await registry.execute("entity_extractor", {"text": active_query})
            entities = json.loads(entities_result)

            # CRITICAL: Limit entities to 3 to prevent embedding blowout and memory exhaustion
            if len(entities) > 3:
                logger.debug(f"Limiting {len(entities)} extracted entities to top 3")
                entities = entities[:3]

            logger.debug(f"Active entities: {entities}")

            # -------------------------------------------------------------
            # Step 3: Generate query embedding
            # -------------------------------------------------------------
            try:
                if query_image:
                    query_embedding = embedder.embed_multimodal(query, query_image)
                else:
                    query_embedding = embedder.embed_text(query)

                logger.debug(f"Generated embedding (dim={len(query_embedding)})")

            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                # Create zero embedding as fallback
                query_embedding = [0.0] * 2048

            # -------------------------------------------------------------
            # Step 4: Retrieve documents from Local RAG
            # -------------------------------------------------------------
            retrieval_result = await registry.execute(
                "hybrid_retriever",
                {"query": query, "top_k": settings.retrieval_top_k, "mode": retrieval_mode},
            )

            retrieval_data = json.loads(retrieval_result)
            retrieved_docs = retrieval_data.get("retrieved_docs", [])
            retrieval_scores = retrieval_data.get("retrieval_scores", [])
            retrieval_method = retrieval_data.get("retrieval_method", "keyword")

            logger.info(
                f"Retrieved {len(retrieved_docs)} documents from local RAG using {retrieval_method}"
            )

            # -------------------------------------------------------------
            # Step 5: Web Search Fallback/Augmentation
            # -------------------------------------------------------------
            # If we have very few results, or if Brave key is available and query seems broad
            if len(retrieved_docs) < 5 and settings.brave_api_key:
                logger.info("Insufficient local results, triggering web search...")
                search_result = await registry.execute("web_search", {"query": query, "count": 5})

                # Simple parsing of search results into document format
                if not search_result.startswith("Error"):
                    web_docs = []
                    # Basic parser for the string output of WebSearchTool
                    lines = search_result.split("\n")
                    current_doc: dict[str, Any] | None = None
                    for line in lines:
                        if line and line[0].isdigit() and ". " in line:
                            if current_doc:
                                web_docs.append(current_doc)
                            current_doc = {
                                "text": line.split(". ", 1)[1],
                                "source": "web_search",
                                "score": 0.7,
                            }
                        elif line.strip().startswith("http"):
                            if current_doc:
                                current_doc["url"] = line.strip()
                        elif line.strip() and current_doc:
                            current_doc["text"] += " " + line.strip()

                    if current_doc:
                        web_docs.append(current_doc)

                    # Add web docs to retrieved_docs
                    for doc in web_docs:
                        retrieved_docs.append(
                            {
                                "content": doc["text"],
                                "metadata": {"source": doc.get("url", "web"), "type": "web_search"},
                                "score": doc["score"],
                            }
                        )
                        retrieval_scores.append(doc["score"])

                    logger.info(f"Added {len(web_docs)} documents from web search")
                    retrieval_method = str(retrieval_method) + "+web"

            # -------------------------------------------------------------
            # Step 6: Proactive Background Research
            # -------------------------------------------------------------
            # CRITICAL: Only trigger deep-dives if NOT in local mode to avoid overloading CPU/RAM
            if entities and settings.llm_mode != "local":
                for entity in entities[:2]:  # Max 2 deep dives
                    logger.info(f"Triggering proactive deep-dive for: {entity}")
                    await registry.execute("entity_deep_dive", {"entity": entity})
            elif entities and settings.llm_mode == "local":
                logger.debug("Skipping proactive deep-dives in local mode to conserve resources")

            # -------------------------------------------------------------
            # Return updated state
            # -------------------------------------------------------------
            result = {
                "query_type": query_type,
                "entities": entities,
                "query_embedding": query_embedding,
                "retrieved_docs": retrieved_docs,
                "retrieval_scores": retrieval_scores,
                "retrieval_method": retrieval_method,
            }

            logger.info(
                f"✓ Enhanced Retriever complete: type={query_type}, entities={len(entities)}, docs={len(retrieved_docs)}"
            )

            return result

        except Exception as e:
            logger.error(f"Enhanced Retriever failed: {e}", exc_info=True)
            # Return safe defaults
            return {
                "query_type": "text",
                "entities": [],
                "query_embedding": [0.0] * 2048,
                "retrieved_docs": [],
                "retrieval_scores": [],
                "retrieval_method": "keyword",  # Fallback mode
            }
