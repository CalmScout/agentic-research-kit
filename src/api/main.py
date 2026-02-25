"""FastAPI server for Agentic Research Kit (ARK).

Provides REST API endpoints for querying the research system.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from lightrag.kg.shared_storage import initialize_share_data
from pydantic import BaseModel, Field

from src.agents.lancedb_storage import LanceDBDocStatusStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Research Kit (ARK) API",
    description="Multi-agent RAG system for deep research and analysis",
    version="0.1.0",
)


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="User's query text", min_length=1)
    session: str = Field(default="default", description="Session ID for conversation history")
    debug: bool = Field(default=False, description="Enable debug logging")


class Source(BaseModel):
    """Source document model."""

    text: str = Field(..., description="Source text/content")
    url: str | None = Field(None, description="Source URL")
    score: float | None = Field(None, description="Relevance score")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    sources: list[Source] = Field(default_factory=list, description="Source documents")
    entities: list[str] = Field(default_factory=list, description="Extracted entities")
    retrieved_count: int = Field(default=0, description="Number of documents retrieved")


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str = Field(..., description="Server status")
    architecture: str = Field(..., description="System architecture")
    ingested_docs: int = Field(..., description="Number of ingested documents")


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------


async def get_doc_count() -> int:
    """Get number of ingested documents from LanceDB storage.

    Returns:
        int: Number of documents
    """
    try:
        initialize_share_data()
        storage = LanceDBDocStatusStorage(
            namespace="doc_status",
            workspace="default",
            global_config={"working_dir": "./rag_storage"},
            embedding_func=None,  # Not needed for status count
        )
        await storage.initialize()
        counts = await storage.get_status_counts()
        # Sum all statuses that indicate presence
        return sum(counts.values())
    except Exception as e:
        logger.error(f"Error getting doc count: {e}")
        return 0


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------


@app.get("/", response_model=dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Agentic Research Kit (ARK) API",
        "version": "0.1.0",
        "architecture": "3-agent LangGraph",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "docs": "GET /docs",
            "stats": "GET /stats",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        architecture="3-agent LangGraph",
        ingested_docs=await get_doc_count(),
    )


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Query endpoint using multi-agent workflow.

    Args:
        request: Query request with query text and optional parameters

    Returns:
        QueryResponse: Generated response and sources

    Raises:
        HTTPException: If query processing fails
    """
    try:
        from src.agents.workflow import query_with_agents

        logger.info(f"Received query: '{request.query[:50]}...'")

        # Execute multi-agent workflow
        result = await query_with_agents(
            query=request.query,
            query_image=None,  # TODO: Support image queries
            debug=request.debug,
        )

        # Convert sources to Pydantic models
        sources = []
        for source in result.get("sources", []):
            # Check if source is already grouped (has 'chunks')
            if "chunks" in source:
                text_content = (
                    source["chunks"][0].get("content", "")[:500] if source["chunks"] else ""
                )
            else:
                text_content = source.get("text", source.get("content", ""))[:500]

            sources.append(
                Source(
                    text=text_content,
                    url=source.get("source", source.get("url")),
                    score=source.get("score"),
                )
            )

        # Build response
        response = QueryResponse(
            query=request.query,
            response=result["response"],
            sources=sources,
            entities=result.get("entities", []),
            retrieved_count=result.get("retrieved_count", 0),
        )

        logger.info(f"Query complete for: {request.query[:30]}...")

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        ) from e


@app.get("/stats", response_model=dict[str, Any])
async def stats():
    """Get system statistics."""
    return {
        "ingested_docs": await get_doc_count(),
        "architecture": "3-agent LangGraph",
        "agents": ["Enhanced Retriever", "Enhanced Response Generator", "Verification Agent"],
    }


# -------------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"error": str(exc)},
    )


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
