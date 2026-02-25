"""LightRAG HTTP API Client.

Provides async HTTP client for communicating with LightRAG's FastAPI server.
This avoids async context manager conflicts when using LightRAG with LangGraph.
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
import httpx

from src.utils.config import Settings

logger = logging.getLogger(__name__)


class LightRAGHTTPClient:
    """HTTP client for LightRAG API server.

    Manages LightRAG server lifecycle and provides query methods.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize LightRAG HTTP client.

        Args:
            settings: Application settings (uses defaults if None)
        """
        self.settings = settings or Settings()
        self.host = self.settings.lightrag_api_host
        self.port = self.settings.lightrag_api_port
        self.base_url = f"http://{self.host}:{self.port}"
        self._server_process: Optional[subprocess.Popen] = None
        self._client: Optional[httpx.AsyncClient] = None

        # Query timeout (seconds)
        self.query_timeout = 120  # 2 minutes for complex queries

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            httpx.AsyncClient: HTTP client
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.query_timeout)
        return self._client

    async def health_check(self) -> bool:
        """Check if LightRAG server is healthy.

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def start_server(self) -> None:
        """Start LightRAG API server programmatically.

        Raises:
            RuntimeError: If server fails to start
        """
        if self._server_process is not None:
            logger.info("LightRAG server already running")
            return

        logger.info(f"Starting LightRAG API server on {self.host}:{self.port}...")

        # Prepare command to start LightRAG server
        # Note: LightRAG server has its own CLI (not standard uvicorn)
        # Use openai binding (compatible with DeepSeek API)
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lightrag.api.lightrag_server",
            "--host", self.host,
            "--port", str(self.port),
            "--llm-binding", "openai",
            "--embedding-binding", "openai",
            "--log-level", "INFO",
        ]

        # Set environment variables
        env = {
            "LIGHTRAG_WORKING_DIR": self.settings.rag_working_dir,
            # Use high-performance LanceDB storage backends
            "LIGHTRAG_KV_STORAGE": "LanceDBKVStorage",
            "LIGHTRAG_VECTOR_STORAGE": "LanceDBVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE": "LanceDBDocStatusStorage",
            # Set embedding dimension to match Qwen3-VL-Embedding-2B (2048)
            "EMBEDDING_DIM": "2048",
        }

        try:
            # Start server as subprocess
            self._server_process = subprocess.Popen(
                cmd,
                env={**subprocess.os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                if await self.health_check():
                    logger.info("✓ LightRAG server started successfully")
                    return

            raise RuntimeError("LightRAG server failed to start within 30 seconds")

        except Exception as e:
            logger.error(f"Failed to start LightRAG server: {e}")
            if self._server_process:
                self._server_process.kill()
                self._server_process = None
            raise

    async def stop_server(self) -> None:
        """Stop LightRAG API server.

        Gracefully shuts down the server process.
        """
        if self._server_process is None:
            logger.debug("LightRAG server not running")
            return

        logger.info("Stopping LightRAG server...")
        self._server_process.terminate()

        try:
            self._server_process.wait(timeout=10)
            logger.info("✓ LightRAG server stopped")
        except subprocess.TimeoutExpired:
            logger.warning("LightRAG server did not stop gracefully, forcing...")
            self._server_process.kill()
            logger.info("✓ LightRAG server force killed")
        finally:
            self._server_process = None

    async def ensure_server_running(self) -> None:
        """Ensure LightRAG server is running.

        Starts server if not running, checks health if running.
        Auto-starts on first query if configured.

        Raises:
            RuntimeError: If server fails to start or becomes unhealthy
        """
        if self._server_process is None or not await self.health_check():
            if self.settings.lightrag_auto_start_server:
                await self.start_server()
            else:
                raise RuntimeError(
                    "LightRAG server is not running and auto-start is disabled. "
                    "Please start it manually or enable LIGHTRAG_AUTO_START_SERVER."
                )

    async def query_hybrid(
        self, query: str, top_k: int = 50
    ) -> Dict[str, Any]:
        """Query LightRAG using hybrid mode.

        Hybrid mode combines:
        - Vector similarity search
        - BM25 keyword search
        - Knowledge graph traversal

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Dict with keys:
                - context: List of retrieved documents
                - response: Generated answer text
                - sources: Source documents

        Raises:
            RuntimeError: If query fails
        """
        await self.ensure_server_running()

        logger.info(f"Querying LightRAG API: '{query[:50]}...' (mode=hybrid, top_k={top_k})")

        try:
            client = await self._get_client()

            # Prepare request for /query/data endpoint (retrieval-only)
            payload = {
                "query": query,
                "mode": "naive",  # Pure vector similarity search (fastest)
                "chunk_top_k": top_k,  # Number of chunks to retrieve
            }

            # Send query to /query/data endpoint (retrieval-only, no LLM needed)
            response = await client.post(
                f"{self.base_url}/query/data",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            response.raise_for_status()

            result = response.json()

            # Extract documents from /query/data response structure
            documents = []
            scores = []

            # /query/data returns: {"data": {"chunks": [...], "entities": [...], "relationships": [...]}}
            if "data" in result and "chunks" in result["data"]:
                chunks = result["data"]["chunks"]
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if isinstance(chunk, dict):
                            doc = {
                                "text": chunk.get("content", ""),
                                "score": 1.0,  # /query/data doesn't return similarity scores
                                "metadata": {
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "file_path": chunk.get("file_path", ""),
                                    "reference_id": chunk.get("reference_id", ""),
                                },
                            }
                            documents.append(doc)
                            scores.append(doc["score"])

            logger.info(f"✓ Retrieved {len(documents)} documents from LightRAG API (endpoint: /query/data)")

            # If no documents retrieved, raise error to trigger fallback
            if len(documents) == 0:
                raise RuntimeError("LightRAG API returned no documents (possibly due to empty index or query mismatch)")

            return {
                "retrieved_docs": documents,
                "retrieval_scores": scores,
                "retrieval_method": "hybrid",
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during query: {e}")
            raise RuntimeError(f"LightRAG API query failed: {e}")
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to query LightRAG API: {e}")

    async def close(self) -> None:
        """Close resources and cleanup.

        Stops server if auto-started and closes HTTP client.
        """
        if self.settings.lightrag_auto_start_server:
            await self.stop_server()

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        logger.info("LightRAG HTTP client closed")


# Singleton instance for efficiency
_client: Optional[LightRAGHTTPClient] = None


def get_lightrag_client() -> LightRAGHTTPClient:
    """Get singleton LightRAG HTTP client instance.

    Returns:
        LightRAGHTTPClient: Shared client instance

    Example:
        >>> client = get_lightrag_client()
        >>> result = await client.query_hybrid("climate change", top_k=10)
        >>> print(f"Retrieved {len(result['retrieved_docs'])} docs")
    """
    global _client
    if _client is None:
        _client = LightRAGHTTPClient()
    return _client
