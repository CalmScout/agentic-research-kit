"""Isolated LightRAG execution to avoid async context conflicts.

This module provides a thread-based isolation layer for LightRAG async operations,
allowing LightRAG to work within LangGraph's async context without event loop conflicts.

Solution:
- Run LightRAG in a dedicated background thread with a persistent event loop
- Use asyncio.run_coroutine_threadsafe for thread-safe interaction
- Propagate OpenTelemetry context across threads for Phoenix observability
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, cast

from lightrag import LightRAG, QueryParam

logger = logging.getLogger(__name__)


class IsolatedLightRAG:
    """Run LightRAG in a dedicated background thread with a persistent event loop.

    This ensures that LightRAG's internal async components (like PriorityQueues)
    are always bound to the same loop, avoiding 'different event loop' errors.
    """

    def __init__(
        self,
        rag_factory: Callable[[], LightRAG],
        max_workers: int = 1,
        timeout: float = 180.0,
    ):
        """Initialize isolated LightRAG wrapper.

        Args:
            rag_factory: Callable that returns a LightRAG instance.
            max_workers: Unused, kept for backward compatibility.
            timeout: Default timeout in seconds for operations.
        """
        self.rag_factory = rag_factory
        self.max_workers = max_workers
        self.timeout = timeout
        self.loop: asyncio.AbstractEventLoop | None = None
        self.rag: LightRAG | None = None
        self.thread: threading.Thread | None = None
        self._ready = threading.Event()

        # Start the background thread
        self._start_background_thread()

        # Wait for initialization to complete
        if not self._ready.wait(timeout=60.0):
            raise RuntimeError("Failed to initialize IsolatedLightRAG background thread")

        logger.info(f"✓ IsolatedLightRAG background thread started (timeout={timeout}s)")

    def _start_background_thread(self):
        """Start the dedicated background thread for LightRAG."""
        self.thread = threading.Thread(target=self._run_loop, name="LightRAGWorker", daemon=True)
        self.thread.start()

    def _run_loop(self):
        """Background thread target: creates and runs an event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Initialize RAG instance within this thread's loop
            logger.debug("Initializing LightRAG instance in background thread...")
            self.rag = self.rag_factory()

            # Initialize storages
            self.loop.run_until_complete(self.rag.initialize_storages())

            # Signal that we are ready
            self._ready.set()

            # Run the loop forever until stop is requested
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Error in LightRAG background thread: {e}", exc_info=True)
        finally:
            # Cleanup - Safe shutdown without run_until_complete
            self._ready.set()  # Ensure we don't block __init__ on error

            try:
                # Cancel all remaining tasks
                for task in asyncio.all_tasks(self.loop):
                    task.cancel()

                # Close the loop
                self.loop.close()
            except Exception as cleanup_err:
                logger.debug(f"Loop cleanup notice: {cleanup_err}")

            logger.debug("LightRAG background thread exiting")

    def _run_coro(
        self, coro_factory: Callable[[LightRAG], Any], timeout: float | None = None
    ) -> Any:
        """Run a coroutine in the background thread's loop."""
        if timeout is None:
            timeout = self.timeout

        if not self.loop or not self.rag:
            raise RuntimeError("IsolatedLightRAG is not initialized")

        # Capture current OpenTelemetry context
        current_context = None
        try:
            from opentelemetry import context

            current_context = context.get_current()
        except ImportError:
            pass

        async def wrapper():
            """Wrapper to execute inside the background loop."""
            # Attach parent context
            token = None
            if current_context:
                try:
                    from opentelemetry import context

                    token = context.attach(current_context)
                except Exception:
                    pass

            try:
                # Create and await the actual coroutine
                coro = coro_factory(self.rag)
                return await coro
            finally:
                if token:
                    try:
                        from opentelemetry import context

                        context.detach(token)
                    except Exception:
                        pass

        # Submit to the background loop
        try:
            # Check loop status
            if not self.loop.is_running():
                raise RuntimeError("Background event loop is not running")

            future: Future = asyncio.run_coroutine_threadsafe(wrapper(), self.loop)
            return future.result(timeout=timeout)
        except Exception as e:
            # Suppress non-fatal initialization error from LightRAG internal background tasks
            if "'list' object has no attribute 'get'" in str(e):
                logger.debug(f"Suppressed non-fatal LightRAG initialization error: {e}")
            else:
                logger.error(f"Error in background task: {e}")
            raise

    def aquery_sync(
        self,
        query: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """Synchronous wrapper for LightRAG queries."""
        logger.debug(f"IsolatedLightRAG.aquery_sync: mode={mode}")

        def coro_factory(rag: LightRAG):
            if only_need_context:
                return rag.aquery_data(query, param=QueryParam(mode=mode))
            else:
                return rag.aquery(query, param=QueryParam(mode=mode))

        return self._run_coro(coro_factory, timeout=timeout)

    def asearch_sync(self, query: str, mode: str = "hybrid", timeout: float | None = None) -> str:
        """Synchronous wrapper for LightRAG.asearch."""

        def coro_factory(rag: LightRAG):
            return rag.asearch(query, param=QueryParam(mode=mode))

        return cast(str, self._run_coro(coro_factory, timeout=timeout))

    def ainsert_sync(self, text: str, timeout: float | None = None) -> None:
        """Synchronous wrapper for LightRAG.ainsert."""

        def coro_factory(rag: LightRAG):
            return rag.ainsert(text)

        self._run_coro(coro_factory, timeout=timeout)

    def close(self):
        """Shutdown the background loop and thread."""
        if self.loop and self.loop.is_running():
            logger.info("Stopping LightRAG background thread...")
            # Schedule loop stop
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread:
                self.thread.join(timeout=5.0)
            logger.info("✓ IsolatedLightRAG closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # We don't close the singleton automatically
        pass


# Global singleton instance
_global_isolated_rag: dict[str, IsolatedLightRAG] = {}


def get_isolated_lightrag(
    rag_factory: Callable[[], LightRAG], timeout: float = 180.0
) -> IsolatedLightRAG:
    """Get or create singleton IsolatedLightRAG instance.

    Includes a liveness check to re-initialize if the thread or loop was closed.
    """
    global _global_isolated_rag
    key = "default"

    # Check if instance exists and is still healthy
    instance = _global_isolated_rag.get(key)
    is_healthy = (
        instance is not None
        and instance.thread is not None
        and instance.thread.is_alive()
        and instance.loop is not None
        and instance.loop.is_running()
    )

    if not is_healthy:
        if instance:
            logger.debug("Re-initializing closed or unhealthy IsolatedLightRAG singleton...")
        _global_isolated_rag[key] = IsolatedLightRAG(rag_factory, timeout=timeout)

    return _global_isolated_rag[key]


def create_isolated_lightrag(
    rag_factory: Callable[[], LightRAG], max_workers: int = 1, timeout: float = 60.0
) -> IsolatedLightRAG:
    """Factory function (backward compatible, ignores max_workers)."""
    return get_isolated_lightrag(rag_factory, timeout=timeout)
