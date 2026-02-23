"""Isolated LightRAG execution to avoid async context conflicts.

This module provides a thread-based isolation layer for LightRAG async operations,
allowing LightRAG to work within LangGraph's async context without event loop conflicts.

Problem:
- LightRAG's embedding workers expect their own event loop
- When called from within LangGraph's async context, the async context managers fail
- Error: 'NoneType' object does not support the asynchronous context manager protocol

Solution:
- Run LightRAG in a separate thread with its own event loop
- Use ThreadPoolExecutor to isolate execution contexts
- Maintain synchronous interface for easy integration
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Any, Dict, Callable
from threading import local

from lightrag import LightRAG, QueryParam

logger = logging.getLogger(__name__)


class IsolatedLightRAG:
    """Run LightRAG in isolated thread to avoid async conflicts.

    This class wraps a LightRAG instance and executes its async methods in a
    separate thread with its own event loop, preventing conflicts with LangGraph's
    async context.

    Note: The LightRAG instance should ideally be initialized within the isolated 
    thread to avoid any event loop binding issues.
    """

    def __init__(
        self,
        rag_factory: Callable[[], LightRAG],
        max_workers: int = 1,
        timeout: float = 60.0
    ):
        """Initialize isolated LightRAG wrapper.

        Args:
            rag_factory: Callable that returns a LightRAG instance. 
                         Called within the isolated thread.
            max_workers: Maximum number of worker threads (default: 1)
            timeout: Default timeout in seconds for async operations
        """
        self.rag_factory = rag_factory
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Thread-local storage for event loops and RAG instances
        self._thread_local = local()

        logger.info(f"✓ IsolatedLightRAG initialized (workers={max_workers}, timeout={timeout}s)")

    def _get_thread_resources(self) -> tuple[asyncio.AbstractEventLoop, LightRAG]:
        """Get or create event loop and RAG instance for current thread.

        Each thread gets its own event loop and its own RAG instance
        to avoid event loop binding conflicts.

        Returns:
            tuple: (event_loop, rag_instance)
        """
        # Check if thread already has resources
        if hasattr(self._thread_local, 'loop') and self._thread_local.loop is not None:
            loop = self._thread_local.loop
            rag = self._thread_local.rag
            if not loop.is_closed():
                return loop, rag

        # Create new event loop for this thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Initialize RAG instance within this loop
        logger.info("Initializing LightRAG instance within isolated thread...")
        rag = self.rag_factory()
        
        # CRITICAL: Ensure storages are initialized within this thread's event loop
        # This creates the necessary async locks (JsonKVStorage._storage_lock)
        # without which queries will fail with 'NoneType' object errors.
        logger.debug("Initializing LightRAG storages in isolated thread...")
        loop.run_until_complete(rag.initialize_storages())
        
        # Store in thread-local storage
        self._thread_local.loop = loop
        self._thread_local.rag = rag
        return loop, rag

    def _run_in_thread(self, coro_factory: Callable[[LightRAG], Any], timeout: Optional[float] = None) -> Any:
        """Run async coroutine in isolated thread.

        Args:
            coro_factory: Callable that takes a LightRAG instance and returns a coroutine
            timeout: Timeout in seconds (uses default if None)

        Returns:
            Result of the coroutine
        """
        if timeout is None:
            timeout = self.timeout

        def run_in_thread():
            """Execute coroutine in thread's event loop."""
            loop, rag = self._get_thread_resources()
            try:
                # Create coroutine using the RAG instance bound to this thread
                coro = coro_factory(rag)
                return loop.run_until_complete(coro)
            except Exception as e:
                logger.error(f"Exception in thread loop: {e}", exc_info=True)
                raise

        try:
            future = self.executor.submit(run_in_thread)
            result = future.result(timeout=timeout)
            return result
        except FuturesTimeoutError:
            logger.error(f"LightRAG operation timed out after {timeout}s")
            raise TimeoutError(f"LightRAG operation timed out after {timeout}s")

    def aquery_sync(
        self,
        query: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        timeout: Optional[float] = None
    ) -> Any:
        """Synchronous wrapper for LightRAG queries (runs in isolated thread).

        Args:
            query: Query string
            mode: Query mode ("local", "global", "hybrid")
            only_need_context: If True, uses aquery_data for structured results
            timeout: Timeout in seconds

        Returns:
            Query response string or dict
        """
        logger.debug(f"IsolatedLightRAG.aquery_sync: query='{query[:50]}...', mode={mode}, only_need_context={only_need_context}")

        def coro_factory(rag: LightRAG):
            if only_need_context:
                return rag.aquery_data(query, param=QueryParam(mode=mode))
            else:
                return rag.aquery(query, param=QueryParam(mode=mode))

        return self._run_in_thread(coro_factory, timeout=timeout)

    def asearch_sync(
        self,
        query: str,
        mode: str = "hybrid",
        timeout: Optional[float] = None
    ) -> str:
        """Synchronous wrapper for LightRAG.asearch (runs in isolated thread)."""
        logger.debug(f"IsolatedLightRAG.asearch_sync: query='{query[:50]}...', mode={mode}")

        def coro_factory(rag: LightRAG):
            return rag.asearch(query, param=QueryParam(mode=mode))

        return self._run_in_thread(coro_factory, timeout=timeout)

    def ainsert_sync(
        self,
        text: str,
        timeout: Optional[float] = None
    ) -> None:
        """Synchronous wrapper for LightRAG.ainsert (runs in isolated thread)."""
        logger.debug(f"IsolatedLightRAG.ainsert_sync: text length={len(text)}")

        def coro_factory(rag: LightRAG):
            return rag.ainsert(text)

        self._run_in_thread(coro_factory, timeout=timeout)

    def close(self):
        """Clean up resources.

        Shuts down the thread pool and cleans up event loops.
        """
        try:
            def cleanup_thread():
                """Cleanup event loop in the worker thread."""
                if hasattr(self._thread_local, 'loop') and self._thread_local.loop is not None:
                    loop = self._thread_local.loop
                    if not loop.is_closed():
                        logger.debug("Closing event loop in isolated thread...")
                        # Cancel all pending tasks in this loop
                        pending = asyncio.all_tasks(loop)
                        if pending:
                            logger.debug(f"Cancelling {len(pending)} pending tasks in isolated thread")
                            for task in pending:
                                task.cancel()
                            # Run the loop until all tasks are cancelled
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        
                        loop.close()
                        self._thread_local.loop = None
                        logger.debug("✓ Isolated event loop closed")

            # Submit cleanup task to the executor
            self.executor.submit(cleanup_thread)
            
            # Shut down executor
            self.executor.shutdown(wait=True)
            logger.info("✓ IsolatedLightRAG closed")
        except Exception as e:
            logger.warning(f"Error closing IsolatedLightRAG: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_isolated_lightrag(
    rag_factory: Callable[[], LightRAG],
    max_workers: int = 1,
    timeout: float = 60.0
) -> IsolatedLightRAG:
    """Factory function to create IsolatedLightRAG instance.

    Args:
        rag_factory: Callable that returns a LightRAG instance
        max_workers: Maximum number of worker threads
        timeout: Default timeout in seconds

    Returns:
        IsolatedLightRAG: Wrapped instance ready for use
    """
    return IsolatedLightRAG(rag_factory, max_workers=max_workers, timeout=timeout)
