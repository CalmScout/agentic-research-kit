"""Retry logic with exponential backoff using tenacity.

Provides automatic retry with exponential backoff for agent operations.
Useful for handling transient failures in API calls, database operations, etc.

Example:
    >>> from src.agents.retry import retry_with_backoff, async_retry_with_backoff
    >>> from src.agents.errors import RetrievalError
    >>>
    >>> @retry_with_backoff(max_attempts=3)
    >>> def unreliable_operation():
    ...     # May fail temporarily
    ...     pass
    >>>
    >>> @async_retry_with_backoff(max_attempts=3)
    >>> async def async_unreliable_operation():
    ...     # May fail temporarily
    ...     pass
"""

import logging
from functools import wraps
from typing import Callable, TypeVar, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.agents.errors import AgentError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_attempts: int = 3,
    multiplier: float = 1.0,
    max_wait: float = 10.0,
    exception_types: tuple = (Exception,)
) -> Callable:
    """Decorator for synchronous functions with retry and exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        multiplier: Multiplier for exponential backoff (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exception_types: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_attempts=3, exception_types=(ConnectionError,))
        >>> def fetch_data(url):
        ...     # May fail temporarily
        ...     return requests.get(url)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=multiplier, max=max_wait),
            retry=retry_if_exception_type(exception_types),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Retry attempt failed for {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_attempts: int = 3,
    multiplier: float = 1.0,
    max_wait: float = 10.0,
    exception_types: tuple = (Exception,)
) -> Callable:
    """Decorator for async functions with retry and exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        multiplier: Multiplier for exponential backoff (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exception_types: Tuple of exception types to retry on

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @async_retry_with_backoff(max_attempts=3, exception_types=(ConnectionError,))
        >>> async def fetch_data_async(url):
        ...     # May fail temporarily
        ...     return await aiohttp.get(url)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=multiplier, max=max_wait),
            retry=retry_if_exception_type(exception_types),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Async retry attempt failed for {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper

    return decorator


# -------------------------------------------------------------------------
# Pre-configured retry decorators for common operations
# -------------------------------------------------------------------------

retry_llm_call = async_retry_with_backoff(
    max_attempts=3,
    multiplier=2.0,
    max_wait=10.0,
    exception_types=(ConnectionError, TimeoutError)
)
"""Decorator for retrying LLM API calls with 2x exponential backoff."""

retry_vector_search = async_retry_with_backoff(
    max_attempts=2,
    multiplier=1.0,
    max_wait=5.0,
    exception_types=(ConnectionError, TimeoutError)
)
"""Decorator for retrying vector search operations."""

retry_tool_execution = async_retry_with_backoff(
    max_attempts=2,
    multiplier=1.0,
    max_wait=5.0,
    exception_types=(AgentError,)
)
"""Decorator for retrying tool execution operations."""
