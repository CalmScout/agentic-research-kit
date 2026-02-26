import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.agents.retry import retry_with_backoff, async_retry_with_backoff
from src.agents.errors import AgentError

def test_retry_with_backoff_success():
    mock_func = MagicMock(return_value="success")
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_attempts=3)(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 1

def test_retry_with_backoff_failure_then_success():
    mock_func = MagicMock(side_effect=[ValueError("fail"), "success"])
    mock_func.__name__ = "mock_func"
    # Set multiplier to 0 to speed up tests
    decorated = retry_with_backoff(max_attempts=3, multiplier=0)(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 2

def test_retry_with_backoff_all_failures():
    mock_func = MagicMock(side_effect=ValueError("fail"))
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_attempts=3, multiplier=0)(mock_func)
    
    with pytest.raises(ValueError, match="fail"):
        decorated()
    
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_async_retry_with_backoff_success():
    mock_func = MagicMock(return_value=asyncio.Future())
    mock_func.return_value.set_result("success")
    mock_func.__name__ = "mock_func"
    
    decorated = async_retry_with_backoff(max_attempts=3)(mock_func)
    
    result = await decorated()
    
    assert result == "success"
    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_async_retry_with_backoff_failure_then_success():
    future1 = asyncio.Future()
    future1.set_exception(ValueError("fail"))
    future2 = asyncio.Future()
    future2.set_result("success")
    
    mock_func = MagicMock(side_effect=[future1, future2])
    mock_func.__name__ = "mock_func"
    decorated = async_retry_with_backoff(max_attempts=3, multiplier=0)(mock_func)
    
    result = await decorated()
    
    assert result == "success"
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_async_retry_with_backoff_all_failures():
    future = asyncio.Future()
    future.set_exception(ValueError("fail"))
    
    mock_func = MagicMock(side_effect=ValueError("fail"))
    mock_func.__name__ = "mock_func"
    decorated = async_retry_with_backoff(max_attempts=3, multiplier=0)(mock_func)
    
    with pytest.raises(ValueError, match="fail"):
        await decorated()
    
    assert mock_func.call_count == 3
