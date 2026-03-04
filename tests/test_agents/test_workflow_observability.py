"""Tests for workflow observability and Phoenix integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import os
from pathlib import Path

from src.agents.workflow import query_with_agents
from src.utils.observability import setup_observability
from src.utils.config import clear_settings_cache

@pytest.mark.asyncio
async def test_setup_observability_enabled():
    """Test Phoenix initialization when enabled."""
    # Reset settings cache to read from environment
    clear_settings_cache()
    
    with patch.dict(os.environ, {"PHOENIX_ENABLED": "true"}):
        with patch("openinference.instrumentation.langchain.LangChainInstrumentor") as mock_instrumentor:
            with patch("phoenix.otel.register") as mock_register:
                # Reset initialized flag for test
                import src.utils.observability
                src.utils.observability._observability_initialized = False
                
                setup_observability()
                
                assert src.utils.observability._observability_initialized is True
                mock_register.assert_called_once()
                mock_instrumentor.return_value.instrument.assert_called_once()
    
    # Clean up settings cache after test
    clear_settings_cache()

@pytest.mark.asyncio
async def test_setup_observability_import_error():
    """Test Phoenix initialization with missing dependencies."""
    # Reset settings cache
    clear_settings_cache()
    
    with patch.dict(os.environ, {"PHOENIX_ENABLED": "true"}):
        with patch("phoenix.otel.register", side_effect=ImportError("No module named 'phoenix'")):
            import src.utils.observability
            src.utils.observability._observability_initialized = False
            
            setup_observability()
            
            assert src.utils.observability._observability_initialized is False
    
    # Clean up settings cache after test
    clear_settings_cache()

@pytest.mark.asyncio
async def test_query_with_agents_trace_capture():
    """Test that Phoenix trace ID is captured when enabled."""
    query = "test query"
    
    # Mock Phoenix enabled
    with patch("src.agents.workflow._observability_initialized", True):
        # Mock OpenTelemetry trace
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        # Ensure trace_id is an integer so formatting works
        mock_span.get_span_context.return_value.trace_id = 0x1234567890abcdef1234567890abcdef
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
            with patch("src.agents.workflow.create_multi_agent_workflow") as mock_wf:
                mock_app = MagicMock()
                mock_app.ainvoke = AsyncMock(return_value={
                    "response": "test",
                    "retrieved_docs": [],
                    "entities": []
                })
                mock_wf.return_value = mock_app
                
                # Mock memory store
                with patch("src.agents.workflow.MemoryStore"):
                    result = await query_with_agents(query)
                    
                    assert "phoenix_trace_id" in result
                    assert result["phoenix_trace_id"] == "1234567890abcdef1234567890abcdef"

@pytest.mark.asyncio
async def test_query_with_agents_trace_capture_error():
    """Test trace capture handles errors gracefully."""
    query = "test query"
    
    with patch("src.agents.workflow._observability_initialized", True):
        # side_effect on get_tracer will trigger the broad try-except in query_with_agents
        with patch("opentelemetry.trace.get_tracer", side_effect=Exception("Trace error")):
            with patch("src.agents.workflow.create_multi_agent_workflow") as mock_wf:
                mock_app = MagicMock()
                mock_app.ainvoke = AsyncMock(return_value={
                    "response": "test",
                    "retrieved_docs": [],
                    "entities": []
                })
                mock_wf.return_value = mock_app
                
                with patch("src.agents.workflow.MemoryStore"):
                    result = await query_with_agents(query)
                    
                    # When an error occurs in the Phoenix block, it returns an error response
                    assert "error" in result
                    assert "Trace error" in result["error"]
