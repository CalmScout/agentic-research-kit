"""Tests for MCP tool integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from contextlib import AsyncExitStack

from src.agents.tools.mcp import MCPServerConfig, connect_mcp_servers, MCPToolWrapper
from src.agents.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_tool_wrapper():
    """Test wrapping an MCP tool."""
    mock_session = AsyncMock()
    
    # Mock tool definition from MCP
    mock_tool_def = MagicMock()
    mock_tool_def.name = "test_tool"
    mock_tool_def.description = "A test tool"
    mock_tool_def.inputSchema = {
        "type": "object",
        "properties": {
            "arg1": {"type": "string"}
        }
    }
    
    # Mock MCP content types
    from mcp import types
    mock_response = MagicMock()
    mock_response.content = [types.TextContent(type="text", text="Tool result")]
    mock_session.call_tool.return_value = mock_response
    
    wrapper = MCPToolWrapper(mock_session, "test_server", mock_tool_def)
    
    assert wrapper.name == "mcp_test_server_test_tool"
    assert wrapper.description == "A test tool"
    assert wrapper.parameters == mock_tool_def.inputSchema
    
    result = await wrapper.execute(arg1="hello")
    assert result == "Tool result"
    mock_session.call_tool.assert_called_once_with("test_tool", arguments={"arg1": "hello"})


@pytest.mark.asyncio
async def test_connect_mcp_servers_stdio():
    """Test connecting to MCP server via stdio."""
    registry = ToolRegistry()
    stack = AsyncExitStack()
    
    cfg = MCPServerConfig(
        name="test_server",
        command="echo",
        args=["hello"]
    )
    
    mock_tool_def = MagicMock()
    mock_tool_def.name = "echo_tool"
    mock_tool_def.description = "Echoes input"
    mock_tool_def.inputSchema = {"type": "object", "properties": {}}
    
    mock_tools_list = MagicMock()
    mock_tools_list.tools = [mock_tool_def]
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_list)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    
    # Mock mcp.client.stdio.stdio_client
    with patch("mcp.client.stdio.stdio_client", return_value=AsyncMock()) as mock_stdio:
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        
        # Mock ClientSession
        with patch("mcp.ClientSession", return_value=mock_session):
            await connect_mcp_servers([cfg], registry, stack)
            
            assert registry.has("mcp_test_server_echo_tool")
            assert len(registry) == 1
            mock_session.initialize.assert_called_once()
            mock_session.list_tools.assert_called_once()


@pytest.mark.asyncio
async def test_connect_mcp_servers_http():
    """Test connecting to MCP server via HTTP."""
    registry = ToolRegistry()
    stack = AsyncExitStack()
    
    cfg = MCPServerConfig(
        name="http_server",
        url="http://example.com/mcp"
    )
    
    mock_tools_list = MagicMock()
    mock_tools_list.tools = []
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_list)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    
    # Mock mcp.client.streamable_http.streamable_http_client
    with patch("mcp.client.streamable_http.streamable_http_client", return_value=AsyncMock()) as mock_http:
        mock_http.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock(), None)
        
        # Mock ClientSession
        with patch("mcp.ClientSession", return_value=mock_session):
            await connect_mcp_servers([cfg], registry, stack)
            
            assert len(registry) == 0
            mock_session.initialize.assert_called_once()
