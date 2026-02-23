"""MCP client: connects to MCP servers and wraps their tools as native agent tools.

Adapted from nanobot framework for ARK.
"""

import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import httpx
from loguru import logger

from src.agents.tools.base import Tool
from src.agents.tools.registry import ToolRegistry


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as an ARK Tool."""

    def __init__(self, session, server_name: str, tool_def):
        self._session = session
        self._original_name = tool_def.name
        # Prefix with mcp_ and server name to avoid collisions
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        try:
            result = await self._session.call_tool(self._original_name, arguments=kwargs)
            parts = []
            for block in result.content:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)
                elif isinstance(block, types.ImageContent):
                    parts.append(f"[Image: {block.mimeType}]")
                elif isinstance(block, types.EmbeddedResource):
                    parts.append(f"[Resource: {block.resource.uri}]")
                else:
                    parts.append(str(block))
            return "\n".join(parts) or "(no output)"
        except Exception as e:
            logger.error(f"Error executing MCP tool {self._name}: {e}")
            return f"Error executing tool: {str(e)}"


async def connect_mcp_servers(
    mcp_servers: List[MCPServerConfig], 
    registry: ToolRegistry, 
    stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools.

    Args:
        mcp_servers: List of MCPServerConfig objects.
        registry: ToolRegistry to register discovered tools into.
        stack: AsyncExitStack to manage lifetimes of connections.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.error("mcp-sdk not installed. Please install with 'uv add mcp'")
        return

    for cfg in mcp_servers:
        name = cfg.name
        try:
            if cfg.command:
                params = StdioServerParameters(
                    command=cfg.command, 
                    args=cfg.args, 
                    env=cfg.env
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif cfg.url:
                from mcp.client.streamable_http import streamable_http_client
                if cfg.headers:
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=cfg.headers,
                            follow_redirects=True
                        )
                    )
                    read, write, _ = await stack.enter_async_context(
                        streamable_http_client(cfg.url, http_client=http_client)
                    )
                else:
                    read, write, _ = await stack.enter_async_context(
                        streamable_http_client(cfg.url)
                    )
            else:
                logger.warning(f"MCP server '{name}': no command or url configured, skipping")
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_list = await session.list_tools()
            for tool_def in tools_list.tools:
                wrapper = MCPToolWrapper(session, name, tool_def)
                registry.register(wrapper)
                logger.debug(f"MCP: registered tool '{wrapper.name}' from server '{name}'")

            logger.info(f"MCP server '{name}': connected, {len(tools_list.tools)} tools registered")
        except Exception as e:
            logger.error(f"MCP server '{name}': failed to connect: {e}")
