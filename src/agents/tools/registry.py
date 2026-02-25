"""Tool registry for dynamic tool management.

Copied from nanobot framework with modifications for ARK's use case.
"""

import json
from typing import Any
from loguru import logger

from src.agents.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool execution result as string.

        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        # Phoenix/OpenTelemetry Tracing
        try:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
            
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"tool:{name}",
                kind=SpanKind.CLIENT,
                attributes={
                    "tool.name": name,
                    "tool.params": json.dumps(params)
                }
            ) as span:
                errors = tool.validate_params(params)
                if errors:
                    error_msg = f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
                    span.set_attribute("tool.error", error_msg)
                    return error_msg
                
                result = await tool.execute(**params)
                # Cap result length in trace to avoid overhead
                span.set_attribute("tool.result", str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
                return result
        except ImportError:
            # Fallback if opentelemetry is not installed
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        except Exception as e:
            logger.error(f"Error executing {name}: {str(e)}")
            return f"Error executing {name}: {str(e)}"

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    async def close(self) -> None:
        """Close all registered tools and clean up resources."""
        for tool in self._tools.values():
            try:
                await tool.close()
            except Exception as e:
                logger.warning(f"Error closing tool '{tool.name}': {e}")
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
