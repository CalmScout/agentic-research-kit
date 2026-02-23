"""Tool registry system for RAG agents.

Copied from nanobot framework with modifications for ARK's use case.
"""

from src.agents.tools.base import Tool
from src.agents.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry"]
