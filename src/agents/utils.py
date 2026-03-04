"""Utility functions for multi-agent RAG system.

Common helper functions used across agents.
"""

import json
import re
from typing import Any


def parse_title_from_content(content: str) -> str | None:
    """Extract title from content string if it follows 'Title: ...\nContent: ...' pattern.

    Args:
        content: The text content to parse

    Returns:
        The extracted title or None if not found
    """
    match = re.search(r"^Title:\s*(.*?)\s*\nContent:", content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def group_docs_by_source(docs: list[dict[str, Any]], max_sources: int = 5) -> list[dict[str, Any]]:
    """Group retrieved document chunks by their original source.

    Args:
        docs: List of document chunks with metadata
        max_sources: Maximum number of unique sources to return

    Returns:
        List of unique sources, each containing its chunks
    """
    grouped: dict[str, dict[str, Any]] = {}

    for doc in docs:
        content = doc.get("text", doc.get("content", "")).strip()
        metadata = doc.get("metadata", {})

        # Try to parse title from content if missing from metadata
        parsed_title = parse_title_from_content(content)
        title = metadata.get("title") or parsed_title or "Untitled Document"

        # Determine source key (prefer URL, then file_path, then title, then source metadata)
        source_key = (
            doc.get("url")
            or metadata.get("file_path")
            or parsed_title
            or metadata.get("source")
            or "Unknown source"
        )

        if source_key not in grouped:
            if len(grouped) >= max_sources:
                continue
            grouped[source_key] = {"source": source_key, "title": title, "chunks": []}

        # Add chunk content if not already present
        if content and content not in [c["content"] for c in grouped[source_key]["chunks"]]:
            grouped[source_key]["chunks"].append(
                {"content": content, "score": doc.get("score", 0.0)}
            )

    return list(grouped.values())


def format_response_for_display(
    response: str, sources: list[dict[str, Any]], show_sources: bool = True
) -> str:
    """Format response for CLI/API display.

    Args:
        response: Generated response
        sources: Source documents (can be raw chunks or grouped sources)
        show_sources: Whether to include source documents

    Returns:
        str: Formatted response
    """
    output = [response]

    # Format sources
    if show_sources and sources:
        output.append("\n**Sources**:")

        # Check if sources are already grouped
        if sources and "chunks" in sources[0]:
            unique_sources = sources
        else:
            # Group them now for display
            unique_sources = group_docs_by_source(sources, max_sources=5)

        for i, src in enumerate(unique_sources, 1):
            source_label = src.get("source", "Unknown source")
            title = src.get("title")

            if title and title != "Untitled Document":
                # Don't repeat title if it's the same as source_label (often happens with URLs)
                if title != source_label:
                    output.append(f"{i}. {title}")
                    output.append(f"   {source_label}")
                else:
                    output.append(f"{i}. {source_label}")
            else:
                output.append(f"{i}. {source_label}")

            # Show a snippet of the first chunk
            if src.get("chunks"):
                first_chunk = src["chunks"][0]["content"][:150].strip()
                # Clean up text - remove "Title: ...\nContent: " if present
                first_chunk = re.sub(
                    r"^Title:.*?\nContent:\s*", "", first_chunk, flags=re.DOTALL | re.MULTILINE
                ).strip()
                output.append(f"   Snippet: {first_chunk}...")

    return "\n".join(output)


class QwenToolParser:
    """Parser for Qwen-specific tool calling format.

    Qwen 2.5/3.5 models output tool calls in specific formats like:
    <|im_start|>call:tool_name{"arg1": "val1"}<|im_end|>
    or within ChatML structures.
    """

    @staticmethod
    def parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from Qwen model output.

        Args:
            text: Raw text output from the model

        Returns:
            List of tool call dicts compatible with LangChain tool_calls format
        """
        tool_calls: list[dict[str, Any]] = []

        # Pattern 1: <|im_start|>call:tool_name{...}<|im_end|>
        pattern1 = r"<\|im_start\|>call:([a-zA-Z0-9_]+)(\{.*?\})<\|im_end\|>"
        matches1 = re.finditer(pattern1, text, re.DOTALL)

        for match in matches1:
            name = match.group(1)
            args_str = match.group(2)
            try:
                args = json.loads(args_str)
                tool_calls.append(
                    {
                        "name": name,
                        "args": args,
                        "id": f"call_{len(tool_calls)}_{name}",
                        "type": "tool_call",
                    }
                )
            except json.JSONDecodeError:
                continue

        # Pattern 2: Action: tool_name\nAction Input: {...}
        if not tool_calls:
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", text)
            input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)

            if action_match and input_match:
                name = action_match.group(1)
                args_str = input_match.group(1)
                try:
                    args = json.loads(args_str)
                    tool_calls.append(
                        {
                            "name": name,
                            "args": args,
                            "id": f"call_action_{name}",
                            "type": "tool_call",
                        }
                    )
                except json.JSONDecodeError:
                    pass

        return tool_calls

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove tool calling blocks from text to get clean assistant response.

        Args:
            text: Raw text output

        Returns:
            Cleaned text
        """
        # Remove <|im_start|>call:...<|im_end|> blocks
        text = re.sub(r"<\|im_start\|>call:.*?<\|im_end\|>", "", text, flags=re.DOTALL)

        # Remove Thought blocks if present
        text = re.sub(r"<\|thought\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|im_start\|>thought.*?<\|im_end\|>", "", text, flags=re.DOTALL)

        return text.strip()
