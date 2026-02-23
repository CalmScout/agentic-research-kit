"""Utility functions for multi-agent RAG system.

Common helper functions used across agents.
"""

import re
from typing import Dict, Any, List


def parse_title_from_content(content: str) -> str | None:
    """Extract title from content string if it follows 'Title: ...\nContent: ...' pattern.

    Args:
        content: The text content to parse

    Returns:
        The extracted title or None if not found
    """
    match = re.search(r'^Title:\s*(.*?)\s*\nContent:', content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def group_docs_by_source(docs: List[Dict[str, Any]], max_sources: int = 5) -> List[Dict[str, Any]]:
    """Group retrieved document chunks by their original source.

    Args:
        docs: List of document chunks with metadata
        max_sources: Maximum number of unique sources to return

    Returns:
        List of unique sources, each containing its chunks
    """
    grouped = {}
    
    for doc in docs:
        content = doc.get("text", doc.get("content", "")).strip()
        metadata = doc.get("metadata", {})
        
        # Try to parse title from content if missing from metadata
        parsed_title = parse_title_from_content(content)
        title = metadata.get("title") or parsed_title or "Untitled Document"
        
        # Determine source key (prefer URL, then file_path, then title, then source metadata)
        source_key = doc.get("url") or metadata.get("file_path") or parsed_title or metadata.get("source") or "Unknown source"
        
        if source_key not in grouped:
            if len(grouped) >= max_sources:
                continue
            grouped[source_key] = {
                "source": source_key,
                "title": title,
                "chunks": []
            }
        
        # Add chunk content if not already present
        if content and content not in [c["content"] for c in grouped[source_key]["chunks"]]:
            grouped[source_key]["chunks"].append({
                "content": content,
                "score": doc.get("score", 0.0)
            })

    return list(grouped.values())


def format_response_for_display(
    response: str, 
    sources: List[Dict[str, Any]],
    show_sources: bool = True
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
                first_chunk = re.sub(r'^Title:.*?\nContent:\s*', '', first_chunk, flags=re.DOTALL | re.MULTILINE).strip()
                output.append(f"   Snippet: {first_chunk}...")

    return "\n".join(output)
