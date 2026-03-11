"""
Web Search Subgraph for specialized real-time information gathering.
Part of the hardware-aware fractal subagent pattern.
"""

import asyncio
import json
import re
from typing import Any

from src.agents.tools.web import WebFetchTool, WebSearchTool
from src.utils.config import get_settings
from src.utils.logger import logger


async def web_search_node(state: dict) -> dict[str, Any]:
    """
    Sub-node that performs web search and fetching.
    Now includes snippets as immediate evidence fallbacks.
    """
    query = state["query"]
    existing_docs = state.get("retrieved_docs", [])
    settings = get_settings()

    if not settings.brave_api_key:
        logger.warning("WebSearch Subgraph: BRAVE_API_KEY missing, skipping.")
        return {"retrieved_docs": existing_docs}

    logger.info(f"WebSearch Subgraph: Searching for '{query[:50]}'")

    search_tool = WebSearchTool(api_key=settings.brave_api_key)
    fetch_tool = WebFetchTool()

    # 1. Perform Search
    search_raw = await search_tool.execute(query=query, count=5)

    # 2. Extract Snippets and URLs
    # Brave Search tool returns a formatted string. We parse it to get both.
    new_results = []
    urls = []

    # Simple regex to find URL and the line after it (the snippet)
    # The tool format is: 1. Title\n   URL\n   Snippet
    sections = re.split(r"\n\d+\. ", search_raw)
    for section in sections:
        lines = [line.strip() for line in section.split("\n") if line.strip()]
        if len(lines) >= 2:
            # First line might be title, but we look for the URL
            url = None
            snippet = ""
            for i, line in enumerate(lines):
                if line.startswith("http"):
                    url = line
                    # The rest is the snippet
                    snippet = " ".join(lines[i + 1 :])
                    break

            if url:
                urls.append(url)
                if snippet:
                    new_results.append(
                        {
                            "content": f"SEARCH SNIPPET: {snippet}",
                            "metadata": {"source": url, "type": "web_snippet"},
                            "score": 0.6,  # Lower score than full fetch
                        }
                    )

    logger.info(f"WebSearch Subgraph: Captured {len(new_results)} search snippets")

    # 3. Concurrent Fetching (Attempt to get full detail)
    logger.info(f"WebSearch Subgraph: Attempting full fetch for top {min(3, len(urls))} URLs...")

    fetch_tasks = []
    for url in urls[:3]:  # Hardware safety
        fetch_tasks.append(fetch_tool.execute(url=url))

    fetch_results_raw = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    fetched_count = 0
    for i, res in enumerate(fetch_results_raw):
        if isinstance(res, str):
            try:
                data = json.loads(res)
                if "text" in data and "error" not in data:
                    new_results.append(
                        {
                            "content": data["text"],
                            "metadata": {"source": data.get("url", urls[i]), "type": "web_fetch"},
                            "score": 0.8,
                        }
                    )
                    fetched_count += 1
            except Exception:
                continue

    logger.info(f"WebSearch Subgraph: Successfully fetched {fetched_count} full pages")

    # Combine all results
    all_docs = existing_docs + new_results

    return {
        "retrieved_docs": all_docs,
        "retrieval_method": state.get("retrieval_method", "") + "+web",
    }
