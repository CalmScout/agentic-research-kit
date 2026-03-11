"""
Simple Query Example

This example demonstrates how to use the Agentic Research Kit (ARK)
to perform research queries and get synthesized responses.

Usage:
    uv run python examples/simple_query.py
"""

import asyncio

from src.agents.workflow import query_with_agents


async def main():
    """Run a simple query through the multi-agent system."""

    # Example research queries to try
    queries = [
        "Is climate change real?",
        "What did the Federal Reserve say about interest rates?",
        "How has bias in biomedical AI evolved?",
    ]

    print("=" * 80)
    print("Agentic Research Kit (ARK) - Simple Query Example")
    print("=" * 80)
    print()

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print(f"{'─' * 80}\n")

        try:
            # Run query through the 2-agent workflow
            result = await query_with_agents(query)

            # Display response
            print(f"Response: {result.get('response', 'No response')}")
            print()

            # Display sources
            if 'sources' in result and result['sources']:
                print("Top Sources:")
                for j, source in enumerate(result['sources'][:3], 1):
                    # Try to find a title or snippet
                    if 'title' in source:
                        title = source['title']
                    elif 'chunks' in source and source['chunks']:
                        title = source['chunks'][0].get('content', 'N/A')[:50] + "..."
                    else:
                        title = source.get('text', source.get('content', 'N/A'))[:50] + "..."

                    print(f"  {j}. {title}")
                    if source.get('score'):
                        print(f"     Relevance: {source['score']:.2f}")
            print()

        except Exception as e:
            print(f"Error: {e}")
            print("Note: Make sure you've ingested data first:")
            print("  uv run ark ingest-dir data/pdf --pattern \"*.pdf\"")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
