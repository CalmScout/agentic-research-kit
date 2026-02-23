"""
Custom Agent Example

This example demonstrates how to extend the MultiModal Agentic RAG system
with custom agents for specialized tasks.

Usage:
    uv run python examples/custom_agent.py
"""

import asyncio
from typing import TypedDict, List, Dict, Any
from langchain_core.language_model.chat_models import BaseChatModel


class CustomAgentState(TypedDict):
    """State for custom agent workflow."""
    query: str
    entities: List[str]
    retrieved_docs: List[Dict[str, Any]]
    analysis: str
    response: str


async def custom_analyzer_agent(
    state: CustomAgentState,
    llm: BaseChatModel
) -> CustomAgentState:
    """
    Custom Agent 1: Domain-Specific Analyzer

    This agent specializes in analyzing queries for a specific domain
    (e.g., finance, healthcare, politics).

    Args:
        state: Current agent state
        llm: Language model to use

    Returns:
        Updated state with analysis
    """

    query = state["query"]

    # Example: Detect domain and add domain-specific context
    prompt = f"""Analyze the following query and identify:
1. The domain (finance, healthcare, politics, etc.)
2. Key entities mentioned
3. The type of verification needed

Query: {query}

Provide your analysis in JSON format with keys: domain, entities, verification_type
"""

    try:
        response = await llm.ainvoke(prompt)
        # Parse response and update state
        state["analysis"] = response.content
        return state
    except Exception as e:
        print(f"Error in custom analyzer: {e}")
        state["analysis"] = "Analysis failed"
        return state


async def custom_retriever_agent(
    state: CustomAgentState,
    retriever_func
) -> CustomAgentState:
    """
    Custom Agent 2: Domain-Aware Retriever

    This agent uses the domain analysis from Agent 1 to optimize
    retrieval for specific domains.

    Args:
        state: Current agent state
        retriever_func: Function to retrieve documents

    Returns:
        Updated state with retrieved documents
    """

    query = state["query"]

    # Use the analysis to customize retrieval
    # For example, prioritize certain document types based on domain

    try:
        # Call the retriever function
        docs = await retriever_func(query, top_k=50)

        # Filter/rank based on domain analysis
        # (This is where you'd add custom logic)

        state["retrieved_docs"] = docs
        return state
    except Exception as e:
        print(f"Error in custom retriever: {e}")
        state["retrieved_docs"] = []
        return state


async def main():
    """Demonstrate custom agent usage."""

    print("=" * 80)
    print("MultiModal Agentic RAG - Custom Agent Example")
    print("=" * 80)
    print()
    print("This example shows how to extend the system with custom agents.")
    print()
    print("Key concepts:")
    print("  1. Define custom state (TypedDict)")
    print("  2. Create custom agent functions")
    print("  3. Use LangGraph for orchestration")
    print("  4. Integrate with existing components")
    print()
    print("Example Use Cases:")
    print("  • Domain-specific analyzers (finance, healthcare)")
    print("  • Specialized retrievers (temporal, geospatial)")
    print("  • Custom aggregators (multi-source fusion)")
    print("  • Domain-specific generators (legal, medical)")
    print()
    print("See the code in this file for implementation examples.")
    print()
    print("=" * 80)

    # Example: Define a simple workflow
    print("\n🔧 Example: Domain-Specific Workflow\n")

    # Simulated state
    state: CustomAgentState = {
        "query": "What is the Federal Reserve's interest rate policy?",
        "entities": [],
        "retrieved_docs": [],
        "analysis": "",
        "response": ""
    }

    print(f"Input Query: {state['query']}")
    print()
    print("Workflow Steps:")
    print("  1. custom_analyzer_agent - Detect domain (finance)")
    print("  2. custom_retriever_agent - Retrieve with finance context")
    print("  3. [Your custom aggregator]")
    print("  4. [Your custom generator]")
    print()

    print("💡 Tip: You can mix custom agents with existing agents")
    print("         from the main workflow for maximum flexibility.")
    print()

    print("=" * 80)
    print("For a complete working example, see:")
    print("  - src/agents/workflow.py (main 2-agent workflow)")
    print("  - src/agents/enhanced_retriever.py (Agent 1 implementation)")
    print("  - src/agents/enhanced_response_generator.py (Agent 2 implementation)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
