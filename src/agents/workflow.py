"""LangGraph workflow orchestration for multi-agent RAG system.

Orchestrates 3 agents in sequence:
1. Enhanced Retriever (Query Analyzer + Retriever) → Extract entities, embeddings, and retrieve documents
2. Enhanced Response Generator (Evidence Aggregator + Response Generator) → Rerank, synthesize evidence, and generate response
3. Verification Node (Critique Agent) → Validates generated response against retrieved evidence to prevent hallucinations

This simplified workflow reduces complexity while maintaining functionality.
Follows LangGraph best practices from the agents-from-scratch course.
"""

import os
from pathlib import Path
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agents.base_state import BaseAgentState
from src.agents.enhanced_response_generator import enhanced_response_generator_agent
from src.agents.enhanced_retriever import enhanced_retriever_agent
from src.agents.memory import MemoryStore
from src.agents.verification import verification_agent

# -----------------------------------------------------------------------------
# Phoenix Observability Setup
# -----------------------------------------------------------------------------
_phoenix_initialized = False
_tracer_provider = None


def _initialize_phoenix():
    """Initialize Phoenix observability for tracing.

    This function sets up Phoenix tracing for the multi-agent workflow.
    It's called on module import to ensure all traces are captured.

    To enable Phoenix:
    1. Set environment variable PHOENIX_ENABLED=true
    2. Optional: Set PHOENIX_COLLECTOR_ENDPOINT (default: http://localhost:6006/v1/traces)

    Phoenix UI will be available at: http://localhost:6006
    """
    global _phoenix_initialized, _tracer_provider

    phoenix_enabled = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"

    if not phoenix_enabled:
        logger.debug("Phoenix observability disabled (set PHOENIX_ENABLED=true to enable)")
        return

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from phoenix.otel import register as phoenix_register

        # Get collector endpoint from env or use default
        collector_endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )

        logger.info(f"Initializing Phoenix observability at {collector_endpoint}...")

        # Register Phoenix with OpenTelemetry
        _tracer_provider = phoenix_register(
            project_name="agentic-research-kit",
            endpoint=collector_endpoint,
        )

        # CRITICAL: Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Instrument LangChain (which LangGraph uses internally)
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)

        _phoenix_initialized = True
        logger.info("✓ Phoenix + LangChain instrumentation enabled successfully")
        logger.info("  Traces will appear at: http://localhost:6006")

    except ImportError as e:
        logger.warning(f"Phoenix dependencies not installed: {e}")
        logger.warning(
            "  Install with: uv add arize-phoenix openinference-instrumentation-langchain"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix: {e}")
        import traceback

        logger.debug(traceback.format_exc())


# Initialize Phoenix on module import
_initialize_phoenix()


def create_multi_agent_workflow():
    """Create the simplified 3-agent LangGraph workflow.

    Returns:
        Compiled workflow ready for invocation

    Example:
        >>> workflow = create_multi_agent_workflow()
        >>> result = await workflow.ainvoke({"query": "Is climate change real?"})
    """
    logger.info("Creating simplified 3-agent workflow...")

    # Initialize state graph
    workflow = StateGraph(BaseAgentState)

    # -----------------------------------------------------------------
    # Add agents as nodes
    # -----------------------------------------------------------------
    workflow.add_node("enhanced_retriever", enhanced_retriever_agent)
    workflow.add_node("enhanced_response_generator", enhanced_response_generator_agent)
    workflow.add_node("verification_agent", verification_agent)

    logger.info("✓ Added 3 agent nodes")

    # -----------------------------------------------------------------
    # Define workflow edges with ReAct loop
    # -----------------------------------------------------------------
    workflow.add_edge(START, "enhanced_retriever")
    workflow.add_edge("enhanced_retriever", "enhanced_response_generator")
    workflow.add_edge("enhanced_response_generator", "verification_agent")

    # Add conditional edge for iterative refinement
    def router(state: BaseAgentState):
        status = state.get("verification_status")
        iteration_count = state.get("iteration_count", 0)

        # Max 2 refinements (3 total iterations)
        if status == "refine" and iteration_count < 3:
            logger.info(
                f"↺ ReAct Loop: Verification requested refinement (Iteration {iteration_count}). Back to Retriever."
            )
            return "enhanced_retriever"

        if iteration_count >= 3:
            logger.warning(
                f"⚠ ReAct Loop: Maximum iterations ({iteration_count}) reached. Ending research."
            )

        return END

    workflow.add_conditional_edges(
        "verification_agent",
        router,
        {
            "enhanced_retriever": "enhanced_retriever",
            END: END,
        },
    )

    logger.info("✓ Configured workflow edges with ReAct loop")

    # -----------------------------------------------------------------
    # Compile workflow
    # -----------------------------------------------------------------
    app = workflow.compile()

    logger.info("✓ Multi-agent workflow compiled successfully")

    return app


async def query_with_agents(
    query: str,
    query_image: str | None = None,
    retrieval_mode: str = "hybrid",
    debug: bool = False,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Execute simplified multi-agent query pipeline.

    This is the main entry point for querying the system.

    Args:
        query: User's text query
        query_image: Optional image path for multimodal queries
        retrieval_mode: LightRAG retrieval strategy (naive, local, global, hybrid)
        debug: Enable debug logging (configure Loguru level)
        workspace: Optional workspace directory for memory storage (default: ./workspace)

    Returns:
        Dict containing:
            - response: Generated response
            - sources: Source documents
            - query: Original query
            - entities: Extracted entities
            - retrieved_count: Number of documents retrieved

    Example:
        >>> result = await query_with_agents("Is climate change caused by humans?")
        >>> print(result["response"])
    """
    if debug:
        # Enable debug level for Loguru
        logger.enable("src.agents")
        logger.info("Debug logging enabled")

    logger.info(f"Starting multi-agent query: '{query[:50]}...' (mode={retrieval_mode})")

    # Initialize memory store
    workspace_path = workspace or Path("./workspace")
    memory = MemoryStore(workspace_path)

    # Load research context from memory
    memory_context = memory.get_research_context(query=query, max_chars=2000, top_k=5)
    if memory_context:
        logger.debug(f"Loaded {len(memory_context)} chars of research context")

    try:
        # Create workflow
        workflow = create_multi_agent_workflow()

        # Initialize state
        initial_state: BaseAgentState = {
            "query": query,
            "query_image": query_image,
            "retrieval_mode": retrieval_mode,
            "memory_context": memory_context,  # Add research context
            "query_type": "",  # Will be set by Agent 1
            "entities": [],
            "query_embedding": [],
            "retrieved_docs": [],
            "retrieval_scores": [],
            "retrieval_method": "",
            "reranked_docs": [],
            "evidence_summary": "",
            "top_results": [],
            "response": "",
            "sources": [],
            "verification_status": None,
            "verification_feedback": None,
            "messages": [],  # LangGraph message history
            "iteration_count": 0,  # Track loops to prevent infinite refinement
        }

        # Execute workflow
        logger.info("Invoking workflow...")

        # Phoenix Root Span
        if _phoenix_initialized:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind, StatusCode

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "query_with_agents",
                kind=SpanKind.SERVER,
                attributes={
                    "input.value": query,
                    "input.mime_type": "text/plain",
                    "openinference.span.kind": "CHAIN",
                    "retrieval_mode": retrieval_mode,
                },
            ) as span:
                # Set recursion limit to 10 nodes total
                result = await workflow.ainvoke(initial_state, {"recursion_limit": 10})

                # Capture final result
                response = result.get("response", "")
                span.set_attribute("output.value", response)
                span.set_attribute("output.mime_type", "text/plain")

                # Capture trace ID
                trace_id = span.get_span_context().trace_id
                result["phoenix_trace_id"] = f"{trace_id:032x}"

                if result.get("verification_status") == "FAIL":
                    span.set_status(StatusCode.ERROR, "Verification failed")
                else:
                    span.set_status(StatusCode.OK)

            # Force flush spans to ensure they appear in Phoenix
            if _tracer_provider:
                _tracer_provider.force_flush()
        else:
            result = await workflow.ainvoke(initial_state)
            result["phoenix_trace_id"] = None

        # Add metadata for convenience
        result["query"] = query
        result["retrieved_count"] = len(result.get("retrieved_docs", []))
        result["entities"] = result.get("entities", [])

        # Log query to memory store
        try:
            memory.append_query_history(query, result)
            logger.debug("Query logged to memory store")
        except Exception as e:
            logger.warning(f"Failed to log query to memory: {e}")

        logger.info(f"✓ Query complete: sources={len(result.get('sources', []))}")

        return cast(dict[str, Any], result)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        # Return error response
        return {
            "query": query,
            "response": f"I encountered an error processing your query: {str(e)}",
            "sources": [],
            "entities": [],
            "retrieved_count": 0,
            "error": str(e),
        }


def query_with_agents_sync(query: str, query_image: str | None = None) -> dict[str, Any]:
    """Synchronous wrapper for query_with_agents.

    Useful for non-async contexts (e.g., CLI commands).

    Args:
        query: User's text query
        query_image: Optional image path

    Returns:
        Dict: Query results
    """
    import asyncio

    result: dict[str, Any] = asyncio.run(query_with_agents(query, query_image))
    return cast(dict[str, Any], result)


# Main entry point
if __name__ == "__main__":
    # Test the workflow
    import asyncio

    async def test():
        result = await query_with_agents("Is climate change caused by humans?")
        print("Response:", result["response"])
        print("Sources:", len(result["sources"]))

    asyncio.run(test())
