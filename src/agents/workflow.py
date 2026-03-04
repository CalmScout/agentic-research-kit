"""LangGraph workflow orchestration for multi-agent RAG system.

Orchestrates 3 agents in sequence:
1. Enhanced Retriever (Query Analyzer + Retriever) → Extract entities, embeddings, and retrieve documents
2. Enhanced Response Generator (Evidence Aggregator + Response Generator) → Rerank, synthesize evidence, and generate response
3. Verification Node (Critique Agent) → Validates generated response against retrieved evidence to prevent hallucinations

This simplified workflow reduces complexity while maintaining functionality.
Follows LangGraph best practices from the agents-from-scratch course.
"""

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace

from src.agents.base_state import BaseAgentState
from src.agents.enhanced_response_generator import enhanced_response_generator_agent
from src.agents.enhanced_retriever import enhanced_retriever_agent
from src.agents.memory import MemoryStore
from src.agents.verification import verification_agent
from src.utils.logger import logger, setup_logging
from src.utils.observability import _observability_initialized, setup_observability

# Initialize logging and observability (Phoenix/OpenTelemetry) if enabled in settings
setup_logging()
setup_observability()


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

    # -----------------------------------------------------------------
    # Define edges and flow
    # -----------------------------------------------------------------
    workflow.add_edge(START, "enhanced_retriever")
    workflow.add_edge("enhanced_retriever", "enhanced_response_generator")
    workflow.add_edge("enhanced_response_generator", "verification_agent")

    # The logic of the ReAct loop is handled by the conditional edge from the verification node
    def router(state: BaseAgentState) -> str:
        """Route the workflow based on verification results.

        Args:
            state: Current agent state

        Returns:
            Next node name
        """
        status = state.get("verification_status")
        iteration = state.get("iteration_count", 0)

        # Max iterations to prevent infinite loops
        max_iterations = 3

        if status == "verified" or iteration >= max_iterations:
            if iteration >= max_iterations and status != "verified":
                logger.warning(
                    f"⚠ ReAct Loop: Maximum iterations ({max_iterations}) reached. Ending research."
                )
            return END

        # If refinement or correction is needed, loop back to retriever
        logger.info(
            f"↺ ReAct Loop: Verification requested {status} (Iteration {iteration}). Back to Retriever."
        )
        return "enhanced_retriever"

    workflow.add_conditional_edges(
        "verification_agent",
        router,
        {"enhanced_retriever": "enhanced_retriever", END: END},
    )

    logger.info("✓ Configured workflow edges with ReAct loop")

    # Compile the workflow
    app = workflow.compile()
    logger.info("✓ Multi-agent workflow compiled successfully")
    return app


async def query_with_agents(
    query: str,
    query_image: str | None = None,
    retrieval_mode: str = "hybrid",
    session_id: str = "default",
    debug: bool = False,
) -> dict[str, Any]:
    """Execute the multi-agent workflow for a research query.

    This is the main entry point for querying the system via agents.

    Args:
        query: User research question
        query_image: Optional path to an image for multimodal analysis
        retrieval_mode: LightRAG mode (naive, local, global, hybrid)
        session_id: ID for conversation memory
        debug: Enable detailed logging

    Returns:
        dict: Final response and metadata
    """
    if debug:
        logger.info("Debug logging enabled")

    logger.info(f"Starting multi-agent query: '{query[:50]}...' (mode={retrieval_mode})")

    # Initialize memory store
    workspace_path = Path("./workspace")
    memory = MemoryStore(workspace_path)
    context = memory.get_research_context(query)
    if context:
        logger.debug(f"Loaded {len(context)} chars of research context")

    # Execute workflow
    try:
        # Compile workflow
        workflow_app = create_multi_agent_workflow()

        # Initial state
        initial_state: BaseAgentState = {
            "query": query,
            "query_image": query_image,
            "retrieval_mode": retrieval_mode,
            "memory_context": context,
            "session_id": session_id,
            "query_type": "text",
            "entities": [],
            "query_embedding": [],
            "retrieved_docs": [],
            "retrieval_scores": [],
            "retrieval_method": "keyword",
            "reranked_docs": [],
            "evidence_summary": "",
            "top_results": [],
            "response": "",
            "sources": [],
            "verification_status": "pending",
            "verification_feedback": "",
            "iteration_count": 0,
            "messages": [],
            "metadata": {},
        }

        if _observability_initialized:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("research_query") as span:
                logger.info("Invoking workflow...")
                final_state = await workflow_app.ainvoke(initial_state)

                # Capture Phoenix trace ID
                trace_id = format(span.get_span_context().trace_id, "032x")
                if "metadata" not in final_state:
                    final_state["metadata"] = {}
                final_state["metadata"]["phoenix_trace_id"] = trace_id
        else:
            logger.info("Invoking workflow...")
            final_state = await workflow_app.ainvoke(initial_state)

        # Ensure required keys exist even if mocks missed them
        if "metadata" not in final_state:
            final_state["metadata"] = {}
        if "retrieved_docs" not in final_state:
            final_state["retrieved_docs"] = []

        # Log result
        logger.info(f"✓ Query complete: sources={len(final_state.get('retrieved_docs', []))}")

        # Update memory
        memory.append_query_history(query, final_state, session_id=session_id)
        logger.debug("Query logged to memory store")

        return {
            "query": query,
            "response": final_state.get("response", ""),
            "sources": final_state.get("retrieved_docs", []),
            "verification_status": final_state.get("verification_status", "pending"),
            "verification_feedback": final_state.get("verification_feedback", ""),
            "iteration_count": final_state.get("iteration_count", 0),
            "entities": final_state.get("entities", []),
            "messages": final_state.get("messages", []),
            "retrieved_count": len(final_state.get("retrieved_docs", [])),
            "phoenix_trace_id": final_state.get("metadata", {}).get("phoenix_trace_id"),
        }

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {
            "query": query,
            "response": f"Error during research: {str(e)}",
            "error": str(e),
            "sources": [],
            "retrieved_count": 0,
        }


def query_with_agents_sync(
    query: str,
    query_image: str | None = None,
    retrieval_mode: str = "hybrid",
    session_id: str = "default",
    debug: bool = False,
) -> dict[str, Any]:
    """Synchronous wrapper for query_with_agents.

    Useful for CLI and other non-async contexts.
    """
    import asyncio

    return asyncio.run(
        query_with_agents(
            query=query,
            query_image=query_image,
            retrieval_mode=retrieval_mode,
            session_id=session_id,
            debug=debug,
        )
    )
