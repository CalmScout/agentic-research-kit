"""LangGraph workflow orchestration for multi-agent RAG system.

Orchestrates 3 agents in sequence:
1. Research Coordinator → Orchestrates specialized search subgraphs
2. Enhanced Response Generator (Evidence Aggregator + Response Generator) → Rerank, synthesize evidence, and generate response
3. Verification Node (Critique Agent) → Validates generated response against retrieved evidence

This hybrid architecture merges LangGraph predictability with Nanobot flexibility.
"""

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace

from src.agents.base_state import BaseAgentState
from src.agents.enhanced_response_generator import enhanced_response_generator_agent
from src.agents.memory import MemoryStore
from src.agents.skills import SkillsLoader
from src.agents.subgraphs.rag_search import rag_search_node
from src.agents.subgraphs.web_search import web_search_node
from src.agents.verification import verification_agent
from src.utils.logger import logger, setup_logging
from src.utils.observability import _observability_initialized, setup_observability

# Initialize logging and observability (Phoenix/OpenTelemetry) if enabled in settings
setup_logging()
setup_observability()


async def skill_injector_node(state: BaseAgentState) -> dict[str, Any]:
    """Node that injects domain-specific research protocols into the state."""
    requested = state.get("requested_skills", [])
    if not requested:
        return {"skill_instructions": ""}

    loader = SkillsLoader(Path("./skills"))
    instructions = loader.get_skill_injection(requested)
    return {"skill_instructions": instructions}


async def research_coordinator_node(state: BaseAgentState) -> dict[str, Any]:
    """
    Analyzes query and determines the best research strategy.
    Acts as the entry point for specialized subgraphs.
    """
    query = state["query"]
    logger.info(f"Research Coordinator: Analyzing query '{query[:50]}'")

    # We could do complex entity extraction here if needed,
    # but for Step 7 we just prepare the state for subgraphs.
    return {"retrieved_docs": [], "retrieval_method": "coordinator"}


async def join_research_results(state: BaseAgentState) -> dict[str, Any]:
    """
    Final node in the research phase that merges results from all subgraphs.
    """
    logger.info("Research Coordinator: Merging results from all subgraphs")

    # In a sequential graph, retrieved_docs might have been appended to already.
    # In a parallel graph, we'd need to handle merging more carefully.
    all_docs = state.get("retrieved_docs", [])

    return {
        "retrieved_count": len(all_docs),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def create_multi_agent_workflow():
    """Create the hybrid LangGraph workflow with specialized subgraphs.

    Returns:
        Compiled workflow ready for invocation
    """
    logger.info("Creating hybrid research workflow...")

    # Initialize state graph
    workflow = StateGraph(BaseAgentState)

    # -----------------------------------------------------------------
    # Add nodes
    # -----------------------------------------------------------------
    workflow.add_node("skill_injector", skill_injector_node)
    workflow.add_node("research_coordinator", research_coordinator_node)

    # Specialized Subgraphs (nodes)
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_node("join_results", join_research_results)
    workflow.add_node("enhanced_response_generator", enhanced_response_generator_agent)
    workflow.add_node("verification_agent", verification_agent)

    # -----------------------------------------------------------------
    # Define edges and flow (Hardware-Aware)
    # -----------------------------------------------------------------

    # 1. Start with Skill Injection
    workflow.add_edge(START, "skill_injector")
    workflow.add_edge("skill_injector", "research_coordinator")

    # 2. Research Phase (Sequential for 12GB VRAM)
    # In Phase 7, we prioritize stability on local hardware.
    # We execute RAG, THEN Web Search to keep memory usage low.
    workflow.add_edge("research_coordinator", "rag_search")

    def handle_rag_output(state: dict):
        """Map rag subgraph output back to base state."""
        return {
            "retrieved_docs": state.get("results", []),
            "retrieval_method": state.get("method", "rag"),
        }

    # Note: For simple sequential flow without full subgraph complexity,
    # we can just use regular nodes. If using actual Compiled Graphs as nodes,
    # we'd use the .invoke() pattern within a wrapper node.
    # For now, we use node-based separation for clarity.

    workflow.add_edge("rag_search", "web_search")

    # For web_search, we need to append to existing results
    def web_search_wrapper(state: BaseAgentState):
        # This wrapper would be needed if we were using actual subgraphs
        pass

    workflow.add_edge("web_search", "join_results")

    # 3. Generation & Verification
    workflow.add_edge("join_results", "enhanced_response_generator")
    workflow.add_edge("enhanced_response_generator", "verification_agent")

    # ReAct loop router
    def router(state: BaseAgentState) -> str:
        status = state.get("verification_status")
        iteration = state.get("iteration_count", 0)
        max_iterations = 3

        if status == "verified" or iteration >= max_iterations:
            return END

        logger.info(f"↺ ReAct Loop: Refinement requested ({status}). Returning to Coordinator.")
        return "research_coordinator"

    workflow.add_conditional_edges(
        "verification_agent",
        router,
        {"research_coordinator": "research_coordinator", END: END},
    )

    app = workflow.compile()
    logger.info("✓ Hybrid research workflow compiled successfully")
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
