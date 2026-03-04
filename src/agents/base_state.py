"""Generic agent state management for multi-agent RAG workflows.

Defines the base state that flows through all agents using LangGraph's
state management. Domain-specific states can extend this base.
"""

from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import add_messages


class BaseAgentState(TypedDict):
    """Generic base state for multi-agent RAG workflows.

    This state flows through all 2 agents:
    1. Enhanced Retriever → Extracts entities, generates embeddings, and retrieves documents
    2. Enhanced Response Generator → Reranks, synthesizes evidence, and generates response
    3. Verification Node → Validates generated response against retrieved evidence

    The state is designed to be domain-agnostic. For domain-specific use cases,
    create a subclass that adds additional fields.

    Attributes:
        query: User's original query (text)
        query_image: Optional image for multimodal queries
        retrieval_mode: LightRAG retrieval strategy (naive, local, global, hybrid)
        query_type: Query modality ("text" or "multimodal")
        entities: Extracted entities (people, orgs, locations, concepts)
        query_embedding: Vector representation of query
        retrieved_docs: Documents retrieved from RAG system
        retrieval_scores: Similarity scores for retrieved docs
        retrieval_method: Search method used ("vector", "bm25", "hybrid", "keyword")
        reranked_docs: Top-K documents after reranking
        evidence_summary: Synthesized evidence from top results
        top_results: Top K most relevant results
        response: Final generated response
        sources: Source documents for citation
        verification_status: Status from verification node ("verified" or "corrected")
        verification_feedback: Feedback from the critique LLM
        messages: LangGraph message history (auto-annotated)
        session_id: Optional session identifier
        metadata: Optional metadata dictionary
    """

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------
    query: str
    query_image: str | None
    retrieval_mode: str | None  # LightRAG mode: naive, local, global, hybrid
    memory_context: str | None  # Research context from memory store
    session_id: NotRequired[str | None]
    metadata: NotRequired[dict[str, Any]]

    # -------------------------------------------------------------------------
    # Agent 1: Enhanced Retriever output
    # -------------------------------------------------------------------------
    query_type: str  # "text" or "multimodal"
    entities: list[str]
    query_embedding: list[float]
    retrieved_docs: list[dict]
    retrieval_scores: list[float]
    retrieval_method: str  # "vector", "bm25", "hybrid", "keyword"

    # -------------------------------------------------------------------------
    # Agent 2: Enhanced Response Generator output
    # -------------------------------------------------------------------------
    reranked_docs: list[dict]
    evidence_summary: str
    top_results: list[dict]
    response: str
    sources: list[dict]

    # -------------------------------------------------------------------------
    # Agent 3: Verification Node output
    # -------------------------------------------------------------------------
    verification_status: str | None
    verification_feedback: str | None

    # -------------------------------------------------------------------------
    # LangGraph message history (auto-annotated for automatic reduction)
    # -------------------------------------------------------------------------
    messages: Annotated[list, add_messages]
    iteration_count: int  # Track loops to prevent infinite refinement


class ResearchState(BaseAgentState):
    """Domain-specific state for research and analysis use cases.

    Extends BaseAgentState with research-specific fields.
    """

    # Research-specific fields
    research_methodology: NotRequired[str | None]
    cited_works: NotRequired[list[str] | None]
    limitations: NotRequired[list[str] | None]


class LegacyClaimState(BaseAgentState):
    """Legacy state for claim verification use cases (backward compatibility).

    Extends BaseAgentState with claim-specific fields.
    """

    # Claim-specific fields
    claim_verdict: NotRequired[str | None]
    fact_check_url: NotRequired[str | None]
    top_claims: NotRequired[list[dict] | None]  # Deprecated: Use top_results instead
