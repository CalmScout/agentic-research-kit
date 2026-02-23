"""Enhanced Response Generator Agent (combines Evidence Aggregator + Response Generator).

This agent handles:
- Document reranking (top 50 → top 10 → top 5)
- Evidence synthesis
- Final response generation
- Source formatting

Agent 2 in the simplified 2-agent workflow.
"""

import json
import re
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_state import BaseAgentState
from src.agents.model_selector import get_model_selector
from src.agents.prompts import get_template, PromptTemplate
from src.agents.reranker import get_reranker
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.rag_tools.reranker import RerankerTool
from src.agents.utils import group_docs_by_source

from loguru import logger

# Evidence synthesis prompt
EVIDENCE_SYNTHESIS_PROMPT = """You are an expert at synthesizing research evidence.

Analyze the following retrieved documents and provide a structured summary.

Query: {query}

{documents_text}

Provide a concise summary highlighting:
1. **Main Consensus**: What do the documents agree on?
2. **Key Evidence**: What evidence supports this consensus?
3. **Credibility Assessment**: How reliable are these sources?

Format your response as:
**Main Consensus**: [1-2 sentences]
**Key Evidence**: [bullet points of key evidence]
**Credibility**: [assessment of source quality]

Keep your response under 200 words.
"""


def format_evidence_for_synthesis(evidence_list: List[Dict[str, Any]], max_items: int = 10) -> str:
    """Format evidence documents for synthesis.

    Args:
        evidence_list: List of evidence documents
        max_items: Maximum number of items to format

    Returns:
        str: Formatted evidence text
    """
    formatted_items = []

    for i, item in enumerate(evidence_list[:max_items], 1):
        # Extract relevant fields
        text = item.get("text", "")
        score = item.get("score", 0.0)
        source = item.get("source", item.get("url", "Unknown"))

        # Format item
        item_text = f"""Document {i} [Relevance: {score:.2f}]:
{text[:200]}...
Source: {source}
"""
        formatted_items.append(item_text)

    return "\n".join(formatted_items)


def format_sources_for_prompt(grouped_sources: List[Dict[str, Any]]) -> str:
    """Format grouped sources for prompt.

    Args:
        grouped_sources: List of sources, each with 'title', 'source', and 'chunks'

    Returns:
        str: Formatted sources text for LLM context
    """
    if not grouped_sources:
        return "No sources available."

    formatted = [f"TOTAL SOURCES AVAILABLE: {len(grouped_sources)}\n"]

    for i, src in enumerate(grouped_sources, 1):
        title = src.get("title", "Untitled Document")
        source_id = src.get("source", "Unknown")
        
        formatted.append(f"--- SOURCE {i} START ---")
        formatted.append(f"IDENTIFIER: [Source {i}]")
        formatted.append(f"TITLE: {title}")
        formatted.append(f"REFERENCE: {source_id}")
        formatted.append("CONTENT:")
        
        for j, chunk in enumerate(src.get("chunks", []), 1):
            content = chunk.get("content", "")
            # Clean up redundant title if present
            content = re.sub(r'^Title:.*?\nContent:\s*', '', content, flags=re.DOTALL | re.MULTILINE).strip()
            
            if len(src.get("chunks", [])) > 1:
                formatted.append(f"   (Part {j}): {content}")
            else:
                formatted.append(f"   {content}")
        
        formatted.append(f"--- SOURCE {i} END ---\n")

    return "\n".join(formatted)


async def enhanced_response_generator_agent(
    state: BaseAgentState,
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """Enhanced Response Generator Agent (Evidence Aggregator + Response Generator).

    Combines evidence aggregation and response generation in a single agent.

    Responsibilities:
    1. Rerank documents (top 50 → top 10 → top 5)
    2. Synthesize evidence from top results
    3. Generate final response with citations

    Args:
        state: Current agent state
        prompt_template: Name of prompt template to use (research, claim_verification, etc.)
                       If None, uses DEFAULT_TEMPLATE_NAME

    Returns:
        Dict with keys: response, sources
    """
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])

    logger.info(f"Enhanced Response Generator: Processing {len(retrieved_docs)} retrieved docs")

    async with AsyncExitStack() as stack:
        try:
            # Initialize tool registry
            registry = ToolRegistry()
            stack.push_async_callback(registry.close) # Ensure registry is closed
            registry.register(RerankerTool())

            # -------------------------------------------------------------
            # Step 1: Rerank documents (50 → 10)
            # -------------------------------------------------------------
            if len(retrieved_docs) == 0:
                logger.warning("No documents to process")
                return {
                    "response": "I couldn't find any relevant information for your query.",
                    "sources": [],
                }

            # Use the reranker tool
            rerank_result = await registry.execute("reranker", {
                "docs": retrieved_docs,
                "query": query,
                "top_k": 10
            })

            rerank_data = json.loads(rerank_result)
            reranked_docs = rerank_data.get("reranked_docs", [])

            if not reranked_docs:
                logger.warning("No documents after reranking")
                return {
                    "response": "I couldn't find any relevant information after reranking.",
                    "sources": [],
                }

            logger.info(f"Reranked to {len(reranked_docs)} top documents")

            # -------------------------------------------------------------
            # Step 2: Synthesize evidence from top 10
            # -------------------------------------------------------------
            try:
                model_selector = get_model_selector()
                llm = model_selector.get_local_llm()

                # Format documents for synthesis
                documents_text = format_evidence_for_synthesis(reranked_docs, max_items=10)

                # Create synthesis prompt
                prompt = EVIDENCE_SYNTHESIS_PROMPT.format(
                    query=query, documents_text=documents_text
                )

                # Invoke LLM
                messages = [SystemMessage(content=prompt)]
                response = await llm.ainvoke(messages)

                evidence_summary = response.content.strip()
                logger.debug(f"Evidence summary: {evidence_summary[:100]}...")

            except Exception as e:
                logger.warning(f"Evidence synthesis failed: {e}")
                # Fallback: Simple concatenation
                evidence_summary = f"Found {len(reranked_docs)} relevant documents."
                if reranked_docs:
                    top_text = reranked_docs[0].get("text", "")[:200]
                    evidence_summary += f"\n\nTop result: {top_text}..."

            # -------------------------------------------------------------
            # Step 3: Group by source to avoid "Source X" hallucinations
            # -------------------------------------------------------------
            # Group unique documents (max 5 unique documents)
            top_sources = group_docs_by_source(reranked_docs, max_sources=5)

            # -------------------------------------------------------------
            # Step 4: Generate final response
            # -------------------------------------------------------------
            try:
                model_selector = get_model_selector()
                llm = model_selector.get_llm_with_fallback()

                # Get prompt template
                template_name = prompt_template or "research"  # Default
                template = get_template(template_name)

                logger.debug(f"Using prompt template: {template_name}")

                # Format sources for prompt
                sources_text = format_sources_for_prompt(top_sources)

                # Generate prompt from template
                user_prompt = template.format_user_prompt(
                    query=query,
                    evidence_summary=evidence_summary,
                    sources_text=sources_text,
                )

                # Combine system prompt and user prompt
                full_prompt = template.get_full_prompt(
                    query=query,
                    evidence_summary=evidence_summary,
                    sources_text=sources_text,
                )
                
                # ADD EXTRA EMPHASIS ON SOURCE COUNT
                source_count_warning = f"\n\nIMPORTANT: There are ONLY {len(top_sources)} sources provided above. Do NOT cite any Source number greater than {len(top_sources)}."
                full_prompt += source_count_warning

                # Generate response
                messages = [SystemMessage(content=full_prompt)]
                response = await llm.ainvoke(messages)

                response_text = response.content.strip()
                logger.debug(f"Generated response: {response_text[:100]}...")

            except Exception as e:
                logger.error(f"Response generation failed: {e}", exc_info=True)
                # Fallback response
                response_text = f"""**Answer**: Based on the available evidence:

{evidence_summary[:300]}...

**Sources**:
{format_sources_for_prompt(top_sources)}
"""

            # -------------------------------------------------------------
            # Return updated state
            # -------------------------------------------------------------
            result = {
                "response": response_text,
                "sources": top_sources,
                "reranked_docs": reranked_docs,
            }

            logger.info("✓ Enhanced Response Generator complete: response generated")

            return result

        except Exception as e:
            logger.error(f"Enhanced Response Generator failed: {e}", exc_info=True)
            # Return safe fallback
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "sources": [],
            }
