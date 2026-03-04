"""Verification Node (Critique Agent).

This agent runs after the Response Generator to ensure the generated response
is fully supported by the retrieved evidence. It prevents hallucinations by
acting as a final safety check before the response is returned to the user.
"""

from typing import Any, cast

import json_repair
from langchain_core.messages import HumanMessage, SystemMessage
from src.utils.logger import logger

from src.agents.base_state import BaseAgentState
from src.agents.enhanced_response_generator import format_sources_for_prompt
from src.agents.model_selector import get_model_selector

VERIFICATION_PROMPT = """You are an expert fact-checker and research validator.
Your job is to verify that a generated response is fully supported by the provided source documents.

You must identify:
1. Claims in the Response that are NOT explicitly stated in the Sources (Hallucinations).
2. Missing information that was requested in the Query but not found in the Sources (Research Gaps).

If the Response contains hallucinations, provide a 'corrected_response'.
If there are significant 'research_gaps' that prevent a complete answer, set 'is_verified' to false and explain what is missing in 'feedback'.

## Query
{query}

## Sources
{sources_text}

## Generated Response
{response}

Return your analysis as a JSON object with the following keys:
- "is_verified": true if the response is fully supported and complete, false if it contains hallucinations or significant research gaps.
- "needs_refinement": true if we should loop back to the retriever to find more information, false if we should just show the corrected response.
- "feedback": A brief explanation of what was fabricated or what specific information is still missing.
- "corrected_response": The original response with hallucinations removed (or the original if perfectly fine).

**IMPORTANT**: Respond with ONLY valid JSON, no markdown fences.
"""


async def verification_agent(state: BaseAgentState) -> dict[str, Any]:
    """Verification Agent to critique and fix hallucinations or identify research gaps.

    Args:
        state: Current agent state containing query, sources, and generated response.

    Returns:
        Dict with keys: response (potentially corrected), verification_status, verification_feedback
    """
    query = state.get("query", "")
    response_text = cast(str, state.get("response", ""))
    sources = state.get("sources", [])

    # Track iterations to prevent infinite loops
    iteration_count = state.get("iteration_count", 0)
    logger.info(f"Verification Agent (Iteration {iteration_count + 1}): Analyzing response...")

    # If no sources were retrieved or no response, and it's within retry limit
    if not sources or not response_text:
        if iteration_count < 2:
            logger.info("Verification Agent: No sources/response, requesting refinement...")
            return {
                "verification_status": "refine",
                "verification_feedback": "Initial retrieval yielded no results. Need broader search.",
                "iteration_count": iteration_count + 1,
            }
        else:
            return {
                "verification_status": "failed",
                "verification_feedback": "Exhausted retry limit with no results.",
                "iteration_count": iteration_count + 1,
            }

    try:
        model_selector = get_model_selector()
        llm = model_selector.get_llm_with_fallback()

        sources_text = format_sources_for_prompt(sources)
        prompt = VERIFICATION_PROMPT.format(
            query=query, sources_text=sources_text, response=response_text
        )

        llm_messages = [
            SystemMessage(
                content="You are a strict verification and fact-checking AI. Output only JSON."
            ),
            HumanMessage(content=prompt),
        ]

        llm_response = await llm.ainvoke(llm_messages)
        text = cast(str, llm_response.content).strip()

        # Clean up JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = cast(dict[str, Any], json_repair.loads(text))

        is_verified = result.get("is_verified", True)
        needs_refinement = result.get("needs_refinement", False)
        feedback = result.get("feedback", "No issues detected.")
        corrected_response = result.get("corrected_response", response_text)

        # Determine status - allow max 2 refinements (3 total passes)
        if is_verified:
            status = "verified"
        elif needs_refinement and iteration_count < 2:
            status = "refine"
        else:
            status = "corrected"

        if status == "refine":
            logger.warning(f"Verification requested REFINEMENT. Feedback: {feedback}")
        elif status == "corrected":
            logger.warning(f"Verification Failed. Correcting response. Feedback: {feedback}")
        else:
            logger.info("✓ Response verified successfully.")

        return {
            "response": corrected_response,
            "verification_status": status,
            "verification_feedback": feedback,
            "iteration_count": iteration_count + 1,
        }

    except Exception as e:
        logger.error(f"Verification Agent failed: {e}", exc_info=True)
        # Fail open: return the original response but mark verification as failed
        return {
            "verification_status": "error",
            "verification_feedback": f"Verification encountered an error: {str(e)}",
            "iteration_count": iteration_count + 1,
        }
