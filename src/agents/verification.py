"""Verification Node (Critique Agent).

This agent runs after the Response Generator to ensure the generated response
is fully supported by the retrieved evidence. It prevents hallucinations by
acting as a final safety check before the response is returned to the user.
"""

import json
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage
import json_repair

from src.agents.base_state import BaseAgentState
from src.agents.model_selector import get_model_selector
from src.agents.enhanced_response_generator import format_sources_for_prompt
from loguru import logger

VERIFICATION_PROMPT = """You are an expert fact-checker and hallucination detector. 
Your job is to verify that a generated response is fully supported by the provided source documents.

You must identify any claims in the Response that are NOT explicitly stated in the Sources.
If the Response contains hallucinations or fabrications, you must provide a 'corrected_response' 
that removes those unsupported claims. If the Response is fully supported, return the exact same response.

## Query
{query}

## Sources
{sources_text}

## Generated Response
{response}

Return your analysis as a JSON object with the following keys:
- "is_verified": true if the response is fully supported, false if it contains hallucinations.
- "feedback": A brief explanation of what was fabricated or why it passed.
- "corrected_response": The original response with hallucinations removed (or the original if perfectly fine). Do NOT change the tone or formatting unnecessarily.

**IMPORTANT**: Respond with ONLY valid JSON, no markdown fences.
"""

async def verification_agent(state: BaseAgentState) -> Dict[str, Any]:
    """Verification Agent to critique and fix hallucinations in the generated response.

    Args:
        state: Current agent state containing query, sources, and generated response.

    Returns:
        Dict with keys: response (potentially corrected), verification_status, verification_feedback
    """
    query = state.get("query", "")
    response_text = state.get("response", "")
    sources = state.get("sources", [])

    logger.info("Verification Agent: Analyzing generated response against sources...")

    # If no sources were retrieved or no response, just pass it through
    if not sources or not response_text:
        logger.info("Verification Agent: Skipped (no sources or empty response)")
        return {
            "verification_status": "skipped",
            "verification_feedback": "No sources or response to verify."
        }

    try:
        model_selector = get_model_selector()
        # Use a strong local LLM or fallback for fact checking
        llm = model_selector.get_llm_with_fallback()

        sources_text = format_sources_for_prompt(sources)
        prompt = VERIFICATION_PROMPT.format(
            query=query,
            sources_text=sources_text,
            response=response_text
        )

        messages = [
            SystemMessage(content="You are a strict verification and fact-checking AI. Output only JSON."),
            HumanMessage(content=prompt)
        ]

        llm_response = await llm.ainvoke(messages)
        text = llm_response.content.strip()
        
        # Clean up JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json_repair.loads(text)
        
        is_verified = result.get("is_verified", True)
        feedback = result.get("feedback", "No issues detected.")
        corrected_response = result.get("corrected_response", response_text)

        status = "verified" if is_verified else "corrected"
        
        if not is_verified:
            logger.warning(f"Verification Failed. Correcting response. Feedback: {feedback}")
        else:
            logger.info("✓ Response verified successfully.")

        return {
            "response": corrected_response,
            "verification_status": status,
            "verification_feedback": feedback
        }

    except Exception as e:
        logger.error(f"Verification Agent failed: {e}", exc_info=True)
        # Fail open: return the original response but mark verification as failed
        return {
            "verification_status": "error",
            "verification_feedback": f"Verification encountered an error: {str(e)}"
        }
