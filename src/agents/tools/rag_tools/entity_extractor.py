"""Entity extraction tool.

Extracts entities (people, organizations, locations) from text using local LLM.
"""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.model_selector import get_model_selector
from src.agents.tools.base import Tool
from src.utils.logger import logger

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """Extract key entities from the following query.
Focus on: people, organizations, locations, and important concepts.

Query: {query}

Return ONLY a comma-separated list of entities. For example:
"Joe Biden, White House, inflation, economy"

If no entities are found, return: "none"
"""


class EntityExtractorTool(Tool):
    """Tool for extracting entities from text."""

    @property
    def name(self) -> str:
        return "entity_extractor"

    @property
    def description(self) -> str:
        return "Extract entities (people, organizations, locations) from text"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract entities from"}
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Extract entities from the given text.

        Args:
            **kwargs: Must contain 'text'

        Returns:
            JSON string with list of entities
        """
        text = kwargs.get("text")
        if not text:
            return json.dumps([])

        try:
            model_selector = get_model_selector()
            llm = model_selector.get_local_llm()

            # Create entity extraction prompt
            prompt = ENTITY_EXTRACTION_PROMPT.format(query=text)

            # Invoke LLM
            messages = [
                SystemMessage(
                    content="You are a strict entity extractor. Extract ONLY a comma-separated list of entities (people, organizations, concepts). DO NOT include any introductory text, thinking process, or headers. If no entities, return 'none'."
                ),
                HumanMessage(content=prompt),
            ]
            response = await llm.ainvoke(messages)

            # Parse entities
            entities_text = response.content
            if isinstance(entities_text, str):
                # The llm instance from model_selector is already wrapped in ThinkingProcessStripper,
                # but we add an extra layer of safety here for specific extraction artifacts.

                # Remove common reasoning headers that might leak
                entities_text = re.sub(
                    r"^(Thinking Process:|Analysis:|Entities:|Output:).*$",
                    "",
                    entities_text,
                    flags=re.MULTILINE | re.IGNORECASE,
                )

                # If we still have multiple lines, try to find the one that looks like a list
                if "\n" in entities_text.strip():
                    lines = [line.strip() for line in entities_text.split("\n") if line.strip()]
                    # Prefer lines with commas
                    csv_lines = [
                        line for line in lines if "," in line and not line.startswith("1.")
                    ]
                    if csv_lines:
                        entities_text = csv_lines[-1]
                    else:
                        entities_text = lines[-1]

                entities_text = entities_text.strip()

                if not entities_text or entities_text.lower() == "none":
                    entities: list[str] = []
                else:
                    # Clean up each entity (remove bullets, quotes, etc)
                    entities = []
                    # STOPWORDS: Common reasoning artifacts or conversational filler to ignore
                    blacklist = {
                        "wait",
                        "actually",
                        "thinking",
                        "process",
                        "analysis",
                        "entities",
                        "output",
                        "extract",
                        "found",
                        "sure",
                        "ok",
                        "okay",
                        "here",
                        "none",
                        "the",
                        "a",
                        "an",
                        "this",
                        "that",
                        "background research",
                        "however",
                        "usually",
                        "but",
                        "also",
                        "therefore",
                        "thus",
                        "hence",
                        "research",
                        "query",
                        "input",
                        "seems",
                        "based",
                        "according",
                        "to",
                    }

                    for entity in entities_text.split(","):
                        # Remove leading numbers (e.g., "1. Quantum")
                        clean_e = re.sub(r"^\d+[\.\)]\s*", "", entity.strip())
                        clean_e = clean_e.strip('"').strip("'").strip("*").strip("- ")

                        # VALIDATION:
                        # 1. Not empty or too short
                        # 2. Not in blacklist
                        # 3. Not a full sentence (max 5 words)
                        if (
                            clean_e
                            and clean_e.lower() not in blacklist
                            and len(clean_e) > 1
                            and clean_e.count(" ") < 5
                        ):
                            entities.append(clean_e)
            else:
                entities = []

            logger.debug(f"Extracted {len(entities)} entities: {entities}")
            return json.dumps(entities)

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return json.dumps([])  # Return empty list on error
