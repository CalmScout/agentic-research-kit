"""Entity extraction tool.

Extracts entities (people, organizations, locations) from text using local LLM.
"""

import json
from typing import Any

from langchain_core.messages import SystemMessage

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
            messages = [SystemMessage(content=prompt)]
            response = await llm.ainvoke(messages)

            # Parse entities
            entities_text = response.content
            if isinstance(entities_text, str):
                entities_text = entities_text.strip()
                if entities_text.lower() == "none":
                    entities = []
                else:
                    entities = [
                        e.strip()
                        for e in entities_text.split(",")
                        if e.strip() and e.strip().lower() != "none"
                    ]
            else:
                entities = []

            logger.debug(f"Extracted {len(entities)} entities: {entities}")
            return json.dumps(entities)

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return json.dumps([])  # Return empty list on error
