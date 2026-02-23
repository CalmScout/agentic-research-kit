"""Tool for triggering background deep-dive research into entities."""

from typing import Any, Dict, Optional
from pathlib import Path

from src.agents.tools.base import Tool
from src.agents.research_tasks import get_research_task_manager


class EntityDeepDiveTool(Tool):
    """
    Tool that triggers a background research task for a specific entity.
    
    Useful when the agent identifies an important entity that needs more
    thorough research than a single retrieval turn allows.
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "entity_deep_dive"

    @property
    def description(self) -> str:
        return (
            "Trigger a background deep-dive research task for an entity. "
            "The results will be synthesized and added to the research memory later."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string", 
                    "description": "The name of the entity (person, org, concept) to research."
                },
                "focus_area": {
                    "type": "string", 
                    "description": "Optional specific aspect to focus on (e.g., 'financial history', 'political ties')."
                }
            },
            "required": ["entity"]
        }

    async def execute(self, entity: str, focus_area: Optional[str] = None, **kwargs: Any) -> str:
        manager = get_research_task_manager(self.workspace)
        
        description = f"Perform a deep-dive research into '{entity}'."
        if focus_area:
            description += f" Focus particularly on {focus_area}."
            
        task_id = manager.create_task(
            description=description,
            metadata={"entity": entity, "focus_area": focus_area}
        )
        
        # Start the task in background
        import asyncio
        asyncio.create_task(manager.start_task(task_id))
        
        return f"Started background deep-dive for '{entity}' (Task ID: {task_id}). Findings will be distilled into RESEARCH_MEMORY.md once complete."
