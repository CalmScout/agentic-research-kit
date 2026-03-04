"""Research Task Manager for background and proactive research.

Inspired by nanobot's SubagentManager and CronService.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.model_selector import get_model_selector
from src.agents.tools.rag_tools.hybrid_retriever import HybridRetrieverTool
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.web import WebFetchTool, WebSearchTool
from src.utils.config import get_settings
from src.utils.logger import logger


@dataclass
class ResearchTask:
    """Represents a background research task."""

    id: str
    description: str
    status: str = "pending"  # pending, running, completed, failed
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchTaskManager:
    """Manages background research tasks and their execution."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.settings = get_settings()
        self._tasks: dict[str, ResearchTask] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}

    def create_task(self, description: str, metadata: dict[str, Any] | None = None) -> str:
        """Create a new research task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        task = ResearchTask(id=task_id, description=description, metadata=metadata or {})
        self._tasks[task_id] = task
        logger.info(f"Created research task [{task_id}]: {description[:50]}...")
        return task_id

    async def start_task(self, task_id: str):
        """Start a task in the background."""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        if task.status == "running":
            return

        bg_task = asyncio.create_task(self._run_task_logic(task))
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda t: self._running_tasks.pop(task_id, None))

    async def _run_task_logic(self, task: ResearchTask):
        """Internal logic to execute a research task using an agent loop."""
        task.status = "running"
        logger.info(f"Starting research task [{task.id}]")

        try:
            selector = get_model_selector()
            llm = selector.get_llm_with_fallback()

            # Setup tools for the background researcher
            registry = ToolRegistry()
            registry.register(WebSearchTool(api_key=self.settings.brave_api_key))
            registry.register(WebFetchTool())
            registry.register(HybridRetrieverTool())

            # Bind tools to LLM if supported
            # Use a generic runnable variable to handle different return types
            runnable: Any = llm
            if hasattr(llm, "bind_tools"):
                logger.debug(f"Task [{task.id}] binding {len(registry)} tools to LLM")
                runnable = llm.bind_tools(registry.get_definitions())

            # System prompt for background researcher
            system_prompt = f"""You are an expert Background Researcher for the Agentic Research Kit (ARK).
Your goal is to perform a deep-dive research into the assigned task.

Rules:
1. Use the available tools (web search, fetch, RAG) to gather information.
2. Synthesize your findings into a comprehensive report.
3. Be objective, factual, and cite your sources.
4. If you encounter conflicting information, report both sides.

Current Date: {datetime.now().strftime("%Y-%m-%d")}
Workspace: {self.workspace}
"""

            from langchain_core.messages import HumanMessage, SystemMessage

            messages: list[Any] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please research the following: {task.description}"),
            ]

            # Simple agent loop
            max_iterations = 10
            iteration = 0
            final_report = ""

            while iteration < max_iterations:
                iteration += 1
                # Use the bound runnable if available
                response = await runnable.ainvoke(messages)

                # Check for tool calls (this depends on the LLM implementation)
                # If using ChatOpenAI, it will have tool_calls attribute
                if hasattr(response, "tool_calls") and response.tool_calls:
                    # Append assistant message
                    messages.append(response)

                    for tool_call in response.tool_calls:
                        logger.debug(f"Task [{task.id}] executing {tool_call['name']}")
                        result = await registry.execute(tool_call["name"], tool_call["args"])

                        # Add tool result to messages
                        from langchain_core.messages import ToolMessage

                        messages.append(ToolMessage(tool_call_id=tool_call["id"], content=result))
                else:
                    final_report = str(response.content)
                    break

            task.result = final_report or "No report generated."
            task.status = "completed"
            task.completed_at = time.time()
            logger.info(f"Research task [{task.id}] completed successfully")

            # Auto-append finding to memory store
            from src.agents.memory.store import MemoryStore

            memory = MemoryStore(self.workspace)
            memory.append_research_finding(f"""Background Research [{task.id}]: {task.description}

{task.result}""")

        except Exception as e:
            logger.error(f"Research task [{task.id}] failed: {e}")
            task.status = "failed"
            task.error = str(e)
            task.completed_at = time.time()

    def get_task(self, task_id: str) -> ResearchTask | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ResearchTask]:
        """List all tasks."""
        return list(self._tasks.values())


# Singleton instance
_manager: ResearchTaskManager | None = None


def get_research_task_manager(workspace: Path | None = None) -> ResearchTaskManager:
    """Get global research task manager."""
    global _manager
    if _manager is None:
        workspace_path = workspace or Path("./workspace")
        _manager = ResearchTaskManager(workspace_path)
    return _manager
