from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.research_tasks import ResearchTask, ResearchTaskManager


@pytest.fixture
def manager(tmp_path):
    return ResearchTaskManager(workspace=tmp_path)

def test_create_task(manager):
    task_id = manager.create_task("test research")
    assert task_id in manager._tasks
    assert manager._tasks[task_id].description == "test research"
    assert manager._tasks[task_id].status == "pending"

@pytest.mark.asyncio
async def test_run_task_logic_success(manager):
    task = ResearchTask(id="task1", description="test")

    mock_llm = AsyncMock()
    # Mock bind_tools to return self (required because it's called in _run_task_logic)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    mock_response = MagicMock()
    mock_response.content = "Research report"
    mock_response.tool_calls = []
    mock_llm.ainvoke.return_value = mock_response

    mock_selector = MagicMock()
    mock_selector.get_llm_with_fallback.return_value = mock_llm

    with (
        patch("src.agents.research_tasks.get_model_selector", return_value=mock_selector),
        patch("src.agents.research_tasks.ToolRegistry"),
        patch("src.agents.memory.store.MemoryStore") as mock_memory_class
    ):
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        await manager._run_task_logic(task)

        assert task.status == "completed"
        assert task.result == "Research report"
        mock_memory.append_research_finding.assert_called()

def test_get_task(manager):
    tid = manager.create_task("desc")
    task = manager.get_task(tid)
    assert task is not None
    assert task.description == "desc"

def test_list_tasks(manager):
    manager.create_task("t1")
    manager.create_task("t2")
    tasks = manager.list_tasks()
    assert len(tasks) == 2

