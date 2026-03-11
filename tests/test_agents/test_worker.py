from unittest.mock import MagicMock, patch

import pytest

from src.agents.bus import InboundMessage, MessageBus
from src.agents.worker import AgentWorker


@pytest.mark.asyncio
async def test_worker_processing_success():
    bus = MessageBus()
    worker = AgentWorker(bus)

    # Mock astream to yield node events
    async def mock_astream(*args, **kwargs):
        yield {"enhanced_retriever": {"entities": ["test_entity"], "retrieved_docs": []}}
        yield {"enhanced_response_generator": {"response": "Drafting..."}}
        yield {"verification_agent": {"verification_status": "verified", "response": "Test response"}}

    worker._workflow_app.astream = MagicMock(side_effect=mock_astream)

    with patch("src.agents.worker.MemoryStore") as mock_memory_class:
        mock_memory = MagicMock()
        mock_memory.get_research_context.return_value = ""
        mock_memory_class.return_value = mock_memory

        await worker.start()

        # Push message
        await bus.put_inbound(InboundMessage(
            session_id="s1",
            channel="test_chan",
            chat_id="c1",
            content="test query"
        ))

        # 1. outbound (starting)
        m1 = await bus.get_outbound()
        assert m1.status == "thinking"
        assert "Starting" in m1.content
        bus.task_done_outbound()

        # 2. outbound (retriever)
        m2 = await bus.get_outbound()
        assert "Researcher" in m2.content
        bus.task_done_outbound()

        # 3. outbound (generator)
        m3 = await bus.get_outbound()
        assert "Synthesizer" in m3.content
        bus.task_done_outbound()

        # 4. outbound (verification)
        m4 = await bus.get_outbound()
        assert "Critic" in m4.content
        bus.task_done_outbound()

        # 5. outbound (complete)
        complete = await bus.get_outbound()
        assert complete.status == "complete"
        assert "Test response" in complete.content
        bus.task_done_outbound()

        await worker.stop()


@pytest.mark.asyncio
async def test_worker_processing_error():
    bus = MessageBus()
    worker = AgentWorker(bus)

    # Mock astream to raise exception
    async def mock_astream_fail(*args, **kwargs):
        raise Exception("Workflow failed")
        yield {} # satisfy generator

    worker._workflow_app.astream = MagicMock(side_effect=mock_astream_fail)

    with patch("src.agents.worker.MemoryStore") as mock_memory_class:
        mock_memory = MagicMock()
        mock_memory.get_research_context.return_value = ""
        mock_memory_class.return_value = mock_memory

        await worker.start()
        await bus.put_inbound(InboundMessage(
            session_id="s2",
            channel="test_chan",
            chat_id="c2",
            content="test query"
        ))

        # 1. outbound (starting)
        await bus.get_outbound()
        bus.task_done_outbound()

        # 2. outbound (error)
        error = await bus.get_outbound()
        assert error.status == "error"
        assert "Workflow failed" in error.content
        bus.task_done_outbound()

        await worker.stop()
