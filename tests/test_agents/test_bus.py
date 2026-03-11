import pytest

from src.agents.bus import InboundMessage, MessageBus, OutboundMessage


@pytest.mark.asyncio
async def test_message_bus_inbound():
    bus = MessageBus()
    msg = InboundMessage(session_id="test_session", content="Hello")

    await bus.put_inbound(msg)
    received = await bus.get_inbound()

    assert received.session_id == "test_session"
    assert received.content == "Hello"
    assert received.user_id == "default_user"
    bus.task_done_inbound()


@pytest.mark.asyncio
async def test_message_bus_outbound():
    bus = MessageBus()
    msg = OutboundMessage(session_id="test_session", content="Thinking...", status="thinking")

    await bus.put_outbound(msg)
    received = await bus.get_outbound()

    assert received.session_id == "test_session"
    assert received.content == "Thinking..."
    assert received.status == "thinking"
    bus.task_done_outbound()


@pytest.mark.asyncio
async def test_message_bus_shutdown():
    bus = MessageBus()
    await bus.shutdown()

    msg = InboundMessage(session_id="test", content="fail")
    with pytest.raises(RuntimeError, match="MessageBus is not running"):
        await bus.put_inbound(msg)


@pytest.mark.asyncio
async def test_message_bus_multiple_messages():
    bus = MessageBus()
    for i in range(5):
        await bus.put_inbound(InboundMessage(session_id=f"session_{i}", content=str(i)))

    for i in range(5):
        received = await bus.get_inbound()
        assert received.content == str(i)
        bus.task_done_inbound()
