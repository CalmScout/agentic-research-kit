import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agents.bus import MessageBus
from src.agents.bus import OutboundMessage as BusOutbound
from src.agents.channels.base import Channel
from src.agents.channels.base import InboundMessage as ChanInbound
from src.agents.channels.manager import ChannelManager


class MockChannel(Channel):
    @property
    def name(self): return "mock"
    async def start(self, on_message): self.on_message = on_message
    async def stop(self): pass
    async def send(self, message): self.sent_message = message


@pytest.mark.asyncio
async def test_channel_manager_inbound_routing():
    bus = MessageBus()
    manager = ChannelManager(bus)
    channel = MockChannel()
    manager.register_channel(channel)

    await manager.start()

    # Simulate message from channel
    chan_msg = ChanInbound(
        channel="mock",
        sender_id="u1",
        chat_id="c1",
        content="Hello bus",
        metadata={"session_id": "test_sess"}
    )

    await channel.on_message(chan_msg)

    # Check if it reached the bus
    bus_msg = await bus.get_inbound()
    assert bus_msg.content == "Hello bus"
    assert bus_msg.session_id == "test_sess"
    assert bus_msg.channel == "mock"

    await manager.stop()


@pytest.mark.asyncio
async def test_channel_manager_outbound_routing():
    bus = MessageBus()
    manager = ChannelManager(bus)
    channel = MockChannel()
    channel.send = AsyncMock()
    manager.register_channel(channel)

    await manager.start()

    # Push response to bus
    bus_out = BusOutbound(
        session_id="s1",
        channel="mock",
        chat_id="c1",
        content="Response from agent",
        status="complete"
    )
    await bus.put_outbound(bus_out)

    # Wait a bit for the background task to pick it up
    await asyncio.sleep(0.1)

    # Check if channel.send was called
    channel.send.assert_called_once()
    args, _ = channel.send.call_args
    sent_msg = args[0]
    assert sent_msg.content == "Response from agent"
    assert sent_msg.chat_id == "c1"

    await manager.stop()
