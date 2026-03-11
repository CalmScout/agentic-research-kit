"""Manager for communication channels and message routing using Async MessageBus."""

import asyncio
import traceback

from src.agents.bus import InboundMessage as BusInbound
from src.agents.bus import MessageBus
from src.agents.channels.base import Channel
from src.agents.channels.base import InboundMessage as ChanInbound
from src.agents.channels.base import OutboundMessage as ChanOutbound
from src.utils.logger import logger


class ChannelManager:
    """
    Manages multiple communication channels and routes messages to research agents
    via the Async MessageBus.
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._channels: dict[str, Channel] = {}
        self._running = False
        self._outbound_task: asyncio.Task | None = None

    def register_channel(self, channel: Channel):
        """Register a new channel."""
        self._channels[channel.name] = channel
        logger.info(f"Registered channel: {channel.name}")

    async def start(self):
        """Start all registered channels and the outbound router."""
        self._running = True

        # 1. Start outbound router task
        self._outbound_task = asyncio.create_task(self._outbound_router_loop())

        # 2. Start all channels
        tasks = []
        for channel in self._channels.values():
            tasks.append(channel.start(self._handle_inbound_from_channel))

        logger.info("Starting channel manager with MessageBus...")
        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop all registered channels and the outbound router."""
        self._running = False

        if self._outbound_task:
            self._outbound_task.cancel()
            try:
                await self._outbound_task
            except asyncio.CancelledError:
                pass

        for channel in self._channels.values():
            await channel.stop()
        logger.info("Stopped channel manager")

    async def _handle_inbound_from_channel(self, msg: ChanInbound):
        """
        Callback from channels. Translates channel-specific InboundMessage
        to a BusInbound message and pushes it to the MessageBus.
        """
        logger.info(f"Received message from {msg.channel}:{msg.sender_id} -> Queuing to Bus")

        bus_msg = BusInbound(
            session_id=msg.metadata.get("session_id", f"session_{msg.chat_id}"),
            channel=msg.channel,
            chat_id=msg.chat_id,
            user_id=msg.sender_id,
            content=msg.content,
            metadata=msg.metadata,
        )

        await self.bus.put_inbound(bus_msg)

    async def _outbound_router_loop(self):
        """
        Background loop that pulls messages from the Bus outbound queue
        and routes them back to the appropriate channel.
        """
        logger.info("Outbound router loop started")
        while self._running:
            try:
                # Pull next response from the bus
                bus_msg = await self.bus.get_outbound()

                # Find the target channel
                channel_name = bus_msg.channel
                if channel_name in self._channels:
                    channel = self._channels[channel_name]

                    # Translate to channel-specific OutboundMessage
                    chan_msg = ChanOutbound(
                        channel=channel_name,
                        chat_id=bus_msg.chat_id,
                        content=bus_msg.content,
                        metadata=bus_msg.metadata,
                    )

                    # Special handling for "thinking" status to maybe show a status or typing
                    # For Telegram, we just send it as a message for now
                    await channel.send(chan_msg)
                else:
                    logger.error(f"Outbound Router: Channel '{channel_name}' not registered")

                # Acknowledge
                self.bus.task_done_outbound()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Outbound Router Error: {e}")
                logger.debug(traceback.format_exc())
                await asyncio.sleep(1)
