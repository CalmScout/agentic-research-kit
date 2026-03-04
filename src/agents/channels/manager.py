"""Manager for communication channels and message routing."""

import asyncio

from src.agents.channels.base import Channel, InboundMessage, OutboundMessage
from src.agents.workflow import query_with_agents
from src.utils.logger import logger


class ChannelManager:
    """Manages multiple communication channels and routes messages to research agents."""

    def __init__(self):
        self._channels: dict[str, Channel] = {}
        self._running = False

    def register_channel(self, channel: Channel):
        """Register a new channel."""
        self._channels[channel.name] = channel
        logger.info(f"Registered channel: {channel.name}")

    async def start(self):
        """Start all registered channels."""
        self._running = True
        tasks = []
        for channel in self._channels.values():
            tasks.append(channel.start(self._handle_inbound))

        logger.info("Starting channel manager...")
        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop all registered channels."""
        self._running = False
        for channel in self._channels.values():
            await channel.stop()
        logger.info("Stopped channel manager")

    async def _handle_inbound(self, msg: InboundMessage):
        """Route inbound messages to the research agent and send back results."""
        logger.info(f"Received message from {msg.channel}:{msg.sender_id}")

        # Call the multi-agent workflow
        try:
            # We use the query_with_agents from workflow.py
            result = await query_with_agents(msg.content)

            response_text = result.get("response", "No response generated.")

            # Add citations if available
            sources = result.get("sources", [])
            if sources:
                response_text += "\n\n**Sources:**\n"
                for i, src in enumerate(sources[:3], 1):
                    url = src.get("url", "N/A")
                    response_text += f"{i}. {url}\n"

            # Create outbound message
            outbound = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=response_text,
                metadata={"original_msg": msg.metadata},
            )

            # Send back through the same channel
            if msg.channel in self._channels:
                await self._channels[msg.channel].send(outbound)
            else:
                logger.error(f"Channel {msg.channel} not found for response")

        except Exception as e:
            logger.error(f"Error processing message in gateway: {e}")
            error_msg = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Sorry, I encountered an error: {str(e)}",
            )
            if msg.channel in self._channels:
                await self._channels[msg.channel].send(error_msg)
