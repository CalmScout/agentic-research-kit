"""
Async message bus for decoupling external channels from the LangGraph core.
Adapts patterns from nanobot for a local-first, scalable architecture.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from src.utils.logger import logger


@dataclass(kw_only=True)
class Message:
    """Base class for all messages crossing the bus."""

    session_id: str
    channel: str = "cli"
    chat_id: str = "default"
    user_id: str = "default_user"  # Pre-configured for multi-tenant isolation
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now(UTC).timestamp())


@dataclass(kw_only=True)
class InboundMessage(Message):
    """
    Message received from a channel (Telegram, CLI, Cron) intended for the agent.
    """

    content: str
    content_type: Literal["text", "image", "command", "event"] = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class OutboundMessage(Message):
    """
    Message emitted by the agent (LangGraph) intended for a channel.
    Supports streaming states ("thinking", "partial_update").
    """

    content: str
    status: Literal["thinking", "partial_update", "complete", "error"] = "complete"
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """
    Asynchronous message bus decoupling the LangGraph executor from communication channels.
    Currently utilizes standard asyncio.Queue for local operation, but designed
    to be easily swappable with Redis/RabbitMQ for distributed scaling.
    """

    def __init__(self, max_size: int = 1000):
        self._inbound_queue: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=max_size)
        self._outbound_queue: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=max_size)
        self._is_running: bool = True
        logger.info("Initialized Async MessageBus")

    async def put_inbound(self, message: InboundMessage) -> None:
        """Push a message from a channel to the agent."""
        if not self._is_running:
            raise RuntimeError("MessageBus is not running.")
        await self._inbound_queue.put(message)
        logger.debug(f"Bus: Inbound message queued (session: {message.session_id})")

    async def get_inbound(self) -> InboundMessage:
        """Agent pulls the next message to process."""
        return await self._inbound_queue.get()

    def task_done_inbound(self) -> None:
        """Acknowledge processing of an inbound message."""
        self._inbound_queue.task_done()

    async def put_outbound(self, message: OutboundMessage) -> None:
        """Agent pushes a state update or final response to the channel."""
        if not self._is_running:
            raise RuntimeError("MessageBus is not running.")
        await self._outbound_queue.put(message)
        # Only log completion or errors to avoid spamming "thinking" updates
        if message.status in ("complete", "error"):
            logger.debug(
                f"Bus: Outbound message queued (session: {message.session_id}, status: {message.status})"
            )

    async def get_outbound(self) -> OutboundMessage:
        """Channel pulls the next update to send to the user."""
        return await self._outbound_queue.get()

    def task_done_outbound(self) -> None:
        """Acknowledge processing of an outbound message."""
        self._outbound_queue.task_done()

    async def shutdown(self) -> None:
        """Gracefully shutdown the bus."""
        self._is_running = False
        logger.info("MessageBus shutdown initiated")
