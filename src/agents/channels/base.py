"""Base class for communication channels."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InboundMessage:
    """Represents a message coming from a channel."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """Represents a message being sent to a channel."""

    channel: str
    chat_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Channel(ABC):
    """Abstract base class for all communication channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name (e.g., 'slack', 'telegram')."""
        pass

    @abstractmethod
    async def start(self, on_message: Callable[[InboundMessage], Awaitable[None]]):
        """Start the channel and listen for messages."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the channel."""
        pass

    @abstractmethod
    async def send(self, message: OutboundMessage):
        """Send a message through the channel."""
        pass
