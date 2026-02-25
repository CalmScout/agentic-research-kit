"""Telegram channel implementation for ARK."""

import re
from collections.abc import Awaitable, Callable

from loguru import logger

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from src.agents.channels.base import Channel, InboundMessage, OutboundMessage
from src.utils.config import get_settings


def _markdown_to_telegram_html(text: str) -> str:
    """Convert markdown to Telegram-safe HTML."""
    if not text:
        return ""

    # Extract and protect code blocks
    code_blocks: list[str] = []

    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", save_code_block, text)

    # Extract and protect inline code
    inline_codes: list[str] = []

    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", save_inline_code, text)

    # Headers # Title -> just the title text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # Bold **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Italic _text_
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)

    # Bullet lists - item -> • item
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    # Restore code
    for i, code in enumerate(inline_codes):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    for i, code in enumerate(code_blocks):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    return text


def _split_message(content: str, max_len: int = 4000) -> list[str]:
    """Split content into chunks."""
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        pos = cut.rfind("\n")
        if pos == -1:
            pos = cut.rfind(" ")
        if pos == -1:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


class TelegramChannel(Channel):
    """Telegram channel implementation using long polling."""

    def __init__(self):
        self.settings = get_settings()
        self._app: Application | None = None
        self._on_message_callback: Callable[[InboundMessage], Awaitable[None]] | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self, on_message: Callable[[InboundMessage], Awaitable[None]]):
        if not TELEGRAM_AVAILABLE:
            logger.error("python-telegram-bot not installed")
            return

        if not self.settings.telegram_token:
            logger.error("Telegram bot token not configured")
            return

        self._on_message_callback = on_message
        self._running = True

        # Build application
        req = HTTPXRequest(connection_pool_size=16)
        builder = Application.builder().token(self.settings.telegram_token).request(req)
        if self.settings.telegram_proxy:
            builder = builder.proxy(self.settings.telegram_proxy).get_updates_proxy(
                self.settings.telegram_proxy
            )

        self._app = builder.build()

        # Add handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_telegram_message)
        )

        logger.info("Starting Telegram bot...")
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self):
        self._running = False
        if self._app and self._app.updater:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    async def send(self, message: OutboundMessage):
        if not self._app:
            return

        for chunk in _split_message(message.content):
            try:
                html_content = _markdown_to_telegram_html(chunk)
                await self._app.bot.send_message(
                    chat_id=message.chat_id, text=html_content, parse_mode="HTML"
                )
            except Exception as e:
                logger.warning(f"Telegram send failed, retrying plain text: {e}")
                await self._app.bot.send_message(chat_id=message.chat_id, text=chunk)

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start."""
        if update.message:
            await update.message.reply_text(
                "👋 Welcome to Agentic Research Kit (ARK)!\n\n"
                "Send me a research topic or question, and I'll perform deep research using RAG and web search."
            )

    async def _handle_telegram_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        if not update.message or not update.effective_user or not update.message.text:
            return

        user = update.effective_user
        sender_id = str(user.id)
        username = user.username or ""

        # Simple ACL
        allowed = self.settings.telegram_allowed_users
        if allowed and sender_id not in allowed and username not in allowed:
            logger.warning(f"Unauthorized access attempt from {sender_id} (@{username})")
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
            return

        effective_chat = update.effective_chat
        if effective_chat is None:
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=sender_id,
            chat_id=str(effective_chat.id),
            content=update.message.text,
            metadata={"username": username, "message_id": update.message.message_id},
        )

        if self._on_message_callback:
            await self._on_message_callback(msg)
