"""Structured logging configuration using Loguru.

Standardizes logging across the application, supporting both text (dev)
and JSON (prod) formats with automatic rotation and retention.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .config import get_settings

# Global flag to track initialization state
_logging_initialized = False


def setup_logging() -> None:
    """Configure Loguru for application-wide logging based on settings."""
    global _logging_initialized

    if _logging_initialized:
        return

    settings = get_settings()

    # Create log directory
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # -----------------------------------------------------------------
    # Console handler
    # -----------------------------------------------------------------
    if settings.log_format == "json":
        # For production/structured environments
        logger.add(
            sys.stdout,
            format="{message}",
            serialize=True,
            level=settings.log_level.upper(),
        )
    else:
        # Colorful text format for development
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            level=settings.log_level.upper(),
            colorize=True,
        )

    # -----------------------------------------------------------------
    # File handlers (Persistent logs)
    # -----------------------------------------------------------------
    # General log file (all levels from settings.log_level and up)
    logger.add(
        log_dir / "ark_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level.upper(),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    # Error-only log file (Always capture errors separately)
    logger.add(
        log_dir / "ark_errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    _logging_initialized = True
    logger.debug("logging_initialized", log_dir=str(log_dir), log_level=settings.log_level)


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance.

    Args:
        name: Optional name for the logger (unused by loguru directly
              but kept for API compatibility with standard logging).

    Returns:
        Configured loguru logger instance
    """
    return logger
