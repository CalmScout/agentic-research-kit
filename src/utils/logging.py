"""Logging configuration using Loguru for application-level logging.

Complements Phoenix (Arize AI) distributed tracing with structured logging.
- Phoenix: Distributed tracing, LLM observability, vector DB visualization
- Loguru: Application logging, error tracking, audit logs

Usage:
    >>> from src.utils.logging import setup_logging, logger
    >>> setup_logging()
    >>> logger.info("application_started", version="1.0.0")
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    console_level: str = "INFO",
    log_dir: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure Loguru for application logging.

    Args:
        console_level: Minimum level for console output (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (default: ./logs)
        rotation: Log file rotation size
        retention: Log file retention period

    Example:
        >>> setup_logging(console_level="DEBUG")
    """
    # Default log directory
    log_dir = log_dir or Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # -----------------------------------------------------------------
    # Console handler with colors
    # -----------------------------------------------------------------
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=console_level,
        colorize=True,
    )

    # -----------------------------------------------------------------
    # General log file (all levels)
    # -----------------------------------------------------------------
    logger.add(
        log_dir / "ark_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # -----------------------------------------------------------------
    # Error-only log file
    # -----------------------------------------------------------------
    logger.add(
        log_dir / "ark_errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Log startup
    logger.info("logging_initialized", log_dir=str(log_dir), console_level=console_level)


def get_logger():
    """Get configured logger instance.

    Returns:
        Logger instance ready for use

    Example:
        >>> logger = get_logger()
        >>> logger.info("event_occurred", detail="value")
    """
    return logger
