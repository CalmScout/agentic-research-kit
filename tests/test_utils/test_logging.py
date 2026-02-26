import pytest
from pathlib import Path
import os
from src.utils.logging import setup_logging, get_logger
from src.utils.logger import configure_logging

def test_setup_logging(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(console_level="DEBUG", log_dir=log_dir)
    
    assert log_dir.exists()
    assert log_dir.is_dir()
    
    logger = get_logger()
    logger.info("test message")
    
    # Check if log files were created
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) >= 1

def test_configure_logging():
    # This just calls setup_logging with defaults
    configure_logging()
    logger = get_logger()
    logger.info("test configure_logging")
    assert True
