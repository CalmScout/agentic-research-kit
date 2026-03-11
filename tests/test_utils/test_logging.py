from unittest.mock import patch

from src.utils.config import get_settings
from src.utils.logger import get_logger, setup_logging


def test_setup_logging(tmp_path):
    # Setup test settings
    settings = get_settings()
    log_dir = tmp_path / "logs"
    settings.log_dir = str(log_dir)
    settings.log_level = "DEBUG"

    # Initialize logging - reset flag to ensure it actually runs
    with patch("src.utils.logger._logging_initialized", False):
        setup_logging()

    assert log_dir.exists()
    assert log_dir.is_dir()

    logger = get_logger()
    logger.info("test message")

    # Check if log files were created
    # Note: Loguru might take a moment to flush to disk or create files
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) >= 1

def test_get_logger():
    logger = get_logger("test_name")
    assert logger is not None
    # In our implementation, get_logger returns the loguru.logger singleton
    logger.info("test get_logger")
