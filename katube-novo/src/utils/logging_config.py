"""
Logging Configuration for Katube
Standardized logging setup without emojis
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Log format without emojis
LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
        console: Enable console logging (default: True)
    """
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance with standard configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Setup default logging on module import
if not logging.getLogger().handlers:
    setup_logging()
