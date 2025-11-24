"""
Logging Configuration for VaxTalk Assistant

This module provides centralized logging configuration for the entire project.
Logs are output to both console and a file in the cache directory.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: Path | str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the VaxTalk application.

    Creates a logger that outputs to both console and a rotating log file.
    Log files are stored in the specified directory with timestamps.

    Args:
        log_dir: Directory where log files will be stored
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance for the application
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"vaxtalk_{timestamp}.log"

    # Get the root logger for the project
    logger = logging.getLogger("vaxtalk")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s"
    )

    # Console handler (outputs to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (outputs to file)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    logger.info("Logging initialized - Log file: %s", log_file)

    return logger


def get_logger(name: str = "vaxtalk") -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the logger (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
