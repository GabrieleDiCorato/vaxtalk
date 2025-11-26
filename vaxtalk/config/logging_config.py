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

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger to capture logs (including from external libraries)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Get the application logger for the project
    logger = logging.getLogger("vaxtalk")
    logger.setLevel(numeric_level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s"
    )

    # Console handler (outputs to stdout) - only show INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)

    # File handler (outputs to file) - capture DEBUG and above for all loggers
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)

    # Add handlers to root logger to capture all library logs
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set specific log levels for noisy libraries to reduce clutter
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.INFO)

    # Pre-configure Google and ADK library loggers to ensure they're captured
    # even if they haven't been imported yet
    google_loggers = [
        "google",
        "google.adk",
        "google.genai",
        "google.generativeai",
        "google_llm",
        "adk",
        "genai"
    ]

    for logger_name in google_loggers:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(numeric_level)
        lib_logger.propagate = True  # Ensure logs propagate to root

    # Also enable any existing loggers that match these patterns
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        if any(logger_name.startswith(prefix) for prefix in google_loggers):
            logging.getLogger(logger_name).setLevel(numeric_level)
            logging.getLogger(logger_name).propagate = True

    # Ensure uvicorn and FastAPI loggers write to the same file handler so
    # the on-disk log mirrors console output without duplicating console lines.
    for uvicorn_logger_name in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "starlette",
    ]:
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.setLevel(numeric_level)
        uvicorn_logger.addHandler(file_handler)

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
