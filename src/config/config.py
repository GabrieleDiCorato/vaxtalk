import os
from pathlib import Path
from dotenv import load_dotenv
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def load_env_variables(env_path : Path | str) -> None:
    logger.info("Looking for .env at: %s", env_path)
    if Path(env_path).exists():
        load_dotenv(env_path)
        logger.info("Loaded environment variables from .env")
    else:
        # Try loading from current directory
        logger.warning(".env file not found at specified path. Trying find_dotenv...")
        found = load_dotenv()
        if not found:
            logger.warning("No .env file found!")


def get_env_variable(key: str, default: str | None = None) -> str:
    """Get environment variable with optional default value."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"{key} not found in environment variables.")
    return value


def get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer with default value."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got: {value}")


def get_env_list(key: str, default: list[int]) -> list[int]:
    """Get environment variable as list of integers with default value."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return [int(x.strip()) for x in value.split(',')]
    except ValueError:
        raise ValueError(f"{key} must be comma-separated integers, got: {value}")
