import os
from pathlib import Path
from dotenv import load_dotenv


def load_env_variables():
    secret_path = Path("../.env")
    print("Looking for dev.env at:", secret_path.resolve())
    if secret_path.exists():
        load_dotenv(secret_path)
        print("Loaded environment variables from dev.env")


def get_env_variable(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} not found in environment variables.")
    return value
