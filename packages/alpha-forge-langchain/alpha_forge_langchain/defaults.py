"""Code-defined settings for this first milestone."""

import os
from pathlib import Path

MODEL = "DeepSeek-V4-Flash"
BASE_URL = "http://litellm:4000/v1"
TIMEOUT = 30.0
MAX_MODEL_CALLS = 10
SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."


def database_path() -> Path:
    root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return root / "alpha-forge-langchain" / "conversations.sqlite3"
