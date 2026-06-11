import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def load_local_env() -> Path:
    uv_env_file = os.environ.get("UV_ENV_FILE")
    if not uv_env_file:
        raise RuntimeError("UV_ENV_FILE must point to the env file used for local secrets.")

    env_path = Path(uv_env_file).expanduser()
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path

    if not env_path.exists():
        raise RuntimeError(f"UV_ENV_FILE points to a missing file: {env_path}")

    load_dotenv(env_path, override=False)
    return env_path


def create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the env file referenced by UV_ENV_FILE."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def print_response(response: Any) -> None:
    print(json.dumps(response.model_dump(), indent=2, sort_keys=True))
