"""Environment configuration for the chat CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4.1-mini"


class ConfigError(Exception):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str | None = None

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "Config":
        if load_dotenv_file:
            load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("OPENAI_API_KEY is required")

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
