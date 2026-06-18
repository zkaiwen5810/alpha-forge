"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from openai import OpenAI

from alpha_forge.config import Config
from alpha_forge.conversation import Message


class ChatClient:
    def __init__(self, config: Config, *, client: OpenAI | None = None) -> None:
        self.config = config
        self.client = client or self._build_client(config)

    @staticmethod
    def _build_client(config: Config) -> OpenAI:
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAI(**kwargs)

    def complete(self, messages: list[Message]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[message.to_openai() for message in messages],
        )
        return response.choices[0].message.content or ""
