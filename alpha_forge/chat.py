"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from alpha_forge.config import Config
from alpha_forge.conversation import Message


class ChatClient:
    def __init__(
        self,
        config: Config,
        *,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config
        kwargs = self._client_kwargs(config)
        self.client = client if client is not None else OpenAI(**kwargs)
        self.async_client = (
            async_client if async_client is not None else AsyncOpenAI(**kwargs)
        )

    @staticmethod
    def _client_kwargs(config: Config) -> dict[str, str]:
        kwargs: dict[str, str] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return kwargs

    def complete(self, messages: list[Message]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[message.to_openai() for message in messages],
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield content deltas as the model produces them.

        Drives OpenAI's streaming chat-completions endpoint via the
        async SDK client. Each yielded string is one token (or short
        fragment) suitable for incremental rendering. The caller is
        responsible for coalescing chunks into the final assistant
        message.
        """
        response = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[message.to_openai() for message in messages],
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return sorted(model.id for model in response.data)
