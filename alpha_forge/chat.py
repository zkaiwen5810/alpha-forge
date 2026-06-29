"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI

from alpha_forge.config import Config
from alpha_forge.conversation import Message


@dataclass(frozen=True)
class ChatStreamEvent:
    """A normalized text or function-call delta from Chat Completions."""

    type: Literal["text_delta", "tool_call_delta"]
    text: str = ""
    index: int | None = None
    call_id: str = ""
    name: str = ""
    arguments: str = ""


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
    def _client_kwargs(config: Config) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return kwargs

    def complete(self, messages: list[Message]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[message.to_openai() for message in messages],
        )
        return response.choices[0].message.content or ""

    async def stream_response(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ChatStreamEvent]:
        """Yield normalized text and function-call deltas for one iteration."""
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_openai() for message in messages],
            "stream": True,
        }
        if tools:
            request["tools"] = tools
        response = await self.async_client.chat.completions.create(**request)
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield ChatStreamEvent(type="text_delta", text=delta.content)
            for tool_call in delta.tool_calls or []:
                function = tool_call.function
                yield ChatStreamEvent(
                    type="tool_call_delta",
                    index=tool_call.index,
                    call_id=tool_call.id or "",
                    name=(function.name or "") if function else "",
                    arguments=(function.arguments or "") if function else "",
                )

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return sorted(model.id for model in response.data)
