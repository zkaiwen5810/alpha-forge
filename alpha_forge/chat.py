"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI

from alpha_forge.config import Config
from alpha_forge.conversation import Message


@dataclass(frozen=True)
class ChatStreamEvent:
    """A normalized delta or usage update from Chat Completions."""

    type: Literal["text_delta", "tool_call_delta", "usage"]
    text: str = ""
    index: int | None = None
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


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

    @staticmethod
    def _usage_field(value: object, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _token_usage(
        cls,
        usage: object,
    ) -> tuple[int | None, int | None, int | None] | None:
        prompt_tokens = cls._usage_field(usage, "prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = cls._usage_field(usage, "input_tokens")
        total_tokens = cls._usage_field(usage, "total_tokens")
        if total_tokens is None:
            output_tokens = cls._usage_field(usage, "output_tokens")
            if isinstance(prompt_tokens, int) and isinstance(output_tokens, int):
                total_tokens = prompt_tokens + output_tokens

        prompt_details = cls._usage_field(usage, "prompt_tokens_details")
        cached_tokens = cls._usage_field(prompt_details, "cached_tokens")
        if cached_tokens is None:
            cached_tokens = cls._usage_field(usage, "prompt_cache_hit_tokens")
        if cached_tokens is None:
            cached_tokens = cls._usage_field(usage, "cached_tokens")

        prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) else None
        cached_tokens = cached_tokens if isinstance(cached_tokens, int) else None
        total_tokens = total_tokens if isinstance(total_tokens, int) else None
        if prompt_tokens is None and cached_tokens is None and total_tokens is None:
            return None
        return prompt_tokens, cached_tokens, total_tokens

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
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tools
        response = await self.async_client.chat.completions.create(**request)
        async for chunk in response:
            usage = getattr(chunk, "usage", None)
            token_usage = self._token_usage(usage) if usage is not None else None
            if token_usage is not None:
                prompt_tokens, cached_tokens, total_tokens = token_usage
                yield ChatStreamEvent(
                    type="usage",
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                    total_tokens=total_tokens,
                )
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
