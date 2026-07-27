"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast

from openai import AsyncOpenAI, OpenAI

from alpha_forge.config import Config
from alpha_forge.model_messages import Message
from alpha_forge.streaming import (
    ModelResponseAccumulator,
    ReasoningDelta,
    RefusalDelta,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    UsageUpdate,
)


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
    def _field(value: object, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _token_usage(
        cls,
        usage: object,
    ) -> tuple[int | None, int | None, int | None] | None:
        prompt_tokens = cls._field(usage, "prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = cls._field(usage, "input_tokens")
        total_tokens = cls._field(usage, "total_tokens")
        if total_tokens is None:
            output_tokens = cls._field(usage, "output_tokens")
            if isinstance(prompt_tokens, int) and isinstance(output_tokens, int):
                total_tokens = prompt_tokens + output_tokens

        prompt_details = cls._field(usage, "prompt_tokens_details")
        cached_tokens = cls._field(prompt_details, "cached_tokens")
        if cached_tokens is None:
            cached_tokens = cls._field(usage, "prompt_cache_hit_tokens")
        if cached_tokens is None:
            cached_tokens = cls._field(usage, "cached_tokens")

        prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) else None
        cached_tokens = cached_tokens if isinstance(cached_tokens, int) else None
        total_tokens = total_tokens if isinstance(total_tokens, int) else None
        if prompt_tokens is None and cached_tokens is None and total_tokens is None:
            return None
        return prompt_tokens, cached_tokens, total_tokens

    def complete(self, messages: list[Message]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=cast(
                Any,
                [message.to_openai() for message in messages],
            ),
        )
        return response.choices[0].message.content or ""

    async def stream_response(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield normalized deltas for one streamed model response."""
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_openai() for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tools
        response = await self.async_client.chat.completions.create(**request)
        accumulator = ModelResponseAccumulator()
        finish_reason: str | None = None
        async for chunk in response:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    event = TextDelta(delta.content)
                    accumulator.apply(event)
                    yield event

                # ``reasoning_content`` is an OpenAI-compatible extension,
                # not part of the official Chat Completions delta schema.
                reasoning = self._field(delta, "reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    event = ReasoningDelta(reasoning)
                    accumulator.apply(event)
                    yield event

                for tool_call in delta.tool_calls or []:
                    if tool_call.index is None:
                        raise RuntimeError("tool-call delta is missing its index")
                    function = tool_call.function
                    event = ToolCallDelta(
                        index=tool_call.index,
                        call_id=tool_call.id or "",
                        name=(function.name or "") if function else "",
                        arguments=((function.arguments or "") if function else ""),
                    )
                    accumulator.apply(event)
                    yield event

                refusal = self._field(delta, "refusal")
                if isinstance(refusal, str) and refusal:
                    event = RefusalDelta(refusal)
                    accumulator.apply(event)
                    yield event

                encountered_reason = getattr(choice, "finish_reason", None)
                if encountered_reason is not None:
                    finish_reason = str(encountered_reason)

            usage = getattr(chunk, "usage", None)
            token_usage = self._token_usage(usage) if usage is not None else None
            if token_usage is not None:
                prompt_tokens, cached_tokens, total_tokens = token_usage
                event = UsageUpdate(
                    TokenUsage(
                        prompt_tokens=prompt_tokens,
                        cached_tokens=cached_tokens,
                        total_tokens=total_tokens,
                    )
                )
                accumulator.apply(event)
                yield event

        # A finish reason describes why generation stopped; it does not mark
        # the transport boundary because a later choice-less usage chunk may
        # still arrive. Emit completion exactly once after clean exhaustion.
        yield StreamCompleted(accumulator.build(finish_reason))

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return sorted(model.id for model in response.data)


__all__ = ["ChatClient"]
