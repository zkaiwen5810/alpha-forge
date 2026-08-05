"""OpenAI Chat Completions adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from typing import Any

from openai import AsyncOpenAI, OpenAI

from alpha_forge.config import Config
from alpha_forge.context.models import (
    ModelContextSnapshot,
    ModelOutputContext,
    SystemMessage,
    ToolResultContext,
    UserMessage,
)
from alpha_forge.json_values import thaw_json
from alpha_forge.providers.base import (
    OutputMessage,
    OutputRefusal,
    OutputText,
    ProviderOutputAccumulator,
    ProviderStreamEvent,
    ReasoningDelta,
    ReasoningItem,
    RefusalDelta,
    StreamCompleted,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    UsageUpdate,
)
from alpha_forge.tools.base import ToolSpec


class OpenAIChatAdapter:
    """Translate provider-neutral values at the OpenAI SDK boundary."""

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

    async def stream(
        self,
        context: ModelContextSnapshot,
        *,
        tools: tuple[ToolSpec, ...],
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_message_to_openai(item) for item in context.items],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = [_tool_to_openai(tool) for tool in tools]
        response = await self.async_client.chat.completions.create(**request)
        accumulator = ProviderOutputAccumulator()
        finish_reason: str | None = None
        async for chunk in response:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    event = TextDelta(delta.content)
                    accumulator.apply(event)
                    yield event

                reasoning = _field(delta, "reasoning_content")
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

                refusal = _field(delta, "refusal")
                if isinstance(refusal, str) and refusal:
                    event = RefusalDelta(refusal)
                    accumulator.apply(event)
                    yield event

                encountered_reason = getattr(choice, "finish_reason", None)
                if encountered_reason is not None:
                    finish_reason = str(encountered_reason)

            usage = getattr(chunk, "usage", None)
            token_usage = _token_usage(usage) if usage is not None else None
            if token_usage is not None:
                event = UsageUpdate(token_usage)
                accumulator.apply(event)
                yield event

        yield StreamCompleted(accumulator.build(finish_reason))

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return sorted(model.id for model in response.data)


def _message_to_openai(item: object) -> dict[str, Any]:
    if isinstance(item, SystemMessage):
        return {"role": "system", "content": item.content}
    if isinstance(item, UserMessage):
        return {"role": "user", "content": item.content}
    if isinstance(item, ModelOutputContext):
        text = "".join(
            part.text
            for output_item in item.items
            if isinstance(output_item, OutputMessage)
            for part in output_item.content
            if isinstance(part, OutputText)
        )
        refusal = "".join(
            part.refusal
            for output_item in item.items
            if isinstance(output_item, OutputMessage)
            for part in output_item.content
            if isinstance(part, OutputRefusal)
        )
        reasoning = "".join(
            output_item.content
            for output_item in item.items
            if isinstance(output_item, ReasoningItem)
        )
        calls = tuple(
            output_item
            for output_item in item.items
            if isinstance(output_item, ToolCall)
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": text or None,
        }
        if refusal:
            message["refusal"] = refusal
        if reasoning:
            # This is an OpenAI-compatible Chat Completions extension.
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in calls
            ]
        return message
    if isinstance(item, ToolResultContext):
        return {
            "role": "tool",
            "content": item.content,
            "tool_call_id": item.call_id,
        }
    raise TypeError(f"unsupported model context item: {type(item).__name__}")


def _tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": thaw_json(tool.input_schema),
        },
    }


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _token_usage(usage: object) -> TokenUsage | None:
    input_tokens = _field(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _field(usage, "input_tokens")
    output_tokens = _field(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _field(usage, "output_tokens")
    total_tokens = _field(usage, "total_tokens")
    if (
        total_tokens is None
        and isinstance(input_tokens, int)
        and isinstance(
            output_tokens,
            int,
        )
    ):
        total_tokens = input_tokens + output_tokens

    details = _field(usage, "prompt_tokens_details")
    cached_tokens = _field(details, "cached_tokens")
    if cached_tokens is None:
        cached_tokens = _field(usage, "prompt_cache_hit_tokens")
    if cached_tokens is None:
        cached_tokens = _field(usage, "cached_tokens")

    values = (
        input_tokens if isinstance(input_tokens, int) else None,
        cached_tokens if isinstance(cached_tokens, int) else None,
        output_tokens if isinstance(output_tokens, int) else None,
        total_tokens if isinstance(total_tokens, int) else None,
    )
    if all(value is None for value in values):
        return None
    return TokenUsage(*values)


__all__ = ["OpenAIChatAdapter"]
