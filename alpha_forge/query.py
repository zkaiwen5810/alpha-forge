"""Stateless multi-round model query orchestration."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from alpha_forge.model_messages import AssistantMessage, Message, ToolMessage
from alpha_forge.models import EditedToolResult, PromptEdit, RawToolResult, ToolCall
from alpha_forge.prompt_editor import ToolResultPromptEditor
from alpha_forge.streaming import (
    ModelDeltaEvent,
    ModelResponse,
    StreamCompleted,
    StreamEvent,
)
from alpha_forge.tool_execution import ToolCallExecutor

MAX_TOOL_ROUNDS = 10


class ModelClient(Protocol):
    def stream_response(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream one provider response for explicit prompt messages."""


@dataclass(frozen=True, slots=True)
class QueryRequest:
    messages: tuple[Message, ...]
    tool_definitions: tuple[dict[str, Any], ...]
    tool_executor: ToolCallExecutor


class QueryEvent:
    """Marker for facts emitted by one stateless query run."""


@dataclass(frozen=True, slots=True)
class ModelRoundStarted(QueryEvent):
    output_id: str


@dataclass(frozen=True, slots=True)
class ModelDeltaReceived(QueryEvent):
    output_id: str
    delta: ModelDeltaEvent


@dataclass(frozen=True, slots=True)
class ModelRoundCompleted(QueryEvent):
    output_id: str
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class ToolBatchStarted(QueryEvent):
    output_id: str
    calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted(QueryEvent):
    output_id: str
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolResultProduced(QueryEvent):
    output_id: str
    result: RawToolResult


@dataclass(frozen=True, slots=True)
class ToolPreviewUpdated(QueryEvent):
    output_id: str
    results: tuple[EditedToolResult, ...]


@dataclass(frozen=True, slots=True)
class ToolResultsEdited(QueryEvent):
    output_id: str
    edit: PromptEdit
    results: tuple[EditedToolResult, ...]


@dataclass(frozen=True, slots=True)
class QueryCompleted(QueryEvent):
    output_id: str
    response: ModelResponse


type QueryStreamEvent = (
    ModelRoundStarted
    | ModelDeltaReceived
    | ModelRoundCompleted
    | ToolBatchStarted
    | ToolExecutionStarted
    | ToolResultProduced
    | ToolPreviewUpdated
    | ToolResultsEdited
    | QueryCompleted
)


class QueryEngine:
    """Run an agentic prompt without retaining conversation or session state."""

    def __init__(
        self,
        client: ModelClient,
        *,
        prompt_editor: ToolResultPromptEditor | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")
        self.client = client
        self.prompt_editor = prompt_editor or ToolResultPromptEditor()
        self.max_tool_rounds = max_tool_rounds

    async def run(self, request: QueryRequest) -> AsyncGenerator[QueryStreamEvent, None]:
        messages = list(request.messages)
        tools = list(request.tool_definitions)

        for _round_index in range(self.max_tool_rounds):
            output_id = uuid4().hex
            yield ModelRoundStarted(output_id)
            response: ModelResponse | None = None
            async with aclosing(
                self.client.stream_response(messages, tools=tools)
            ) as stream_events:
                async for event in stream_events:
                    if isinstance(event, StreamCompleted):
                        if response is not None:
                            raise RuntimeError(
                                "model stream emitted completion more than once"
                            )
                        response = event.response
                    else:
                        yield ModelDeltaReceived(output_id, event)
            if response is None:
                raise RuntimeError(
                    "model stream ended without a completed response"
                )
            yield ModelRoundCompleted(output_id, response)
            messages.append(
                AssistantMessage(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                    refusal=response.refusal,
                )
            )
            if not response.tool_calls:
                yield QueryCompleted(output_id, response)
                return

            yield ToolBatchStarted(output_id, response.tool_calls)
            raw_results: list[RawToolResult] = []
            for call in response.tool_calls:
                yield ToolExecutionStarted(output_id, call)
                executed = await request.tool_executor.execute(call)
                raw = RawToolResult(
                    result_id=uuid4().hex,
                    call_id=executed.call_id,
                    content=executed.content,
                    failed=executed.failed,
                )
                raw_results.append(raw)
                yield ToolResultProduced(output_id, raw)
                provisional = self.prompt_editor.edit(tuple(raw_results))
                yield ToolPreviewUpdated(
                    output_id,
                    self.prompt_editor.render_edit(
                        tuple(raw_results),
                        provisional,
                    ),
                )

            complete_results = tuple(raw_results)
            edit = self.prompt_editor.edit(complete_results)
            edited = self.prompt_editor.render_edit(complete_results, edit)
            yield ToolResultsEdited(output_id, edit, edited)
            messages.extend(
                ToolMessage(
                    result.content,
                    tool_call_id=result.call_id,
                    failed=result.failed,
                )
                for result in edited
            )

        raise RuntimeError(f"tool round limit reached ({self.max_tool_rounds})")


__all__ = [
    "MAX_TOOL_ROUNDS",
    "ModelClient",
    "ModelDeltaReceived",
    "ModelRoundCompleted",
    "ModelRoundStarted",
    "QueryCompleted",
    "QueryEngine",
    "QueryEvent",
    "QueryRequest",
    "QueryStreamEvent",
    "ToolBatchStarted",
    "ToolExecutionStarted",
    "ToolPreviewUpdated",
    "ToolResultProduced",
    "ToolResultsEdited",
]
