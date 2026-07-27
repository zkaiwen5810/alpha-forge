"""UI-facing flat projection of durable transcript facts."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.providers.base import (
    OutputMessage,
    OutputRefusal,
    OutputText,
    ReasoningItem,
    TokenUsage,
    ToolCall,
)
from alpha_forge.transcript.events import (
    CommandCompleted,
    CommandMessage,
    InputAccepted,
    ModelOutput,
    QueryFailed,
    SessionLinked,
    ToolResult,
)
from alpha_forge.transcript.store import TranscriptStore


@dataclass(frozen=True, slots=True)
class UiPrompt:
    sequence: int
    prompt_event_id: str
    content: str


@dataclass(frozen=True, slots=True)
class UiModelOutput:
    sequence: int
    output_event_id: str
    prompt_event_id: str
    text: str | None
    reasoning: str | None
    refusal: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage | None


@dataclass(frozen=True, slots=True)
class UiToolResult:
    sequence: int
    result_event_id: str
    model_output_event_id: str
    call_id: str
    content: str
    status: str
    excluded_from_model: bool


@dataclass(frozen=True, slots=True)
class UiCommandMessage:
    sequence: int
    command_event_id: str
    message: CommandMessage


@dataclass(frozen=True, slots=True)
class UiSessionLink:
    sequence: int
    kind: str
    source_session_id: str


@dataclass(frozen=True, slots=True)
class UiQueryFailure:
    sequence: int
    prompt_event_id: str
    message: str


type UiHistoryItem = (
    UiPrompt
    | UiModelOutput
    | UiToolResult
    | UiCommandMessage
    | UiSessionLink
    | UiQueryFailure
)


class UiHistoryProjector:
    def __init__(self, transcript: TranscriptStore) -> None:
        self.transcript = transcript

    def items(self) -> list[UiHistoryItem]:
        state = self.transcript.state
        items: list[UiHistoryItem] = []
        for record in self.transcript.records:
            event = record.event
            if isinstance(event, InputAccepted) and event.kind == "prompt":
                items.append(UiPrompt(record.sequence, record.event_id, event.text))
            elif isinstance(event, ModelOutput):
                text, reasoning, refusal, calls = _model_output_parts(event)
                items.append(
                    UiModelOutput(
                        record.sequence,
                        record.event_id,
                        event.prompt_event_id,
                        text,
                        reasoning,
                        refusal,
                        calls,
                        event.usage,
                    )
                )
            elif isinstance(event, ToolResult):
                items.append(
                    UiToolResult(
                        record.sequence,
                        record.event_id,
                        event.model_output_event_id,
                        event.call_id,
                        event.content,
                        event.status,
                        not state.exchange_visibility[
                            event.model_output_event_id
                        ],
                    )
                )
            elif isinstance(event, CommandCompleted):
                items.extend(
                    UiCommandMessage(
                        record.sequence,
                        event.command_event_id,
                        message,
                    )
                    for message in event.messages
                )
            elif isinstance(event, SessionLinked):
                items.append(
                    UiSessionLink(
                        record.sequence,
                        event.kind,
                        event.source_session_id,
                    )
                )
            elif isinstance(event, QueryFailed):
                items.append(
                    UiQueryFailure(
                        record.sequence,
                        event.prompt_event_id,
                        event.message,
                    )
                )
        return items


def _model_output_parts(
    output: ModelOutput,
) -> tuple[
    str | None,
    str | None,
    str | None,
    tuple[ToolCall, ...],
]:
    text = "".join(
        part.text
        for item in output.items
        if isinstance(item, OutputMessage)
        for part in item.content
        if isinstance(part, OutputText)
    )
    refusal = "".join(
        part.refusal
        for item in output.items
        if isinstance(item, OutputMessage)
        for part in item.content
        if isinstance(part, OutputRefusal)
    )
    reasoning = "".join(
        item.content for item in output.items if isinstance(item, ReasoningItem)
    )
    calls = tuple(item for item in output.items if isinstance(item, ToolCall))
    return text or None, reasoning or None, refusal or None, calls


__all__ = [
    "UiCommandMessage",
    "UiHistoryItem",
    "UiHistoryProjector",
    "UiModelOutput",
    "UiPrompt",
    "UiQueryFailure",
    "UiSessionLink",
    "UiToolResult",
]
