"""Provider-neutral model-stream events and response accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from alpha_forge.events import Event
from alpha_forge.models import TokenUsage, ToolCall


class ModelStreamEvent(Event):
    """Base for normalized events emitted by a model stream adapter."""


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One immutable, authoritative provider response."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None
    refusal: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class TextDelta(ModelStreamEvent):
    text: str
    type: ClassVar[Literal["text_delta"]] = "text_delta"


@dataclass(frozen=True, slots=True)
class ReasoningDelta(ModelStreamEvent):
    text: str
    type: ClassVar[Literal["reasoning_delta"]] = "reasoning_delta"


@dataclass(frozen=True, slots=True)
class ToolCallDelta(ModelStreamEvent):
    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    type: ClassVar[Literal["tool_call_delta"]] = "tool_call_delta"


@dataclass(frozen=True, slots=True)
class RefusalDelta(ModelStreamEvent):
    text: str
    type: ClassVar[Literal["refusal_delta"]] = "refusal_delta"


@dataclass(frozen=True, slots=True)
class UsageUpdate(ModelStreamEvent):
    usage: TokenUsage
    type: ClassVar[Literal["usage"]] = "usage"


@dataclass(frozen=True, slots=True)
class StreamCompleted(ModelStreamEvent):
    response: ModelResponse
    type: ClassVar[Literal["completed"]] = "completed"

    @property
    def finish_reason(self) -> str | None:
        return self.response.finish_reason


type ModelDeltaEvent = (
    TextDelta | ReasoningDelta | ToolCallDelta | RefusalDelta | UsageUpdate
)
type StreamEvent = ModelDeltaEvent | StreamCompleted


@dataclass(slots=True)
class ToolCallAccumulator:
    call_id: str = ""
    name: str = ""
    arguments: str = ""

    def apply(self, event: ToolCallDelta) -> None:
        self.call_id += event.call_id
        self.name += event.name
        self.arguments += event.arguments

    def build(self) -> ToolCall:
        if not self.call_id:
            raise RuntimeError("completed tool call is missing its ID")
        if not self.name:
            raise RuntimeError("completed tool call is missing its name")
        return ToolCall(self.call_id, self.name, self.arguments)


@dataclass(slots=True)
class ModelResponseAccumulator:
    """Mutable assembly for one provider response or UI preview."""

    text: str = ""
    reasoning_content: str = ""
    tool_calls: dict[int, ToolCallAccumulator] = field(default_factory=dict)
    refusal: str = ""
    usage: TokenUsage | None = None

    def apply(self, event: ModelDeltaEvent) -> None:
        if isinstance(event, TextDelta):
            self.text += event.text
        elif isinstance(event, ReasoningDelta):
            self.reasoning_content += event.text
        elif isinstance(event, ToolCallDelta):
            self.tool_calls.setdefault(
                event.index,
                ToolCallAccumulator(),
            ).apply(event)
        elif isinstance(event, RefusalDelta):
            self.refusal += event.text
        elif isinstance(event, UsageUpdate):
            self.usage = event.usage

    def build(self, finish_reason: str | None) -> ModelResponse:
        calls = tuple(
            accumulator.build() for _, accumulator in sorted(self.tool_calls.items())
        )
        return ModelResponse(
            content=self.text or None,
            tool_calls=calls,
            reasoning_content=self.reasoning_content or None,
            refusal=self.refusal or None,
            finish_reason=finish_reason,
            usage=self.usage,
        )


__all__ = [
    "ModelDeltaEvent",
    "ModelResponse",
    "ModelResponseAccumulator",
    "ModelStreamEvent",
    "ReasoningDelta",
    "RefusalDelta",
    "StreamCompleted",
    "StreamEvent",
    "TextDelta",
    "TokenUsage",
    "ToolCallAccumulator",
    "ToolCallDelta",
    "UsageUpdate",
]
