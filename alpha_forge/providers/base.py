"""Provider-neutral request, output, and streaming values."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

if TYPE_CHECKING:
    from alpha_forge.context.models import ModelContextSnapshot
    from alpha_forge.tools.base import ToolSpec


@dataclass(frozen=True, slots=True)
class OutputText:
    text: str
    type: ClassVar[Literal["output_text"]] = "output_text"


@dataclass(frozen=True, slots=True)
class OutputRefusal:
    refusal: str
    type: ClassVar[Literal["refusal"]] = "refusal"


type OutputContent = OutputText | OutputRefusal


@dataclass(frozen=True, slots=True)
class OutputMessage:
    content: tuple[OutputContent, ...]
    type: ClassVar[Literal["message"]] = "message"


@dataclass(frozen=True, slots=True)
class ReasoningItem:
    content: str
    type: ClassVar[Literal["reasoning"]] = "reasoning"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One complete tool call emitted by a provider."""

    call_id: str
    name: str
    arguments: str
    type: ClassVar[Literal["tool_call"]] = "tool_call"


type ModelOutputItem = OutputMessage | ReasoningItem | ToolCall


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    """One immutable, authoritative completed provider response."""

    items: tuple[ModelOutputItem, ...]
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(item for item in self.items if isinstance(item, ToolCall))

    @property
    def output_text(self) -> str | None:
        text = "".join(
            part.text
            for item in self.items
            if isinstance(item, OutputMessage)
            for part in item.content
            if isinstance(part, OutputText)
        )
        return text or None

    @property
    def reasoning(self) -> str | None:
        text = "".join(
            item.content for item in self.items if isinstance(item, ReasoningItem)
        )
        return text or None

    @property
    def refusal(self) -> str | None:
        text = "".join(
            part.refusal
            for item in self.items
            if isinstance(item, OutputMessage)
            for part in item.content
            if isinstance(part, OutputRefusal)
        )
        return text or None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str
    type: ClassVar[Literal["output_text.delta"]] = "output_text.delta"


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str
    type: ClassVar[Literal["reasoning.delta"]] = "reasoning.delta"


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    type: ClassVar[Literal["tool_call.delta"]] = "tool_call.delta"


@dataclass(frozen=True, slots=True)
class RefusalDelta:
    text: str
    type: ClassVar[Literal["refusal.delta"]] = "refusal.delta"


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    usage: TokenUsage
    type: ClassVar[Literal["usage"]] = "usage"


type ProviderDelta = (
    TextDelta | ReasoningDelta | ToolCallDelta | RefusalDelta | UsageUpdate
)


@dataclass(frozen=True, slots=True)
class StreamCompleted:
    output: ProviderOutput
    type: ClassVar[Literal["completed"]] = "completed"


type ProviderStreamEvent = ProviderDelta | StreamCompleted


@dataclass(slots=True)
class _ToolCallAccumulator:
    call_id: str = ""
    name: str = ""
    arguments: str = ""

    def apply(self, event: ToolCallDelta) -> None:
        self.call_id += event.call_id
        self.name += event.name
        self.arguments += event.arguments

    def build(self) -> ToolCall:
        if not self.call_id:
            raise RuntimeError("completed tool call is missing its call ID")
        if not self.name:
            raise RuntimeError("completed tool call is missing its name")
        return ToolCall(self.call_id, self.name, self.arguments)


@dataclass(slots=True)
class ProviderOutputAccumulator:
    """Mutable assembly used only while a provider stream is active."""

    text: str = ""
    reasoning: str = ""
    refusal: str = ""
    tool_calls: dict[int, _ToolCallAccumulator] = field(default_factory=dict)
    usage: TokenUsage | None = None

    def apply(self, event: ProviderDelta) -> None:
        if isinstance(event, TextDelta):
            self.text += event.text
        elif isinstance(event, ReasoningDelta):
            self.reasoning += event.text
        elif isinstance(event, RefusalDelta):
            self.refusal += event.text
        elif isinstance(event, ToolCallDelta):
            self.tool_calls.setdefault(
                event.index,
                _ToolCallAccumulator(),
            ).apply(event)
        elif isinstance(event, UsageUpdate):
            self.usage = event.usage

    def build(self, finish_reason: str | None) -> ProviderOutput:
        items: list[ModelOutputItem] = []
        if self.reasoning:
            items.append(ReasoningItem(self.reasoning))
        content: list[OutputContent] = []
        if self.text:
            content.append(OutputText(self.text))
        if self.refusal:
            content.append(OutputRefusal(self.refusal))
        if content:
            items.append(OutputMessage(tuple(content)))
        items.extend(
            accumulator.build()
            for _, accumulator in sorted(self.tool_calls.items())
        )
        return ProviderOutput(tuple(items), finish_reason, self.usage)


class ModelProvider(Protocol):
    def stream(
        self,
        context: ModelContextSnapshot,
        *,
        tools: tuple[ToolSpec, ...],
    ) -> AsyncGenerator[ProviderStreamEvent]:
        """Stream one provider response for a projected context snapshot."""
        ...

    def list_models(self) -> list[str]:
        """Return model IDs visible to the configured provider."""
        ...


__all__ = [
    "ModelOutputItem",
    "ModelProvider",
    "OutputContent",
    "OutputMessage",
    "OutputRefusal",
    "OutputText",
    "ProviderDelta",
    "ProviderOutput",
    "ProviderOutputAccumulator",
    "ProviderStreamEvent",
    "ReasoningDelta",
    "ReasoningItem",
    "RefusalDelta",
    "StreamCompleted",
    "TextDelta",
    "TokenUsage",
    "ToolCall",
    "ToolCallDelta",
    "UsageUpdate",
]
