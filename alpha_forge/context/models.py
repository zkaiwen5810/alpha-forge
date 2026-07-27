"""Immutable provider-neutral model-context projection values."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.providers.base import ModelOutputItem, ToolCall
from alpha_forge.transcript.events import (
    ToolResultRepresentation,
    ToolResultStatus,
)


@dataclass(frozen=True, slots=True)
class SystemMessage:
    content: str


@dataclass(frozen=True, slots=True)
class UserMessage:
    prompt_event_id: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelOutputContext:
    output_event_id: str
    prompt_event_id: str
    items: tuple[ModelOutputItem, ...]

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(item for item in self.items if isinstance(item, ToolCall))


@dataclass(frozen=True, slots=True)
class ToolResultContext:
    result_event_id: str
    model_output_event_id: str
    call_id: str
    status: ToolResultStatus
    content: str
    original_chars: int
    representation: ToolResultRepresentation


type ModelContextItem = (
    SystemMessage | UserMessage | ModelOutputContext | ToolResultContext
)


@dataclass(frozen=True, slots=True)
class ModelContextSnapshot:
    revision: int
    items: tuple[ModelContextItem, ...]


__all__ = [
    "ModelContextItem",
    "ModelContextSnapshot",
    "ModelOutputContext",
    "SystemMessage",
    "ToolResultContext",
    "UserMessage",
]
