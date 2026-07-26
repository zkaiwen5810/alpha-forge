"""Typed application events consumed by reactive presentation state."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.events import Event
from alpha_forge.models import EditedToolResult, ToolCall
from alpha_forge.streaming import ModelResponse
from alpha_forge.ui_history import UiHistoryItem


@dataclass(frozen=True, slots=True)
class SessionView:
    """Immutable durable-history projection published after a WAL commit."""

    session_id: str
    revision: int
    head_turn_id: str | None
    items: tuple[UiHistoryItem, ...]


class SystemEvent(Event):
    """Base for application facts that do not originate in a model stream."""


@dataclass(frozen=True, slots=True)
class SessionViewChanged(SystemEvent):
    view: SessionView
    reset_active: bool = False


@dataclass(frozen=True, slots=True)
class InputQueued(SystemEvent):
    item_id: str
    raw: str


@dataclass(frozen=True, slots=True)
class InputStarted(SystemEvent):
    item_id: str


@dataclass(frozen=True, slots=True)
class ModelResponseStarted(SystemEvent):
    turn_id: str
    output_id: str


@dataclass(frozen=True, slots=True)
class ModelResponseCompleted(SystemEvent):
    output_id: str
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class AssistantMessageAdded(SystemEvent):
    output_id: str


@dataclass(frozen=True, slots=True)
class ToolBatchStarted(SystemEvent):
    turn_id: str
    output_id: str
    calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class ToolStarted(SystemEvent):
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolResultsUpdated(SystemEvent):
    results: tuple[EditedToolResult, ...]


@dataclass(frozen=True, slots=True)
class ToolResultsFinalized(SystemEvent):
    output_id: str


@dataclass(frozen=True, slots=True)
class PersistenceFailed(SystemEvent):
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class RequestFailed(SystemEvent):
    message: str


@dataclass(frozen=True, slots=True)
class StatusChanged(SystemEvent):
    message: str


@dataclass(frozen=True, slots=True)
class ExitRequested(SystemEvent):
    pass


@dataclass(frozen=True, slots=True)
class ExitReady(SystemEvent):
    exit_code: int = 0


__all__ = [
    "AssistantMessageAdded",
    "ExitReady",
    "ExitRequested",
    "InputQueued",
    "InputStarted",
    "ModelResponseCompleted",
    "ModelResponseStarted",
    "PersistenceFailed",
    "RequestFailed",
    "SessionView",
    "SessionViewChanged",
    "StatusChanged",
    "SystemEvent",
    "ToolBatchStarted",
    "ToolResultsFinalized",
    "ToolResultsUpdated",
    "ToolStarted",
]
