"""Application events consumed by reactive presentation adapters."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.events import Event
from alpha_forge.hooks import PreToolExecution
from alpha_forge.projectors.ui_history import UiHistoryItem
from alpha_forge.providers.base import ToolCall
from alpha_forge.query.protocol import (
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
)


@dataclass(frozen=True, slots=True)
class SessionView:
    session_id: str
    revision: int
    items: tuple[UiHistoryItem, ...]


class ApplicationEvent(Event):
    """Base for coordinator-to-presentation facts."""


@dataclass(frozen=True, slots=True)
class SessionViewChanged(ApplicationEvent):
    view: SessionView
    reset_active: bool = False


@dataclass(frozen=True, slots=True)
class InputQueued(ApplicationEvent):
    item_id: str
    raw: str


@dataclass(frozen=True, slots=True)
class InputStarted(ApplicationEvent):
    item_id: str


@dataclass(frozen=True, slots=True)
class ModelOutputRecorded(ApplicationEvent):
    output_event_id: str


@dataclass(frozen=True, slots=True)
class ToolStarted(ApplicationEvent):
    model_output_event_id: str
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolResultRecorded(ApplicationEvent):
    result_event_id: str
    model_output_event_id: str
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolPermissionRequested(ApplicationEvent):
    request_id: str
    event: PreToolExecution


@dataclass(frozen=True, slots=True)
class ToolPermissionResolved(ApplicationEvent):
    request_id: str
    allowed: bool


@dataclass(frozen=True, slots=True)
class PersistenceFailed(ApplicationEvent):
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class RequestFailed(ApplicationEvent):
    message: str


@dataclass(frozen=True, slots=True)
class StatusChanged(ApplicationEvent):
    message: str


@dataclass(frozen=True, slots=True)
class ExitRequested(ApplicationEvent):
    pass


@dataclass(frozen=True, slots=True)
class ExitReady(ApplicationEvent):
    exit_code: int = 0


__all__ = [
    "ApplicationEvent",
    "ExitReady",
    "ExitRequested",
    "InputQueued",
    "InputStarted",
    "ModelOutputRecorded",
    "PersistenceFailed",
    "ProviderDeltaReceived",
    "ProviderRequestStarted",
    "ProviderResponseCompleted",
    "RequestFailed",
    "SessionView",
    "SessionViewChanged",
    "StatusChanged",
    "ToolResultRecorded",
    "ToolPermissionRequested",
    "ToolPermissionResolved",
    "ToolStarted",
]
