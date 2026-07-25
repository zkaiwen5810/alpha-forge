"""Typed system-originated events consumed by presentation state."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.events import Event
from alpha_forge.models import ToolCall
from alpha_forge.transcript import (
    ModelOutput,
    ToolLimitDecision,
    ToolResult,
    Transcript,
)


class SystemEvent(Event):
    """Base for application facts that do not originate in an LLM stream."""


@dataclass(frozen=True, slots=True)
class SessionSelected(SystemEvent):
    transcript: Transcript
    head_turn_id: str | None


@dataclass(frozen=True, slots=True)
class TranscriptUpdated(SystemEvent):
    head_turn_id: str | None


@dataclass(frozen=True, slots=True)
class ModelResponseStarted(SystemEvent):
    turn_id: str


@dataclass(frozen=True, slots=True)
class AssistantMessageAdded(SystemEvent):
    output: ModelOutput
    head_turn_id: str | None


@dataclass(frozen=True, slots=True)
class AssistantMessageAddFailed(SystemEvent):
    message: str


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
    results: tuple[ToolResult, ...]
    decisions: tuple[ToolLimitDecision, ...]


@dataclass(frozen=True, slots=True)
class ToolResultsFinalized(SystemEvent):
    head_turn_id: str | None


@dataclass(frozen=True, slots=True)
class ToolResultsAddFailed(SystemEvent):
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


__all__ = [
    "AssistantMessageAddFailed",
    "AssistantMessageAdded",
    "ExitRequested",
    "ModelResponseStarted",
    "RequestFailed",
    "SessionSelected",
    "StatusChanged",
    "SystemEvent",
    "ToolBatchStarted",
    "ToolResultsAddFailed",
    "ToolResultsFinalized",
    "ToolResultsUpdated",
    "ToolStarted",
    "TranscriptUpdated",
]
