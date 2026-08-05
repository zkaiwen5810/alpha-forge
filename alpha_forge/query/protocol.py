"""Effect/feedback protocol for one stateless, multi-round query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_forge.context.models import ModelContextSnapshot
from alpha_forge.events import Event
from alpha_forge.providers.base import (
    ProviderDelta,
    ProviderOutput,
    ToolCall,
)
from alpha_forge.tools.base import ToolSpec
from alpha_forge.tools.execution import ToolCallExecutor

type CommittedToolResultStatus = Literal["success", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class PendingIntermediateRound:
    model_output_event_id: str
    missing_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """Everything needed to continue one already-accepted prompt."""

    prompt_event_id: str
    pending_intermediate_round: PendingIntermediateRound | None
    completed_intermediate_rounds: int
    tool_specs: tuple[ToolSpec, ...]
    tool_executor: ToolCallExecutor


class QueryEvent(Event):
    """Base for effects and ephemeral progress emitted by the query engine."""


class QueryEffect(QueryEvent):
    """A requested application-side action that requires feedback."""


class QueryProgress(QueryEvent):
    """An ephemeral observation that must never be persisted as transcript data."""


@dataclass(frozen=True, slots=True)
class PrepareContext(QueryEffect):
    prompt_event_id: str


@dataclass(frozen=True, slots=True)
class CommitModelOutput(QueryEffect):
    prompt_event_id: str
    output: ProviderOutput


@dataclass(frozen=True, slots=True)
class CommitToolResult(QueryEffect):
    model_output_event_id: str
    call_id: str
    status: CommittedToolResultStatus
    content: str


@dataclass(frozen=True, slots=True)
class ProviderRequestStarted(QueryProgress):
    prompt_event_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class ProviderDeltaReceived(QueryProgress):
    request_id: str
    delta: ProviderDelta


@dataclass(frozen=True, slots=True)
class ProviderResponseCompleted(QueryProgress):
    request_id: str
    output: ProviderOutput


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted(QueryProgress):
    model_output_event_id: str
    call: ToolCall


@dataclass(frozen=True, slots=True)
class QueryCompleted(QueryProgress):
    prompt_event_id: str
    model_output_event_id: str


@dataclass(frozen=True, slots=True)
class ContextPrepared:
    snapshot: ModelContextSnapshot


@dataclass(frozen=True, slots=True)
class ModelOutputCommitted:
    output_event_id: str
    revision: int


@dataclass(frozen=True, slots=True)
class ToolResultCommitted:
    result_event_id: str
    revision: int


type QueryFeedback = ContextPrepared | ModelOutputCommitted | ToolResultCommitted
type QueryStreamEvent = (
    PrepareContext
    | CommitModelOutput
    | CommitToolResult
    | ProviderRequestStarted
    | ProviderDeltaReceived
    | ProviderResponseCompleted
    | ToolExecutionStarted
    | QueryCompleted
)
type QueryFailureStage = Literal[
    "context",
    "provider",
    "intermediate_round_limit",
    "internal",
]


class QueryExecutionError(RuntimeError):
    """A query failure classified for the durable ``query.failed`` event."""

    def __init__(self, stage: QueryFailureStage, message: str) -> None:
        self.stage: QueryFailureStage = stage
        super().__init__(message)


__all__ = [
    "CommittedToolResultStatus",
    "CommitModelOutput",
    "CommitToolResult",
    "ContextPrepared",
    "ModelOutputCommitted",
    "PendingIntermediateRound",
    "PrepareContext",
    "ProviderDeltaReceived",
    "ProviderRequestStarted",
    "ProviderResponseCompleted",
    "QueryCompleted",
    "QueryEffect",
    "QueryEvent",
    "QueryExecutionError",
    "QueryFeedback",
    "QueryProgress",
    "QueryRequest",
    "QueryStreamEvent",
    "ToolExecutionStarted",
    "ToolResultCommitted",
]
