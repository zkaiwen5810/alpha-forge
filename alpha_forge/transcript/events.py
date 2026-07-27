"""Durable semantic events stored in transcript schema version 1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from alpha_forge.providers.base import ModelOutputItem, TokenUsage
from alpha_forge.json_values import FrozenJsonObject

CommandLevel = Literal["notice", "error"]
CommandStatus = Literal["success", "error"]
InputKind = Literal["prompt", "command"]
SessionLinkKind = Literal["clear", "resume"]
ToolResultStatus = Literal["success", "error", "interrupted"]
QueryFailureStage = Literal[
    "context",
    "provider",
    "tool_round_limit",
    "internal",
]


@dataclass(frozen=True, slots=True)
class CommandMessage:
    content: str
    level: CommandLevel = "notice"


@dataclass(frozen=True, slots=True)
class SessionOpened:
    session_id: str
    instructions: str | None
    type: ClassVar[Literal["session.opened"]] = "session.opened"


@dataclass(frozen=True, slots=True)
class SessionLinked:
    kind: SessionLinkKind
    source_session_id: str
    source_command_event_id: str
    type: ClassVar[Literal["session.linked"]] = "session.linked"


@dataclass(frozen=True, slots=True)
class InputAccepted:
    kind: InputKind
    text: str
    command_name: str | None = None
    command_arguments: str | None = None
    type: ClassVar[Literal["input.accepted"]] = "input.accepted"


@dataclass(frozen=True, slots=True)
class CommandCompleted:
    command_event_id: str
    status: CommandStatus
    messages: tuple[CommandMessage, ...] = ()
    type: ClassVar[Literal["command.completed"]] = "command.completed"


@dataclass(frozen=True, slots=True)
class ModelOutput:
    prompt_event_id: str
    items: tuple[ModelOutputItem, ...]
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    type: ClassVar[Literal["model.output"]] = "model.output"


@dataclass(frozen=True, slots=True)
class ToolResult:
    model_output_event_id: str
    call_id: str
    status: ToolResultStatus
    content: str
    type: ClassVar[Literal["tool.result"]] = "tool.result"


@dataclass(frozen=True, slots=True, init=False)
class PolicyInvocation:
    name: str
    version: int
    parameters: FrozenJsonObject

    def __init__(
        self,
        name: str,
        version: int,
        parameters: Mapping[str, object],
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self,
            "parameters",
            FrozenJsonObject(parameters),
        )


@dataclass(frozen=True, slots=True)
class OriginalRepresentation:
    kind: ClassVar[Literal["original"]] = "original"


ToolPreviewReason = Literal[
    "individual_limit",
    "aggregate_limit",
    "individual_and_aggregate_limits",
]


@dataclass(frozen=True, slots=True)
class HeadTailPreview:
    original_chars: int
    rendered_chars: int
    head_chars: int
    tail_chars: int
    reason: ToolPreviewReason
    renderer: Literal["tool_result_preview"] = "tool_result_preview"
    renderer_version: Literal[1] = 1
    kind: ClassVar[Literal["head_tail"]] = "head_tail"


type ToolResultRepresentation = OriginalRepresentation | HeadTailPreview


@dataclass(frozen=True, slots=True)
class SetToolResultRepresentation:
    result_event_id: str
    representation: ToolResultRepresentation
    type: ClassVar[Literal["tool_result.representation_set"]] = (
        "tool_result.representation_set"
    )


@dataclass(frozen=True, slots=True)
class SetToolExchangeVisibility:
    model_output_event_id: str
    visible: bool
    type: ClassVar[Literal["tool_exchange.visibility_set"]] = (
        "tool_exchange.visibility_set"
    )


type ContextOperation = (
    SetToolResultRepresentation | SetToolExchangeVisibility
)


@dataclass(frozen=True, slots=True)
class ContextEdited:
    policy: PolicyInvocation
    operations: tuple[ContextOperation, ...]
    type: ClassVar[Literal["context.edited"]] = "context.edited"


@dataclass(frozen=True, slots=True)
class QueryFailed:
    prompt_event_id: str
    stage: QueryFailureStage
    message: str
    type: ClassVar[Literal["query.failed"]] = "query.failed"


type TranscriptEvent = (
    SessionOpened
    | SessionLinked
    | InputAccepted
    | CommandCompleted
    | ModelOutput
    | ToolResult
    | ContextEdited
    | QueryFailed
)


__all__ = [
    "CommandCompleted",
    "CommandLevel",
    "CommandMessage",
    "CommandStatus",
    "ContextEdited",
    "ContextOperation",
    "HeadTailPreview",
    "InputAccepted",
    "InputKind",
    "ModelOutput",
    "OriginalRepresentation",
    "PolicyInvocation",
    "QueryFailed",
    "QueryFailureStage",
    "SessionLinked",
    "SessionLinkKind",
    "SessionOpened",
    "SetToolExchangeVisibility",
    "SetToolResultRepresentation",
    "ToolPreviewReason",
    "ToolResult",
    "ToolResultRepresentation",
    "ToolResultStatus",
    "TranscriptEvent",
]
