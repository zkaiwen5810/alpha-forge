"""Append-only, self-contained semantic activity ledger."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from uuid import uuid4

from alpha_forge.models import TokenUsage, ToolCall

SCHEMA_VERSION = 3
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")
PreviewReason = Literal[
    "individual_limit",
    "aggregate_limit",
    "individual_and_aggregate_limits",
]
CommandLevel = Literal["notice", "error"]
CommandStatus = Literal["success", "error"]
TransitionKind = Literal["clear", "resume"]


class TranscriptError(RuntimeError):
    """Base exception for transcript operations."""


class TranscriptCorruptError(TranscriptError):
    """Raised when a completed transcript record is invalid."""


class TranscriptPersistenceError(TranscriptError):
    """Raised when an event cannot be durably appended."""


@dataclass(frozen=True, slots=True)
class CommandMessage:
    content: str
    level: CommandLevel = "notice"


@dataclass(frozen=True, slots=True)
class SessionStart:
    session_id: str
    system_prompt: str | None
    type: ClassVar[Literal["session.start"]] = "session.start"


@dataclass(frozen=True, slots=True)
class SessionTransition:
    kind: TransitionKind
    source_session_id: str
    source_command_id: str
    type: ClassVar[Literal["session.transition"]] = "session.transition"


@dataclass(frozen=True, slots=True)
class UserMessage:
    turn_id: str
    parent_turn_id: str | None
    content: str
    type: ClassVar[Literal["user.message"]] = "user.message"


@dataclass(frozen=True, slots=True)
class Command:
    command_id: str
    context_turn_id: str | None
    raw: str
    name: str
    arguments: str
    type: ClassVar[Literal["user.command"]] = "user.command"


@dataclass(frozen=True, slots=True)
class CommandResult:
    command_id: str
    status: CommandStatus
    messages: tuple[CommandMessage, ...] = ()
    type: ClassVar[Literal["command.result"]] = "command.result"


@dataclass(frozen=True, slots=True)
class ModelOutput:
    output_id: str
    turn_id: str
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None
    refusal: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    type: ClassVar[Literal["model.output"]] = "model.output"


@dataclass(frozen=True, slots=True)
class ToolResult:
    result_id: str
    output_id: str
    call_id: str
    content: str
    failed: bool = False
    type: ClassVar[Literal["tool.result"]] = "tool.result"


@dataclass(frozen=True, slots=True)
class ToolLimitDecision:
    result_id: str
    call_id: str
    allocated_chars: int
    reason: PreviewReason | None = None


@dataclass(frozen=True, slots=True)
class ToolResultLimit:
    output_id: str
    policy_version: Literal["head_tail_v1"]
    individual_limit: int
    aggregate_limit: int
    decisions: tuple[ToolLimitDecision, ...]
    type: ClassVar[Literal["tool.result_limit"]] = "tool.result_limit"


@dataclass(frozen=True, slots=True)
class TurnFailure:
    turn_id: str
    error: str
    type: ClassVar[Literal["turn.failure"]] = "turn.failure"


type TranscriptEvent = (
    SessionStart
    | SessionTransition
    | UserMessage
    | Command
    | CommandResult
    | ModelOutput
    | ToolResult
    | ToolResultLimit
    | TurnFailure
)


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    sequence: int
    event_id: str
    recorded_at: str
    event: TranscriptEvent


def default_transcript_directory() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "alpha-forge" / "transcripts").resolve()


class Transcript:
    """Store and structurally validate one flat append-only activity stream."""

    def __init__(
        self,
        *,
        path: Path | None,
        records: list[TranscriptRecord] | None = None,
    ) -> None:
        self.path = path
        self._records = records or []
        self._poisoned = False

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str | None,
        session_id: str | None = None,
        path: Path | None = None,
    ) -> Transcript:
        resolved_id = session_id or uuid4().hex
        if not _SAFE_SESSION_ID.fullmatch(resolved_id):
            raise ValueError("session ID must contain only letters, digits, _ or -")
        resolved_path = (
            path.expanduser().resolve()
            if path is not None
            else default_transcript_directory() / f"{resolved_id}.jsonl"
        )
        resolved_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            fd = os.open(
                resolved_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except OSError as exc:
            raise TranscriptPersistenceError(
                f"cannot create transcript at {resolved_path}: {exc}"
            ) from exc
        os.close(fd)
        transcript = cls(path=resolved_path)
        transcript.append(SessionStart(resolved_id, system_prompt))
        return transcript

    @classmethod
    def in_memory(
        cls,
        *,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> Transcript:
        resolved_id = session_id or uuid4().hex
        if not _SAFE_SESSION_ID.fullmatch(resolved_id):
            raise ValueError("session ID must contain only letters, digits, _ or -")
        transcript = cls(path=None)
        transcript.append(SessionStart(resolved_id, system_prompt))
        return transcript

    @classmethod
    def resume(cls, path: Path) -> Transcript:
        resolved = path.expanduser().resolve()
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            raise TranscriptError(
                f"cannot read transcript at {resolved}: {exc}"
            ) from exc
        if not raw:
            raise TranscriptCorruptError("transcript is empty")

        if not raw.endswith(b"\n"):
            newline = raw.rfind(b"\n")
            if newline < 0:
                raise TranscriptCorruptError("transcript contains no completed records")
            raw = raw[: newline + 1]
            try:
                with resolved.open("r+b") as stream:
                    stream.truncate(len(raw))
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise TranscriptPersistenceError(
                    f"cannot repair transcript tail at {resolved}: {exc}"
                ) from exc

        records: list[TranscriptRecord] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                decoded = json.loads(line)
                record = _decode_record(decoded)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise TranscriptCorruptError(
                    f"invalid transcript record on line {line_number}: {exc}"
                ) from exc
            expected = len(records)
            if record.sequence != expected:
                raise TranscriptCorruptError(
                    f"transcript sequence gap on line {line_number}: "
                    f"expected {expected}, got {record.sequence}"
                )
            records.append(record)
        if not records or not isinstance(records[0].event, SessionStart):
            raise TranscriptCorruptError("first transcript event must be session.start")
        transcript = cls(path=resolved, records=records)
        transcript.validate()
        return transcript

    @property
    def records(self) -> tuple[TranscriptRecord, ...]:
        return tuple(self._records)

    @property
    def events(self) -> tuple[TranscriptEvent, ...]:
        return tuple(record.event for record in self._records)

    @property
    def revision(self) -> int:
        return len(self._records)

    @property
    def session_id(self) -> str:
        started = self._records[0].event
        assert isinstance(started, SessionStart)
        return started.session_id

    @property
    def system_prompt(self) -> str | None:
        started = self._records[0].event
        assert isinstance(started, SessionStart)
        return started.system_prompt

    def append(self, event: TranscriptEvent) -> TranscriptRecord:
        return self.append_many((event,))[0]

    def append_many(
        self,
        events: tuple[TranscriptEvent, ...],
    ) -> tuple[TranscriptRecord, ...]:
        if not events:
            return ()
        if self._poisoned:
            raise TranscriptPersistenceError(
                "transcript writer is unavailable after an earlier write failure"
            )
        recorded_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(
            TranscriptRecord(
                sequence=len(self._records) + offset,
                event_id=uuid4().hex,
                recorded_at=recorded_at,
                event=event,
            )
            for offset, event in enumerate(events)
        )
        self._records.extend(records)
        try:
            self.validate()
        except Exception:
            del self._records[-len(records) :]
            raise
        del self._records[-len(records) :]

        if self.path is not None:
            data = b"".join(
                json.dumps(
                    _encode_record(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
                for record in records
            )
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
                try:
                    view = memoryview(data)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise OSError("zero-byte transcript write")
                        view = view[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except (OSError, UnicodeError) as exc:
                self._poisoned = True
                raise TranscriptPersistenceError(
                    f"cannot append transcript at {self.path}: {exc}"
                ) from exc
        self._records.extend(records)
        return records

    def validate(self) -> None:
        if not self._records:
            raise TranscriptCorruptError("transcript is empty")
        if not isinstance(self._records[0].event, SessionStart):
            raise TranscriptCorruptError("first transcript event must be session.start")

        turns: set[str] = set()
        commands: set[str] = set()
        command_results: set[str] = set()
        event_ids: set[str] = set()
        outputs: dict[str, ModelOutput] = {}
        results: dict[str, ToolResult] = {}
        result_calls: set[tuple[str, str]] = set()
        limits: set[str] = set()

        for index, record in enumerate(self._records):
            if (
                isinstance(record.sequence, bool)
                or not isinstance(record.sequence, int)
                or record.sequence != index
            ):
                raise TranscriptCorruptError("transcript sequences are not contiguous")
            if (
                not isinstance(record.event_id, str)
                or not record.event_id
                or record.event_id in event_ids
            ):
                raise TranscriptCorruptError(
                    f"duplicate or empty event ID: {record.event_id}"
                )
            if not isinstance(record.recorded_at, str) or not record.recorded_at:
                raise TranscriptCorruptError("recorded_at cannot be empty")
            event_ids.add(record.event_id)
            event = record.event
            if isinstance(event, SessionStart):
                if index != 0:
                    raise TranscriptCorruptError("session.start must be first")
                if not _SAFE_SESSION_ID.fullmatch(event.session_id):
                    raise TranscriptCorruptError("invalid session ID")
                if event.system_prompt is not None and not isinstance(
                    event.system_prompt,
                    str,
                ):
                    raise TranscriptCorruptError(
                        "system prompt must be a string or null"
                    )
            elif isinstance(event, SessionTransition):
                if event.kind not in ("clear", "resume"):
                    raise TranscriptCorruptError(
                        f"invalid session transition kind: {event.kind}"
                    )
                if not event.source_session_id or not event.source_command_id:
                    raise TranscriptCorruptError(
                        "session transition references cannot be empty"
                    )
            elif isinstance(event, UserMessage):
                if not isinstance(event.content, str):
                    raise TranscriptCorruptError(
                        "user message content must be a string"
                    )
                if not event.turn_id or event.turn_id in turns:
                    raise TranscriptCorruptError(
                        f"duplicate or empty turn ID: {event.turn_id}"
                    )
                if (
                    event.parent_turn_id is not None
                    and event.parent_turn_id not in turns
                ):
                    raise TranscriptCorruptError(
                        f"unknown parent turn: {event.parent_turn_id}"
                    )
                turns.add(event.turn_id)
            elif isinstance(event, Command):
                if not all(
                    isinstance(value, str)
                    for value in (event.raw, event.name, event.arguments)
                ):
                    raise TranscriptCorruptError("command fields must be strings")
                if not event.command_id or event.command_id in commands:
                    raise TranscriptCorruptError(
                        f"duplicate or empty command ID: {event.command_id}"
                    )
                if (
                    event.context_turn_id is not None
                    and event.context_turn_id not in turns
                ):
                    raise TranscriptCorruptError(
                        f"unknown command context turn: {event.context_turn_id}"
                    )
                commands.add(event.command_id)
            elif isinstance(event, CommandResult):
                if event.status not in ("success", "error"):
                    raise TranscriptCorruptError(
                        f"invalid command result status: {event.status}"
                    )
                if any(
                    message.level not in ("notice", "error")
                    or not isinstance(message.content, str)
                    for message in event.messages
                ):
                    raise TranscriptCorruptError("invalid command result message")
                if event.command_id not in commands:
                    raise TranscriptCorruptError(
                        f"command result references unknown command: {event.command_id}"
                    )
                if event.command_id in command_results:
                    raise TranscriptCorruptError(
                        f"duplicate command result: {event.command_id}"
                    )
                command_results.add(event.command_id)
            elif isinstance(event, ModelOutput):
                if any(
                    value is not None and not isinstance(value, str)
                    for value in (
                        event.content,
                        event.reasoning_content,
                        event.refusal,
                        event.finish_reason,
                    )
                ):
                    raise TranscriptCorruptError(
                        "model output text fields must be strings or null"
                    )
                if event.usage is not None and any(
                    value is not None
                    and (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    )
                    for value in (
                        event.usage.prompt_tokens,
                        event.usage.cached_tokens,
                        event.usage.total_tokens,
                    )
                ):
                    raise TranscriptCorruptError(
                        "token usage values must be non-negative integers or null"
                    )
                if not event.output_id or event.output_id in outputs:
                    raise TranscriptCorruptError(
                        f"duplicate or empty model output ID: {event.output_id}"
                    )
                if event.turn_id not in turns:
                    raise TranscriptCorruptError(
                        f"model output references unknown turn: {event.turn_id}"
                    )
                call_ids = [call.id for call in event.tool_calls]
                if len(call_ids) != len(set(call_ids)):
                    raise TranscriptCorruptError(
                        "tool-call IDs must be unique within a model output"
                    )
                if any(
                    not isinstance(call.id, str)
                    or not call.id
                    or not isinstance(call.name, str)
                    or not call.name
                    or not isinstance(call.arguments, str)
                    for call in event.tool_calls
                ):
                    raise TranscriptCorruptError(
                        "tool calls require string IDs, names, and arguments"
                    )
                outputs[event.output_id] = event
            elif isinstance(event, ToolResult):
                if not isinstance(event.content, str) or not isinstance(
                    event.failed,
                    bool,
                ):
                    raise TranscriptCorruptError(
                        "tool result content/failed fields are invalid"
                    )
                output = outputs.get(event.output_id)
                if output is None:
                    raise TranscriptCorruptError(
                        f"tool result references unknown output: {event.output_id}"
                    )
                if event.call_id not in {call.id for call in output.tool_calls}:
                    raise TranscriptCorruptError(
                        f"tool result references unknown call: {event.call_id}"
                    )
                if not event.result_id or event.result_id in results:
                    raise TranscriptCorruptError(
                        f"duplicate or empty result ID: {event.result_id}"
                    )
                result_call = (event.output_id, event.call_id)
                if result_call in result_calls:
                    raise TranscriptCorruptError(
                        f"duplicate result for tool call: {event.call_id}"
                    )
                results[event.result_id] = event
                result_calls.add(result_call)
            elif isinstance(event, ToolResultLimit):
                output = outputs.get(event.output_id)
                if output is None:
                    raise TranscriptCorruptError(
                        f"tool limit references unknown output: {event.output_id}"
                    )
                if event.output_id in limits:
                    raise TranscriptCorruptError(
                        f"duplicate tool limit: {event.output_id}"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in (
                        event.individual_limit,
                        event.aggregate_limit,
                    )
                ):
                    raise TranscriptCorruptError("tool result limits must be positive")
                if event.policy_version != "head_tail_v1":
                    raise TranscriptCorruptError(
                        f"unsupported tool limit policy: {event.policy_version}"
                    )
                if [decision.call_id for decision in event.decisions] != [
                    call.id for call in output.tool_calls
                ]:
                    raise TranscriptCorruptError(
                        "tool limit decisions must match tool-call order"
                    )
                if any(
                    decision.reason
                    not in (
                        None,
                        "individual_limit",
                        "aggregate_limit",
                        "individual_and_aggregate_limits",
                    )
                    for decision in event.decisions
                ):
                    raise TranscriptCorruptError(
                        "tool limit contains an invalid preview reason"
                    )
                if any(
                    decision.result_id not in results
                    or results[decision.result_id].output_id != event.output_id
                    or results[decision.result_id].call_id != decision.call_id
                    or isinstance(decision.allocated_chars, bool)
                    or not isinstance(decision.allocated_chars, int)
                    or decision.allocated_chars < 0
                    for decision in event.decisions
                ):
                    raise TranscriptCorruptError(
                        "tool limit references an invalid result"
                    )
                limits.add(event.output_id)
            elif isinstance(event, TurnFailure):
                if not isinstance(event.error, str):
                    raise TranscriptCorruptError("turn failure error must be a string")
                if event.turn_id not in turns:
                    raise TranscriptCorruptError(
                        f"turn failure references unknown turn: {event.turn_id}"
                    )

    def result(self, result_id: str) -> ToolResult:
        for event in self.events:
            if isinstance(event, ToolResult) and event.result_id == result_id:
                return event
        raise KeyError(result_id)


def _encode_record(record: TranscriptRecord) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": record.sequence,
        "event_id": record.event_id,
        "recorded_at": record.recorded_at,
        "type": record.event.type,
        "payload": _encode_event(record.event),
    }


def _encode_event(event: TranscriptEvent) -> dict[str, Any]:
    if isinstance(event, SessionStart):
        return {
            "session_id": event.session_id,
            "system_prompt": event.system_prompt,
        }
    if isinstance(event, SessionTransition):
        return {
            "kind": event.kind,
            "source_session_id": event.source_session_id,
            "source_command_id": event.source_command_id,
        }
    if isinstance(event, UserMessage):
        return {
            "turn_id": event.turn_id,
            "parent_turn_id": event.parent_turn_id,
            "content": event.content,
        }
    if isinstance(event, Command):
        return {
            "command_id": event.command_id,
            "context_turn_id": event.context_turn_id,
            "raw": event.raw,
            "name": event.name,
            "arguments": event.arguments,
        }
    if isinstance(event, CommandResult):
        return {
            "command_id": event.command_id,
            "status": event.status,
            "messages": [
                {"content": message.content, "level": message.level}
                for message in event.messages
            ],
        }
    if isinstance(event, ModelOutput):
        return {
            "output_id": event.output_id,
            "turn_id": event.turn_id,
            "content": event.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in event.tool_calls
            ],
            "reasoning_content": event.reasoning_content,
            "refusal": event.refusal,
            "finish_reason": event.finish_reason,
            "usage": (
                {
                    "prompt_tokens": event.usage.prompt_tokens,
                    "cached_tokens": event.usage.cached_tokens,
                    "total_tokens": event.usage.total_tokens,
                }
                if event.usage is not None
                else None
            ),
        }
    if isinstance(event, ToolResult):
        return {
            "result_id": event.result_id,
            "output_id": event.output_id,
            "call_id": event.call_id,
            "content": event.content,
            "failed": event.failed,
        }
    if isinstance(event, ToolResultLimit):
        return {
            "output_id": event.output_id,
            "policy_version": event.policy_version,
            "individual_limit": event.individual_limit,
            "aggregate_limit": event.aggregate_limit,
            "decisions": [
                {
                    "result_id": decision.result_id,
                    "call_id": decision.call_id,
                    "allocated_chars": decision.allocated_chars,
                    "reason": decision.reason,
                }
                for decision in event.decisions
            ],
        }
    if isinstance(event, TurnFailure):
        return {"turn_id": event.turn_id, "error": event.error}
    raise TypeError(f"unsupported transcript event: {type(event).__name__}")


def _decode_record(raw: object) -> TranscriptRecord:
    if not isinstance(raw, dict):
        raise TypeError("record must be an object")
    record = cast(dict[str, Any], raw)
    version = record["schema_version"]
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    event_type = _required_str(record, "type")
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    event_payload = cast(dict[str, Any], payload)
    return TranscriptRecord(
        sequence=_required_int(record, "sequence"),
        event_id=_required_str(record, "event_id"),
        recorded_at=_required_str(record, "recorded_at"),
        event=_decode_event(event_type, event_payload),
    )


def _decode_event(event_type: str, payload: dict[str, Any]) -> TranscriptEvent:
    if event_type == "session.start":
        prompt = _optional_str(payload, "system_prompt")
        return SessionStart(_required_str(payload, "session_id"), prompt)
    if event_type == "session.transition":
        kind = _required_str(payload, "kind")
        if kind not in ("clear", "resume"):
            raise ValueError(f"invalid transition kind: {kind}")
        return SessionTransition(
            kind,
            _required_str(payload, "source_session_id"),
            _required_str(payload, "source_command_id"),
        )
    if event_type == "user.message":
        return UserMessage(
            _required_str(payload, "turn_id"),
            _optional_str(payload, "parent_turn_id"),
            _required_str(payload, "content"),
        )
    if event_type == "user.command":
        return Command(
            _required_str(payload, "command_id"),
            _optional_str(payload, "context_turn_id"),
            _required_str(payload, "raw"),
            _required_str(payload, "name"),
            _required_str(payload, "arguments"),
        )
    if event_type == "command.result":
        status = _required_str(payload, "status")
        if status not in ("success", "error"):
            raise ValueError(f"invalid command status: {status}")
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise TypeError("messages must be an array")
        messages: list[CommandMessage] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise TypeError("command message must be an object")
            level = raw_message.get("level", "notice")
            if level not in ("notice", "error"):
                raise ValueError(f"invalid command message level: {level}")
            messages.append(
                CommandMessage(
                    _required_str(raw_message, "content"),
                    level,
                )
            )
        return CommandResult(
            _required_str(payload, "command_id"),
            status,
            tuple(messages),
        )
    if event_type == "model.output":
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list):
            raise TypeError("tool_calls must be an array")
        calls: list[ToolCall] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise TypeError("tool call must be an object")
            calls.append(
                ToolCall(
                    _required_str(raw_call, "id"),
                    _required_str(raw_call, "name"),
                    _required_str(raw_call, "arguments"),
                )
            )
        raw_usage = payload.get("usage")
        usage = None
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                raise TypeError("usage must be an object or null")
            usage = TokenUsage(
                _optional_int(raw_usage, "prompt_tokens"),
                _optional_int(raw_usage, "cached_tokens"),
                _optional_int(raw_usage, "total_tokens"),
            )
        return ModelOutput(
            output_id=_required_str(payload, "output_id"),
            turn_id=_required_str(payload, "turn_id"),
            content=_optional_str(payload, "content"),
            tool_calls=tuple(calls),
            reasoning_content=_optional_str(payload, "reasoning_content"),
            refusal=_optional_str(payload, "refusal"),
            finish_reason=_optional_str(payload, "finish_reason"),
            usage=usage,
        )
    if event_type == "tool.result":
        failed = payload.get("failed", False)
        if not isinstance(failed, bool):
            raise TypeError("failed must be a boolean")
        return ToolResult(
            result_id=_required_str(payload, "result_id"),
            output_id=_required_str(payload, "output_id"),
            call_id=_required_str(payload, "call_id"),
            content=_required_str(payload, "content"),
            failed=failed,
        )
    if event_type == "tool.result_limit":
        policy = _required_str(payload, "policy_version")
        if policy != "head_tail_v1":
            raise ValueError(f"unsupported tool limit policy: {policy}")
        raw_decisions = payload.get("decisions")
        if not isinstance(raw_decisions, list):
            raise TypeError("decisions must be an array")
        decisions: list[ToolLimitDecision] = []
        for raw in raw_decisions:
            if not isinstance(raw, dict):
                raise TypeError("tool limit decision must be an object")
            reason = raw.get("reason")
            if reason not in (
                None,
                "individual_limit",
                "aggregate_limit",
                "individual_and_aggregate_limits",
            ):
                raise ValueError(f"invalid preview reason: {reason}")
            decisions.append(
                ToolLimitDecision(
                    _required_str(raw, "result_id"),
                    _required_str(raw, "call_id"),
                    _required_int(raw, "allocated_chars"),
                    reason,
                )
            )
        return ToolResultLimit(
            output_id=_required_str(payload, "output_id"),
            policy_version="head_tail_v1",
            individual_limit=_required_int(payload, "individual_limit"),
            aggregate_limit=_required_int(payload, "aggregate_limit"),
            decisions=tuple(decisions),
        )
    if event_type == "turn.failure":
        return TurnFailure(
            _required_str(payload, "turn_id"),
            _required_str(payload, "error"),
        )
    raise ValueError(f"unsupported transcript event type: {event_type}")


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_str(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError(f"{key} must be an integer or null")
    return value


__all__ = [
    "SCHEMA_VERSION",
    "Command",
    "CommandLevel",
    "CommandMessage",
    "CommandResult",
    "CommandStatus",
    "ModelOutput",
    "PreviewReason",
    "SessionStart",
    "SessionTransition",
    "TokenUsage",
    "ToolLimitDecision",
    "ToolResult",
    "ToolResultLimit",
    "Transcript",
    "TranscriptCorruptError",
    "TranscriptError",
    "TranscriptEvent",
    "TranscriptPersistenceError",
    "TranscriptRecord",
    "TransitionKind",
    "TurnFailure",
    "UserMessage",
    "default_transcript_directory",
]
