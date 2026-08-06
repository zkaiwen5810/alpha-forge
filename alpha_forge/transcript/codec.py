"""Strict JSON encoding and decoding for transcript schema version 1."""

from __future__ import annotations

from typing import Any, cast

from alpha_forge.json_values import thaw_json
from alpha_forge.providers.base import (
    ModelOutputItem,
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
    ContextEdited,
    ContextOperation,
    HeadTailPreview,
    InputAccepted,
    ModelOutput,
    OriginalRepresentation,
    PolicyInvocation,
    QueryFailed,
    SessionLinked,
    SessionOpened,
    SetToolExchangeVisibility,
    SetToolResultRepresentation,
    ToolResult,
    ToolResultRepresentation,
    TranscriptEvent,
)
from alpha_forge.transcript.records import SCHEMA_VERSION, TranscriptRecord


def encode_record(record: TranscriptRecord) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": record.sequence,
        "event_id": record.event_id,
        "recorded_at": record.recorded_at,
        "type": record.event.type,
        "payload": _encode_event(record.event),
    }


def decode_record(raw: object) -> TranscriptRecord:
    mapping = _mapping(raw, "record")
    version = _int(mapping, "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported transcript schema version: {version}; "
            f"expected {SCHEMA_VERSION}"
        )
    event_type = _str(mapping, "type")
    payload = _mapping(mapping.get("payload"), "payload")
    return TranscriptRecord(
        sequence=_int(mapping, "sequence"),
        event_id=_str(mapping, "event_id"),
        recorded_at=_str(mapping, "recorded_at"),
        event=_decode_event(event_type, payload),
    )


def _encode_event(event: TranscriptEvent) -> dict[str, Any]:
    if isinstance(event, SessionOpened):
        return {
            "session_id": event.session_id,
            "instructions": event.instructions,
        }
    if isinstance(event, SessionLinked):
        return {
            "kind": event.kind,
            "source_session_id": event.source_session_id,
            "source_command_event_id": event.source_command_event_id,
        }
    if isinstance(event, InputAccepted):
        command = (
            {
                "name": event.command_name,
                "arguments": event.command_arguments,
            }
            if event.kind == "command"
            else None
        )
        return {"kind": event.kind, "text": event.text, "command": command}
    if isinstance(event, CommandCompleted):
        return {
            "command_event_id": event.command_event_id,
            "status": event.status,
            "messages": [
                {"level": message.level, "content": message.content}
                for message in event.messages
            ],
        }
    if isinstance(event, ModelOutput):
        return {
            "prompt_event_id": event.prompt_event_id,
            "items": [_encode_output_item(item) for item in event.items],
            "finish_reason": event.finish_reason,
            "usage": _encode_usage(event.usage),
        }
    if isinstance(event, ToolResult):
        return {
            "model_output_event_id": event.model_output_event_id,
            "call_id": event.call_id,
            "status": event.status,
            "content": event.content,
        }
    if isinstance(event, ContextEdited):
        return {
            "policy": {
                "name": event.policy.name,
                "version": event.policy.version,
                "parameters": thaw_json(event.policy.parameters),
            },
            "operations": [
                _encode_context_operation(operation) for operation in event.operations
            ],
        }
    if isinstance(event, QueryFailed):
        return {
            "prompt_event_id": event.prompt_event_id,
            "stage": event.stage,
            "message": event.message,
        }
    raise TypeError(f"unsupported transcript event: {type(event).__name__}")


def _decode_event(event_type: str, payload: dict[str, Any]) -> TranscriptEvent:
    if event_type == "session.opened":
        return SessionOpened(
            _str(payload, "session_id"),
            _optional_str(payload, "instructions"),
        )
    if event_type == "session.linked":
        kind = _str(payload, "kind")
        if kind not in ("clear", "resume"):
            raise ValueError(f"invalid session link kind: {kind}")
        return SessionLinked(
            cast(Any, kind),
            _str(payload, "source_session_id"),
            _str(payload, "source_command_event_id"),
        )
    if event_type == "input.accepted":
        kind = _str(payload, "kind")
        if kind not in ("prompt", "command"):
            raise ValueError(f"invalid input kind: {kind}")
        command = payload.get("command")
        if kind == "prompt":
            if command is not None:
                raise ValueError("prompt input cannot contain command data")
            return InputAccepted("prompt", _str(payload, "text"))
        command_mapping = _mapping(command, "command")
        return InputAccepted(
            "command",
            _str(payload, "text"),
            _str(command_mapping, "name"),
            _str(command_mapping, "arguments"),
        )
    if event_type == "command.completed":
        status = _str(payload, "status")
        if status not in ("success", "error"):
            raise ValueError(f"invalid command status: {status}")
        raw_messages = _list(payload, "messages")
        messages: list[CommandMessage] = []
        for raw_message in raw_messages:
            message = _mapping(raw_message, "command message")
            level = _str(message, "level")
            if level not in ("notice", "error"):
                raise ValueError(f"invalid command message level: {level}")
            messages.append(CommandMessage(_str(message, "content"), cast(Any, level)))
        return CommandCompleted(
            _str(payload, "command_event_id"),
            cast(Any, status),
            tuple(messages),
        )
    if event_type == "model.output":
        return ModelOutput(
            prompt_event_id=_str(payload, "prompt_event_id"),
            items=tuple(_decode_output_item(item) for item in _list(payload, "items")),
            finish_reason=_optional_str(payload, "finish_reason"),
            usage=_decode_usage(payload.get("usage")),
        )
    if event_type == "tool.result":
        status = _str(payload, "status")
        if status not in ("success", "error", "interrupted"):
            raise ValueError(f"invalid tool-result status: {status}")
        return ToolResult(
            _str(payload, "model_output_event_id"),
            _str(payload, "call_id"),
            cast(Any, status),
            _str(payload, "content"),
        )
    if event_type == "context.edited":
        raw_policy = _mapping(payload.get("policy"), "policy")
        raw_parameters = _mapping(raw_policy.get("parameters"), "policy parameters")
        policy = PolicyInvocation(
            _str(raw_policy, "name"),
            _int(raw_policy, "version"),
            raw_parameters,
        )
        operations = tuple(
            _decode_context_operation(operation)
            for operation in _list(payload, "operations")
        )
        return ContextEdited(policy, operations)
    if event_type == "query.failed":
        stage = _str(payload, "stage")
        if stage not in (
            "context",
            "provider",
            "intermediate_round_limit",
            "internal",
        ):
            raise ValueError(f"invalid query failure stage: {stage}")
        return QueryFailed(
            _str(payload, "prompt_event_id"),
            cast(Any, stage),
            _str(payload, "message"),
        )
    raise ValueError(f"unknown transcript event type: {event_type}")


def _encode_output_item(item: ModelOutputItem) -> dict[str, Any]:
    if isinstance(item, OutputMessage):
        return {
            "type": item.type,
            "content": [
                (
                    {"type": part.type, "text": part.text}
                    if isinstance(part, OutputText)
                    else {"type": part.type, "refusal": part.refusal}
                )
                for part in item.content
            ],
        }
    if isinstance(item, ReasoningItem):
        return {"type": item.type, "content": item.content}
    if isinstance(item, ToolCall):
        return {
            "type": item.type,
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }
    raise TypeError(f"unsupported output item: {type(item).__name__}")


def _decode_output_item(raw: object) -> ModelOutputItem:
    item = _mapping(raw, "model output item")
    item_type = _str(item, "type")
    if item_type == "message":
        content = []
        for raw_part in _list(item, "content"):
            part = _mapping(raw_part, "output content")
            part_type = _str(part, "type")
            if part_type == "output_text":
                content.append(OutputText(_str(part, "text")))
            elif part_type == "refusal":
                content.append(OutputRefusal(_str(part, "refusal")))
            else:
                raise ValueError(f"unknown output content type: {part_type}")
        return OutputMessage(tuple(content))
    if item_type == "reasoning":
        return ReasoningItem(_str(item, "content"))
    if item_type == "tool_call":
        return ToolCall(
            _str(item, "call_id"),
            _str(item, "name"),
            _str(item, "arguments"),
        )
    raise ValueError(f"unknown model output item type: {item_type}")


def _encode_usage(usage: TokenUsage | None) -> dict[str, int | None] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "cached_tokens": usage.cached_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _decode_usage(raw: object) -> TokenUsage | None:
    if raw is None:
        return None
    usage = _mapping(raw, "usage")
    return TokenUsage(
        _optional_int(usage, "input_tokens"),
        _optional_int(usage, "cached_tokens"),
        _optional_int(usage, "output_tokens"),
        _optional_int(usage, "total_tokens"),
    )


def _encode_context_operation(operation: ContextOperation) -> dict[str, Any]:
    if isinstance(operation, SetToolResultRepresentation):
        return {
            "type": operation.type,
            "result_event_id": operation.result_event_id,
            "representation": _encode_representation(operation.representation),
        }
    if isinstance(operation, SetToolExchangeVisibility):
        return {
            "type": operation.type,
            "model_output_event_id": operation.model_output_event_id,
            "visible": operation.visible,
        }
    raise TypeError(f"unsupported context operation: {type(operation).__name__}")


def _decode_context_operation(raw: object) -> ContextOperation:
    operation = _mapping(raw, "context operation")
    operation_type = _str(operation, "type")
    if operation_type == "tool_result.representation_set":
        return SetToolResultRepresentation(
            _str(operation, "result_event_id"),
            _decode_representation(operation.get("representation")),
        )
    if operation_type == "tool_exchange.visibility_set":
        return SetToolExchangeVisibility(
            _str(operation, "model_output_event_id"),
            _bool(operation, "visible"),
        )
    raise ValueError(f"unknown context operation type: {operation_type}")


def _encode_representation(
    representation: ToolResultRepresentation,
) -> dict[str, Any]:
    if isinstance(representation, OriginalRepresentation):
        return {"kind": representation.kind}
    return {
        "kind": representation.kind,
        "renderer": representation.renderer,
        "renderer_version": representation.renderer_version,
        "original_chars": representation.original_chars,
        "rendered_chars": representation.rendered_chars,
        "head_chars": representation.head_chars,
        "tail_chars": representation.tail_chars,
        "reason": representation.reason,
    }


def _decode_representation(raw: object) -> ToolResultRepresentation:
    representation = _mapping(raw, "tool-result representation")
    kind = _str(representation, "kind")
    if kind == "original":
        return OriginalRepresentation()
    if kind != "head_tail":
        raise ValueError(f"unknown tool-result representation: {kind}")
    renderer = _str(representation, "renderer")
    renderer_version = _int(representation, "renderer_version")
    reason = _str(representation, "reason")
    if renderer != "tool_result_preview" or renderer_version != 1:
        raise ValueError(
            f"unsupported tool-result renderer: {renderer} v{renderer_version}"
        )
    if reason not in (
        "individual_limit",
        "aggregate_limit",
        "individual_and_aggregate_limits",
    ):
        raise ValueError(f"invalid tool-result preview reason: {reason}")
    return HeadTailPreview(
        original_chars=_int(representation, "original_chars"),
        rendered_chars=_int(representation, "rendered_chars"),
        head_chars=_int(representation, "head_chars"),
        tail_chars=_int(representation, "tail_chars"),
        reason=cast(Any, reason),
    )


def _mapping(raw: object, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{label} must be an object with string keys")
    return cast(dict[str, Any], raw)


def _list(mapping: dict[str, Any], key: str) -> list[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be an array")
    return value


def _str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_str(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer or null")
    return value


def _bool(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


__all__ = ["decode_record", "encode_record"]
