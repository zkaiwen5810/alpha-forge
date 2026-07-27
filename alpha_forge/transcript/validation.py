"""State-machine validation for a linear schema-v1 transcript."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from alpha_forge.providers.base import (
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
)
from alpha_forge.transcript.records import (
    TranscriptCorruptError,
    TranscriptRecord,
)
from alpha_forge.transcript.representations import (
    validate_tool_result_representation,
)

_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")


def tool_calls(output: ModelOutput) -> tuple[ToolCall, ...]:
    return tuple(item for item in output.items if isinstance(item, ToolCall))


@dataclass(slots=True)
class TranscriptState:
    """Small replay index containing references, not projected message copies."""

    session: SessionOpened | None = None
    event_ids: set[str] = field(default_factory=set)
    inputs: dict[str, InputAccepted] = field(default_factory=dict)
    command_completions: set[str] = field(default_factory=set)
    outputs: dict[str, ModelOutput] = field(default_factory=dict)
    output_order: list[str] = field(default_factory=list)
    outputs_by_prompt: dict[str, list[str]] = field(default_factory=dict)
    results: dict[str, ToolResult] = field(default_factory=dict)
    results_by_output: dict[str, dict[str, str]] = field(default_factory=dict)
    result_representations: dict[str, ToolResultRepresentation] = field(
        default_factory=dict
    )
    exchange_visibility: dict[str, bool] = field(default_factory=dict)
    active_prompt_event_id: str | None = None
    failed_prompts: set[str] = field(default_factory=set)

    @property
    def revision(self) -> int:
        return len(self.event_ids)

    def clone(self) -> TranscriptState:
        return TranscriptState(
            session=self.session,
            event_ids=set(self.event_ids),
            inputs=dict(self.inputs),
            command_completions=set(self.command_completions),
            outputs=dict(self.outputs),
            output_order=list(self.output_order),
            outputs_by_prompt={
                prompt_id: list(output_ids)
                for prompt_id, output_ids in self.outputs_by_prompt.items()
            },
            results=dict(self.results),
            results_by_output={
                output_id: dict(result_ids)
                for output_id, result_ids in self.results_by_output.items()
            },
            result_representations=dict(self.result_representations),
            exchange_visibility=dict(self.exchange_visibility),
            active_prompt_event_id=self.active_prompt_event_id,
            failed_prompts=set(self.failed_prompts),
        )

    def apply(self, record: TranscriptRecord) -> None:
        self._validate_envelope(record)
        event = record.event
        if isinstance(event, SessionOpened):
            self._session_opened(record, event)
        elif self.session is None:
            self._fail("first transcript event must be session.opened")
        elif isinstance(event, SessionLinked):
            self._session_linked(event)
        elif isinstance(event, InputAccepted):
            self._input_accepted(record, event)
        elif isinstance(event, CommandCompleted):
            self._command_completed(event)
        elif isinstance(event, ModelOutput):
            self._model_output(record, event)
        elif isinstance(event, ToolResult):
            self._tool_result(record, event)
        elif isinstance(event, ContextEdited):
            self._context_edited(event)
        elif isinstance(event, QueryFailed):
            self._query_failed(event)
        else:
            self._fail(f"unsupported event: {type(event).__name__}")
        self.event_ids.add(record.event_id)

    def _validate_envelope(self, record: TranscriptRecord) -> None:
        if (
            isinstance(record.sequence, bool)
            or not isinstance(record.sequence, int)
            or record.sequence != len(self.event_ids)
        ):
            self._fail(
                f"expected sequence {len(self.event_ids)}, got {record.sequence}"
            )
        if (
            not isinstance(record.event_id, str)
            or not record.event_id
            or record.event_id in self.event_ids
        ):
            self._fail(f"duplicate or empty event ID: {record.event_id}")
        if not isinstance(record.recorded_at, str) or not record.recorded_at:
            self._fail("recorded_at cannot be empty")
        try:
            parsed = datetime.fromisoformat(
                record.recorded_at.replace("Z", "+00:00")
            )
        except ValueError:
            self._fail(f"invalid recorded_at: {record.recorded_at}")
        if parsed.tzinfo is None:
            self._fail("recorded_at must include a timezone")

    def _session_opened(
        self,
        record: TranscriptRecord,
        event: SessionOpened,
    ) -> None:
        if record.sequence != 0 or self.session is not None:
            self._fail("session.opened must be the first and only session event")
        if (
            not isinstance(event.session_id, str)
            or not _SAFE_SESSION_ID.fullmatch(event.session_id)
        ):
            self._fail("invalid session ID")
        if event.instructions is not None and not isinstance(
            event.instructions,
            str,
        ):
            self._fail("session instructions must be a string or null")
        self.session = event

    def _session_linked(self, event: SessionLinked) -> None:
        if event.kind not in ("clear", "resume"):
            self._fail(f"invalid session link kind: {event.kind}")
        if (
            not isinstance(event.source_session_id, str)
            or not event.source_session_id
            or not isinstance(event.source_command_event_id, str)
            or not event.source_command_event_id
        ):
            self._fail("session link references cannot be empty")

    def _input_accepted(
        self,
        record: TranscriptRecord,
        event: InputAccepted,
    ) -> None:
        if self.active_prompt_event_id is not None:
            self._fail("cannot accept another input while a prompt is open")
        if not isinstance(event.text, str):
            self._fail("input text must be a string")
        if event.kind == "prompt":
            if event.command_name is not None or event.command_arguments is not None:
                self._fail("prompt input cannot include command fields")
            self.active_prompt_event_id = record.event_id
        elif event.kind == "command":
            if (
                not isinstance(event.command_name, str)
                or not event.command_name
                or not isinstance(event.command_arguments, str)
            ):
                self._fail("command input requires parsed name and arguments")
        else:
            self._fail(f"invalid input kind: {event.kind}")
        self.inputs[record.event_id] = event

    def _command_completed(self, event: CommandCompleted) -> None:
        if not isinstance(event.command_event_id, str):
            self._fail("command completion reference must be a string")
        command = self.inputs.get(event.command_event_id)
        if command is None or command.kind != "command":
            self._fail(
                f"command completion references unknown command: "
                f"{event.command_event_id}"
            )
        if event.command_event_id in self.command_completions:
            self._fail(
                f"duplicate command completion: {event.command_event_id}"
            )
        if event.status not in ("success", "error"):
            self._fail(f"invalid command status: {event.status}")
        if not isinstance(event.messages, tuple) or any(
            not isinstance(message, CommandMessage)
            or message.level not in ("notice", "error")
            or not isinstance(message.content, str)
            for message in event.messages
        ):
            self._fail("invalid command completion message")
        self.command_completions.add(event.command_event_id)

    def _model_output(
        self,
        record: TranscriptRecord,
        event: ModelOutput,
    ) -> None:
        if (
            not isinstance(event.prompt_event_id, str)
            or event.prompt_event_id != self.active_prompt_event_id
        ):
            self._fail(
                f"model output does not reference the active prompt: "
                f"{event.prompt_event_id}"
            )
        prompt = self.inputs.get(event.prompt_event_id)
        if prompt is None or prompt.kind != "prompt":
            self._fail(f"unknown prompt: {event.prompt_event_id}")
        prior_ids = self.outputs_by_prompt.get(event.prompt_event_id, [])
        if prior_ids and not self.output_complete(prior_ids[-1]):
            self._fail(
                "cannot append another model output before all tool results exist"
            )
        if not isinstance(event.items, tuple) or not event.items:
            self._fail("model output must contain at least one item")
        if event.finish_reason is not None and not isinstance(
            event.finish_reason,
            str,
        ):
            self._fail("model stop reason must be a string or null")
        calls = tool_calls(event)
        for item in event.items:
            if isinstance(item, ToolCall):
                if (
                    not isinstance(item.call_id, str)
                    or not item.call_id
                    or not isinstance(item.name, str)
                    or not item.name
                ):
                    self._fail("tool calls require nonempty IDs and names")
                if not isinstance(item.arguments, str):
                    self._fail("tool call arguments must be a string")
            elif isinstance(item, OutputMessage):
                if not isinstance(item.content, tuple) or not item.content:
                    self._fail("output message content cannot be empty")
                for part in item.content:
                    if isinstance(part, OutputText):
                        if not isinstance(part.text, str):
                            self._fail("invalid output text")
                    elif isinstance(part, OutputRefusal):
                        if not isinstance(part.refusal, str):
                            self._fail("invalid output refusal")
                    else:
                        self._fail("invalid output message content")
            elif isinstance(item, ReasoningItem):
                if not isinstance(item.content, str):
                    self._fail("reasoning content must be a string")
            else:
                self._fail(f"unsupported model output item: {type(item).__name__}")
        call_ids = [call.call_id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            self._fail("tool-call IDs must be unique within a model output")
        if event.usage is not None:
            if not isinstance(event.usage, TokenUsage):
                self._fail("model usage must be token usage or null")
            if any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                )
                for value in (
                    event.usage.input_tokens,
                    event.usage.cached_tokens,
                    event.usage.output_tokens,
                    event.usage.total_tokens,
                )
            ):
                self._fail(
                    "token usage values must be non-negative integers or null"
                )
        self.outputs[record.event_id] = event
        self.output_order.append(record.event_id)
        self.outputs_by_prompt.setdefault(event.prompt_event_id, []).append(
            record.event_id
        )
        self.results_by_output[record.event_id] = {}
        self.exchange_visibility[record.event_id] = True
        if not calls:
            self.active_prompt_event_id = None

    def _tool_result(
        self,
        record: TranscriptRecord,
        event: ToolResult,
    ) -> None:
        if (
            not isinstance(event.model_output_event_id, str)
            or not isinstance(event.call_id, str)
        ):
            self._fail("tool-result references must be strings")
        output = self.outputs.get(event.model_output_event_id)
        if output is None:
            self._fail(
                f"tool result references unknown model output: "
                f"{event.model_output_event_id}"
            )
        if output.prompt_event_id != self.active_prompt_event_id:
            self._fail("tool result does not belong to the active prompt")
        prompt_outputs = self.outputs_by_prompt[output.prompt_event_id]
        if prompt_outputs[-1] != event.model_output_event_id:
            self._fail("tool result must belong to the latest model output")
        calls = {call.call_id: call for call in tool_calls(output)}
        if event.call_id not in calls:
            self._fail(f"tool result references unknown call: {event.call_id}")
        result_ids = self.results_by_output[event.model_output_event_id]
        if event.call_id in result_ids:
            self._fail(f"duplicate result for tool call: {event.call_id}")
        ordered_calls = tool_calls(output)
        expected_call = ordered_calls[len(result_ids)]
        if event.call_id != expected_call.call_id:
            self._fail(
                "tool results must be appended in model-output call order"
            )
        if event.status not in ("success", "error", "interrupted"):
            self._fail(f"invalid tool-result status: {event.status}")
        if not isinstance(event.content, str):
            self._fail("tool-result content must be a string")
        self.results[record.event_id] = event
        result_ids[event.call_id] = record.event_id
        self.result_representations[record.event_id] = OriginalRepresentation()

    def _context_edited(self, event: ContextEdited) -> None:
        if not isinstance(event.policy, PolicyInvocation):
            self._fail("context edit requires a policy invocation")
        if not isinstance(event.policy.name, str) or not event.policy.name:
            self._fail("context policy name cannot be empty")
        if (
            isinstance(event.policy.version, bool)
            or not isinstance(event.policy.version, int)
            or event.policy.version <= 0
        ):
            self._fail("context policy version must be positive")
        if not isinstance(event.policy.parameters, Mapping):
            self._fail("context policy parameters must be an object")
        if not isinstance(event.operations, tuple) or not event.operations:
            self._fail("context edit must contain at least one operation")

        changed_slots: set[tuple[str, str]] = set()
        for operation in event.operations:
            if isinstance(operation, SetToolResultRepresentation):
                if not isinstance(operation.result_event_id, str):
                    self._fail(
                        "tool-result representation reference must be a string"
                    )
                slot = ("result-representation", operation.result_event_id)
                if slot in changed_slots:
                    self._fail(
                        "context edit assigns a result representation twice"
                    )
                changed_slots.add(slot)
                self._set_result_representation(operation)
            elif isinstance(operation, SetToolExchangeVisibility):
                if not isinstance(operation.model_output_event_id, str):
                    self._fail("tool-exchange reference must be a string")
                slot = ("exchange-visibility", operation.model_output_event_id)
                if slot in changed_slots:
                    self._fail("context edit assigns exchange visibility twice")
                changed_slots.add(slot)
                self._set_exchange_visibility(operation)
            else:
                self._fail(
                    f"unsupported context operation: {type(operation).__name__}"
                )

    def _set_result_representation(
        self,
        operation: SetToolResultRepresentation,
    ) -> None:
        if not isinstance(operation.result_event_id, str):
            self._fail("tool-result representation reference must be a string")
        result = self.results.get(operation.result_event_id)
        if result is None:
            self._fail(
                f"context edit references unknown tool result: "
                f"{operation.result_event_id}"
            )
        current = self.result_representations[operation.result_event_id]
        target = operation.representation
        if target == current:
            self._fail("context operation does not change result representation")
        try:
            validate_tool_result_representation(
                result_event_id=operation.result_event_id,
                content=result.content,
                representation=target,
            )
        except ValueError as exc:
            self._fail(str(exc))
        self.result_representations[operation.result_event_id] = target

    def _set_exchange_visibility(
        self,
        operation: SetToolExchangeVisibility,
    ) -> None:
        if not isinstance(operation.model_output_event_id, str):
            self._fail("tool-exchange reference must be a string")
        output = self.outputs.get(operation.model_output_event_id)
        if output is None:
            self._fail(
                f"context edit references unknown model output: "
                f"{operation.model_output_event_id}"
            )
        if not tool_calls(output):
            self._fail("only a tool-calling model output can form an exchange")
        if not self.output_complete(operation.model_output_event_id):
            self._fail("cannot change visibility of an incomplete tool exchange")
        prompt_outputs = self.outputs_by_prompt[output.prompt_event_id]
        target_index = prompt_outputs.index(operation.model_output_event_id)
        if (
            output.prompt_event_id == self.active_prompt_event_id
            and target_index == len(prompt_outputs) - 1
        ):
            self._fail("cannot hide or restore the current tool exchange tail")
        current = self.exchange_visibility[operation.model_output_event_id]
        if not isinstance(operation.visible, bool):
            self._fail("exchange visibility must be a boolean")
        if operation.visible == current:
            self._fail("context operation does not change exchange visibility")
        self.exchange_visibility[operation.model_output_event_id] = operation.visible

    def _query_failed(self, event: QueryFailed) -> None:
        if event.prompt_event_id != self.active_prompt_event_id:
            self._fail("query failure does not reference the active prompt")
        if event.stage not in (
            "context",
            "provider",
            "tool_round_limit",
            "internal",
        ):
            self._fail(f"invalid query failure stage: {event.stage}")
        if not isinstance(event.message, str):
            self._fail("query failure message must be a string")
        output_ids = self.outputs_by_prompt.get(event.prompt_event_id, [])
        if output_ids and not self.output_complete(output_ids[-1]):
            self._fail("cannot fail a query with an incomplete tool exchange")
        self.failed_prompts.add(event.prompt_event_id)
        self.active_prompt_event_id = None

    def output_complete(self, output_event_id: str) -> bool:
        output = self.outputs[output_event_id]
        calls = tool_calls(output)
        if not calls:
            return True
        results = self.results_by_output[output_event_id]
        return all(call.call_id in results for call in calls)

    @staticmethod
    def _fail(message: str) -> None:
        raise TranscriptCorruptError(message)


def validate_records(
    records: tuple[TranscriptRecord, ...] | list[TranscriptRecord],
) -> TranscriptState:
    state = TranscriptState()
    for record in records:
        state.apply(record)
    if state.session is None:
        raise TranscriptCorruptError("transcript is empty")
    return state


__all__ = ["TranscriptState", "tool_calls", "validate_records"]
