"""UI-facing flat projection of durable transcript activities."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.models import TokenUsage, ToolCall
from alpha_forge.tool_results import TranscriptToolResultLimiter
from alpha_forge.transcript import (
    Command,
    CommandMessage,
    CommandResult,
    ModelOutput,
    SessionTransition,
    ToolResult,
    ToolResultLimit,
    Transcript,
    TurnFailure,
    UserMessage,
)


@dataclass(frozen=True, slots=True)
class UiUserMessage:
    sequence: int
    turn_id: str
    parent_turn_id: str | None
    content: str


@dataclass(frozen=True, slots=True)
class UiModelOutput:
    sequence: int
    output_id: str
    turn_id: str
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    reasoning_content: str | None
    refusal: str | None
    usage: TokenUsage | None


@dataclass(frozen=True, slots=True)
class UiToolResult:
    sequence: int
    output_id: str
    call_id: str
    content: str
    failed: bool
    previewed: bool


@dataclass(frozen=True, slots=True)
class UiCommandMessage:
    sequence: int
    context_turn_id: str | None
    message: CommandMessage


@dataclass(frozen=True, slots=True)
class UiTransition:
    sequence: int
    kind: str
    source_session_id: str


@dataclass(frozen=True, slots=True)
class UiTurnFailure:
    sequence: int
    turn_id: str
    error: str


type UiHistoryItem = (
    UiUserMessage
    | UiModelOutput
    | UiToolResult
    | UiCommandMessage
    | UiTransition
    | UiTurnFailure
)


class UiHistoryProjector:
    """Filter transcript records into flat, durable presentation facts."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript

    def items(self, *, head_turn_id: str | None = None) -> list[UiHistoryItem]:
        selected = {turn.turn_id for turn in self._ancestry(head_turn_id)}
        commands: dict[str, Command] = {}
        outputs: dict[str, ModelOutput] = {}
        raw_results: dict[tuple[str, str], tuple[int, ToolResult]] = {}
        limits: dict[str, ToolResultLimit] = {}

        for record in self.transcript.records:
            event = record.event
            if isinstance(event, Command):
                commands[event.command_id] = event
            elif isinstance(event, ModelOutput):
                outputs[event.output_id] = event
            elif isinstance(event, ToolResult):
                raw_results[(event.output_id, event.call_id)] = (
                    record.sequence,
                    event,
                )
            elif isinstance(event, ToolResultLimit):
                limits[event.output_id] = event

        projected_results: dict[tuple[str, str], UiToolResult] = {}
        for output_id, limit in limits.items():
            output = outputs[output_id]
            if output.turn_id not in selected:
                continue
            for decision in limit.decisions:
                sequence, result = raw_results[(output_id, decision.call_id)]
                projected_results[(output_id, result.call_id)] = UiToolResult(
                    sequence=sequence,
                    output_id=output_id,
                    call_id=result.call_id,
                    content=TranscriptToolResultLimiter.render(
                        result,
                        decision,
                    ),
                    failed=result.failed,
                    previewed=decision.reason is not None,
                )

        items: list[UiHistoryItem] = []
        for record in self.transcript.records:
            event = record.event
            if isinstance(event, UserMessage) and event.turn_id in selected:
                items.append(
                    UiUserMessage(
                        record.sequence,
                        event.turn_id,
                        event.parent_turn_id,
                        event.content,
                    )
                )
            elif isinstance(event, ModelOutput) and event.turn_id in selected:
                items.append(
                    UiModelOutput(
                        record.sequence,
                        event.output_id,
                        event.turn_id,
                        event.content,
                        event.tool_calls,
                        event.reasoning_content,
                        event.refusal,
                        event.usage,
                    )
                )
            elif isinstance(event, ToolResult):
                projected = projected_results.get((event.output_id, event.call_id))
                if projected is not None:
                    items.append(projected)
            elif isinstance(event, CommandResult):
                command = commands[event.command_id]
                if (
                    command.context_turn_id is None
                    or command.context_turn_id in selected
                ):
                    items.extend(
                        UiCommandMessage(
                            record.sequence,
                            command.context_turn_id,
                            message,
                        )
                        for message in event.messages
                    )
            elif isinstance(event, SessionTransition):
                items.append(
                    UiTransition(
                        record.sequence,
                        event.kind,
                        event.source_session_id,
                    )
                )
            elif isinstance(event, TurnFailure) and event.turn_id in selected:
                items.append(
                    UiTurnFailure(
                        record.sequence,
                        event.turn_id,
                        event.error,
                    )
                )
        return items

    def pending_prompts(
        self,
        *,
        head_turn_id: str | None = None,
        exclude_turn_id: str | None = None,
    ) -> list[str]:
        turns = self._ancestry(head_turn_id)
        outputs: dict[str, list[ModelOutput]] = {}
        limits: set[str] = set()
        failures: set[str] = set()
        for event in self.transcript.events:
            if isinstance(event, ModelOutput):
                outputs.setdefault(event.turn_id, []).append(event)
            elif isinstance(event, ToolResultLimit):
                limits.add(event.output_id)
            elif isinstance(event, TurnFailure):
                failures.add(event.turn_id)
        pending: list[str] = []
        for turn in turns:
            if turn.turn_id == exclude_turn_id or turn.turn_id in failures:
                continue
            turn_outputs = outputs.get(turn.turn_id, [])
            if not turn_outputs:
                pending.append(turn.content)
                continue
            latest = turn_outputs[-1]
            if latest.tool_calls and latest.output_id in limits:
                pending.append(turn.content)
        return pending

    def _ancestry(self, head_turn_id: str | None) -> list[UserMessage]:
        turns = {
            event.turn_id: event
            for event in self.transcript.events
            if isinstance(event, UserMessage)
        }
        if not turns:
            return []
        head = head_turn_id or next(reversed(turns))
        if head not in turns:
            raise KeyError(head)
        result: list[UserMessage] = []
        cursor: str | None = head
        while cursor is not None:
            turn = turns[cursor]
            result.append(turn)
            cursor = turn.parent_turn_id
        result.reverse()
        return result


__all__ = [
    "UiCommandMessage",
    "UiHistoryItem",
    "UiHistoryProjector",
    "UiModelOutput",
    "UiToolResult",
    "UiTransition",
    "UiTurnFailure",
    "UiUserMessage",
]
