"""Durable transcript rendering plus UI-owned ephemeral response state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from alpha_forge.events import Event
from alpha_forge.models import TokenUsage, ToolCall
from alpha_forge.streaming import (
    ModelResponse,
    ModelResponseAccumulator,
    ReasoningDelta,
    RefusalDelta,
    TextDelta,
    ToolCallDelta,
    UsageUpdate,
)
from alpha_forge.system_events import (
    AssistantMessageAdded,
    ExitRequested,
    InputQueued,
    InputStarted,
    ModelResponseCompleted,
    ModelResponseStarted,
    PersistenceFailed,
    RequestFailed,
    SessionView,
    SessionViewChanged,
    StatusChanged,
    ToolBatchStarted,
    ToolResultsFinalized,
    ToolResultsUpdated,
    ToolStarted,
)
from alpha_forge.ui_history import (
    UiCommandMessage,
    UiHistoryItem,
    UiModelOutput,
    UiToolResult,
    UiTransition,
    UiTurnFailure,
    UiUserMessage,
)

HistoryRole = Literal[
    "user",
    "assistant",
    "assistant_note",
    "tool_call",
    "tool_result",
    "token_usage",
    "notice",
    "error",
    "spacer",
]


@dataclass(frozen=True, slots=True)
class HistoryLine:
    role: HistoryRole
    text: str


@dataclass(slots=True)
class ActiveModelResponse:
    turn_id: str
    output_id: str
    preview: ModelResponseAccumulator = field(default_factory=ModelResponseAccumulator)
    response: ModelResponse | None = None
    persistence_error: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveToolResult:
    call_id: str
    name: str
    content: str
    failed: bool
    previewed: bool


@dataclass(slots=True)
class ActiveToolBatch:
    turn_id: str
    output_id: str
    calls: tuple[ToolCall, ...]
    running_call_id: str | None = None
    results: dict[str, ActiveToolResult] = field(default_factory=dict)
    persistence_error: str | None = None


type ActiveState = ActiveModelResponse | ActiveToolBatch


class ChatUiState:
    """Reduce immutable application events into presentation-only state."""

    def __init__(self, view: SessionView) -> None:
        self.view = view
        self.head_turn_id = view.head_turn_id
        self.history_items: tuple[UiHistoryItem, ...] = view.items
        self.active: ActiveState | None = None
        self.status = "Ready"
        self.exiting = False
        self.persistence_error: str | None = None
        self._queued_inputs: dict[str, str] = {}
        self._cache_key: tuple[int, str | None, str | None] | None = None
        self._transcript_cache: tuple[HistoryLine, ...] = ()

    @property
    def pending_inputs(self) -> list[str]:
        return list(self._queued_inputs.values())

    @property
    def pending_prompts(self) -> list[str]:
        """Compatibility name for the terminal's queued-input panel."""
        return self.pending_inputs

    @property
    def has_unsaved_active(self) -> bool:
        return (
            isinstance(self.active, ActiveModelResponse)
            and self.active.persistence_error is not None
        ) or (
            isinstance(self.active, ActiveToolBatch)
            and self.active.persistence_error is not None
        )

    def handle(self, event: Event) -> bool:
        """Apply one relevant event and report whether presentation changed."""
        if isinstance(event, SessionViewChanged):
            self.view = event.view
            self.head_turn_id = event.view.head_turn_id
            self.history_items = event.view.items
            if event.reset_active:
                self.active = None
            self._invalidate_transcript()
            self.status = self._queue_status()
        elif isinstance(event, InputQueued):
            self._queued_inputs[event.item_id] = event.raw
            self.status = self._queue_status()
        elif isinstance(event, InputStarted):
            self._queued_inputs.pop(event.item_id, None)
            self.status = self._queue_status()
        elif isinstance(event, ModelResponseStarted):
            self.active = ActiveModelResponse(event.turn_id, event.output_id)
            self.status = "Streaming response"
            self._invalidate_transcript()
        elif isinstance(event, ModelResponseCompleted):
            if not isinstance(self.active, ActiveModelResponse):
                raise RuntimeError("no active model response")
            if self.active.output_id != event.output_id:
                raise RuntimeError("completed response does not match active output")
            self.active.response = event.response
            self.status = "Saving response"
        elif isinstance(
            event,
            (
                TextDelta,
                ReasoningDelta,
                ToolCallDelta,
                RefusalDelta,
                UsageUpdate,
            ),
        ):
            if not isinstance(self.active, ActiveModelResponse):
                raise RuntimeError("no active model response")
            self.active.preview.apply(event)
            self.status = "Streaming response"
        elif isinstance(event, AssistantMessageAdded):
            if (
                not isinstance(self.active, ActiveModelResponse)
                or self.active.output_id != event.output_id
            ):
                raise RuntimeError("saved response does not match active output")
            self.active = None
            self._invalidate_transcript()
            self.status = self._queue_status()
        elif isinstance(event, ToolBatchStarted):
            self.active = ActiveToolBatch(
                event.turn_id,
                event.output_id,
                event.calls,
            )
            self.status = "Preparing tools"
            self._invalidate_transcript()
        elif isinstance(event, ToolStarted):
            if not isinstance(self.active, ActiveToolBatch):
                raise RuntimeError("no active tool batch")
            self.active.running_call_id = event.call_id
            call = self._active_call(self.active, event.call_id)
            self.status = f"Running tool: {call.name}"
        elif isinstance(event, ToolResultsUpdated):
            if not isinstance(self.active, ActiveToolBatch):
                raise RuntimeError("no active tool batch")
            calls = {call.id: call for call in self.active.calls}
            self.active.results = {
                result.call_id: ActiveToolResult(
                    call_id=result.call_id,
                    name=calls[result.call_id].name,
                    content=result.content,
                    failed=result.failed,
                    previewed=result.previewed,
                )
                for result in event.results
            }
            self.active.running_call_id = None
            self.status = "Continuing tools"
        elif isinstance(event, ToolResultsFinalized):
            if (
                not isinstance(self.active, ActiveToolBatch)
                or self.active.output_id != event.output_id
            ):
                raise RuntimeError("finalized tools do not match active output")
            self.active = None
            self._invalidate_transcript()
            self.status = self._queue_status()
        elif isinstance(event, PersistenceFailed):
            self.persistence_error = event.message
            if isinstance(self.active, ActiveModelResponse):
                self.active.persistence_error = event.message
            elif isinstance(self.active, ActiveToolBatch):
                self.active.persistence_error = event.message
            self.status = f"Cannot persist {event.stage}: {event.message}"
        elif isinstance(event, RequestFailed):
            self.active = None
            self._invalidate_transcript()
            self.status = f"Request failed: {event.message}"
        elif isinstance(event, StatusChanged):
            self.status = event.message
        elif isinstance(event, ExitRequested):
            self.exiting = True
            self.status = self._queue_status()
        else:
            return False
        return True

    def transcript_lines(self) -> list[HistoryLine]:
        active_turn_id = self.active.turn_id if self.active is not None else None
        key = (self.view.revision, self.head_turn_id, active_turn_id)
        if self._cache_key != key:
            self._transcript_cache = tuple(
                self._render_transcript(active_turn_id=active_turn_id)
            )
            self._cache_key = key
        return list(self._transcript_cache)

    def active_lines(self) -> list[HistoryLine]:
        if isinstance(self.active, ActiveModelResponse):
            return self._render_active_model(self.active)
        if isinstance(self.active, ActiveToolBatch):
            return self._render_active_tools(self.active)
        return []

    def history_lines(self) -> list[HistoryLine]:
        """Return durable history followed by its ephemeral active tail."""
        return self.transcript_lines() + self.active_lines()

    def transcript_text(self) -> str:
        return "\n".join(line.text for line in self.transcript_lines())

    def active_text(self) -> str:
        return "\n".join(line.text for line in self.active_lines())

    def history_text(self) -> str:
        return "\n".join(line.text for line in self.history_lines())

    def render_pending(self) -> str:
        pending = self.pending_prompts
        if not pending:
            return "No pending prompts."
        return "\n".join(
            f"{index}. {prompt}" for index, prompt in enumerate(pending, start=1)
        )

    def _render_transcript(
        self,
        *,
        active_turn_id: str | None,
    ) -> list[HistoryLine]:
        items = list(self.history_items)
        users = [item for item in items if isinstance(item, UiUserMessage)]
        outputs = {
            item.output_id: item for item in items if isinstance(item, UiModelOutput)
        }
        results_by_output: dict[str, dict[str, UiToolResult]] = {}
        for item in items:
            if isinstance(item, UiToolResult):
                results_by_output.setdefault(item.output_id, {})[item.call_id] = item

        visible_users = [
            user
            for user in users
            if (
                user.turn_id == active_turn_id
                or not self._turn_is_pending(user.turn_id, items)
            )
        ]
        lines: list[HistoryLine] = []
        for user in visible_users:
            if lines:
                lines.append(HistoryLine("spacer", ""))
            lines.extend(self._labeled_lines("You: ", user.content, "user"))
            activities = [
                item
                for item in items
                if (
                    isinstance(item, (UiModelOutput, UiTurnFailure))
                    and item.turn_id == user.turn_id
                )
                or (
                    isinstance(item, UiCommandMessage)
                    and item.context_turn_id == user.turn_id
                )
            ]
            activities.sort(key=lambda item: item.sequence)
            for activity in activities:
                if isinstance(activity, UiModelOutput):
                    lines.extend(
                        self._render_model_output(
                            activity,
                            results_by_output.get(activity.output_id, {}),
                            show_usage=(
                                user is visible_users[-1]
                                and activity
                                is self._latest_output(
                                    visible_users[-1].turn_id,
                                    outputs,
                                )
                            ),
                        )
                    )
                elif isinstance(activity, UiCommandMessage):
                    lines.extend(self._render_command_message(activity))
                else:
                    lines.extend(
                        self._labeled_lines(
                            "Error: ",
                            f"request failed: {activity.error}",
                            "error",
                        )
                    )

        visible_turn_ids = {user.turn_id for user in visible_users}
        global_items = [
            item
            for item in items
            if isinstance(item, UiTransition)
            or (
                isinstance(item, UiCommandMessage)
                and (
                    item.context_turn_id is None
                    or item.context_turn_id not in visible_turn_ids
                )
            )
        ]
        for item in global_items:
            if lines:
                lines.append(HistoryLine("spacer", ""))
            if isinstance(item, UiCommandMessage):
                lines.extend(self._render_command_message(item))
            else:
                lines.append(
                    HistoryLine(
                        "notice",
                        f"Notice: session {item.kind} from {item.source_session_id}",
                    )
                )
        return lines or [HistoryLine("notice", "No messages yet.")]

    def _render_model_output(
        self,
        output: UiModelOutput,
        results: dict[str, UiToolResult],
        *,
        show_usage: bool,
    ) -> list[HistoryLine]:
        lines: list[HistoryLine] = []
        if output.reasoning_content:
            lines.extend(
                self._labeled_lines(
                    "  Assistant reasoning: ",
                    output.reasoning_content,
                    "assistant_note",
                )
            )
        for call in output.tool_calls:
            lines.extend(
                self._labeled_lines(
                    f"  Tool call [{call.name}]: ",
                    call.arguments,
                    "tool_call",
                )
            )
            result = results.get(call.id)
            if result is not None:
                label = "Tool error" if result.failed else "Tool result"
                if result.previewed:
                    label += " preview"
                lines.extend(
                    self._labeled_lines(
                        f"  {label} [{call.name}]: ",
                        result.content,
                        "error" if result.failed else "tool_result",
                    )
                )
        if output.content:
            requesting = bool(output.tool_calls)
            lines.extend(
                self._labeled_lines(
                    "  Assistant note: " if requesting else "Assistant: ",
                    output.content,
                    "assistant_note" if requesting else "assistant",
                )
            )
        if output.refusal:
            lines.extend(
                self._labeled_lines(
                    "Assistant refusal: ",
                    output.refusal,
                    "assistant",
                )
            )
        if show_usage and output.usage is not None:
            lines.append(
                HistoryLine(
                    "token_usage",
                    self._format_token_usage(output.usage),
                )
            )
        return lines

    def _render_active_model(
        self,
        active: ActiveModelResponse,
    ) -> list[HistoryLine]:
        response = active.response
        preview = active.preview
        reasoning = (
            response.reasoning_content
            if response is not None
            else preview.reasoning_content
        )
        content = response.content if response is not None else preview.text
        refusal = response.refusal if response is not None else preview.refusal
        usage = response.usage if response is not None else preview.usage
        lines: list[HistoryLine] = []
        if reasoning:
            lines.extend(
                self._labeled_lines(
                    "  Assistant reasoning: ",
                    reasoning,
                    "assistant_note",
                )
            )
        if response is not None:
            for tool in response.tool_calls:
                lines.extend(
                    self._labeled_lines(
                        f"  Tool call [{tool.name}]: ",
                        tool.arguments,
                        "tool_call",
                    )
                )
            requesting = bool(response.tool_calls)
        else:
            for _, tool in sorted(preview.tool_calls.items()):
                lines.extend(
                    self._labeled_lines(
                        f"  Tool call [{tool.name or '…'}] (streaming): ",
                        tool.arguments or "…",
                        "tool_call",
                    )
                )
            requesting = bool(preview.tool_calls)
        if content:
            lines.extend(
                self._labeled_lines(
                    "  Assistant note: " if requesting else "Assistant: ",
                    content,
                    "assistant_note" if requesting else "assistant",
                )
            )
        if refusal:
            lines.extend(
                self._labeled_lines(
                    "Assistant refusal: ",
                    refusal,
                    "assistant",
                )
            )
        if usage is not None:
            lines.append(
                HistoryLine(
                    "token_usage",
                    self._format_token_usage(usage),
                )
            )
        if active.persistence_error is not None:
            lines.extend(
                self._labeled_lines(
                    "Error: ",
                    f"not persisted: {active.persistence_error}",
                    "error",
                )
            )
        return lines

    def _render_active_tools(
        self,
        active: ActiveToolBatch,
    ) -> list[HistoryLine]:
        lines: list[HistoryLine] = []
        for call in active.calls:
            result = active.results.get(call.id)
            if result is not None:
                label = "Tool error" if result.failed else "Tool result"
                if result.previewed:
                    label += " preview"
                lines.extend(
                    self._labeled_lines(
                        f"  {label} [{result.name}]: ",
                        result.content,
                        "error" if result.failed else "tool_result",
                    )
                )
            if call.id == active.running_call_id:
                lines.append(HistoryLine("tool_call", f"  Running tool [{call.name}]…"))
        if active.persistence_error is not None:
            lines.extend(
                self._labeled_lines(
                    "Error: ",
                    f"tool results not finalized: {active.persistence_error}",
                    "error",
                )
            )
        return lines

    @staticmethod
    def _render_command_message(
        item: UiCommandMessage,
    ) -> list[HistoryLine]:
        label = "Notice: " if item.message.level == "notice" else "Error: "
        return ChatUiState._labeled_lines(
            label,
            item.message.content,
            item.message.level,
        )

    @staticmethod
    def _turn_is_pending(
        turn_id: str,
        items: Sequence[object],
    ) -> bool:
        if any(
            isinstance(item, UiTurnFailure) and item.turn_id == turn_id
            for item in items
        ):
            return False
        outputs = [
            item
            for item in items
            if isinstance(item, UiModelOutput) and item.turn_id == turn_id
        ]
        if not outputs:
            return True
        latest = outputs[-1]
        if not latest.tool_calls:
            return False
        result_calls = {
            item.call_id
            for item in items
            if isinstance(item, UiToolResult) and item.output_id == latest.output_id
        }
        return result_calls == {call.id for call in latest.tool_calls}

    @staticmethod
    def _latest_output(
        turn_id: str,
        outputs: dict[str, UiModelOutput],
    ) -> UiModelOutput | None:
        values = [output for output in outputs.values() if output.turn_id == turn_id]
        return values[-1] if values else None

    @staticmethod
    def _active_call(active: ActiveToolBatch, call_id: str) -> ToolCall:
        for call in active.calls:
            if call.id == call_id:
                return call
        raise KeyError(call_id)

    @staticmethod
    def _labeled_lines(
        prefix: str,
        content: str,
        role: HistoryRole,
    ) -> list[HistoryLine]:
        content_lines = content.splitlines() or [""]
        indentation = " " * len(prefix)
        return [
            HistoryLine(role, (prefix if index == 0 else indentation) + line)
            for index, line in enumerate(content_lines)
        ]

    @staticmethod
    def _format_token_usage(usage: TokenUsage) -> str:
        parts: list[str] = []
        if usage.total_tokens is not None:
            parts.append(f"Total tokens: {usage.total_tokens:,}")
        if usage.cached_tokens is not None:
            if usage.cached_tokens <= 0:
                cache_text = "no reuse yet"
            elif usage.prompt_tokens is None or usage.prompt_tokens <= 0:
                cache_text = "reuse detected"
            else:
                percentage = round(usage.cached_tokens / usage.prompt_tokens * 100)
                cache_text = f"{min(100, max(0, percentage))}% reused"
            parts.append(f"Prompt cache: {cache_text}")
        return " | ".join(parts)

    def _queue_status(self) -> str:
        if self.persistence_error is not None:
            return "Persistence failed; input processing stopped"
        if self.exiting:
            return "Exiting after queued inputs"
        pending = self.pending_prompts
        if not pending:
            return "Ready"
        if len(pending) == 1:
            return "1 input queued"
        return f"{len(pending)} inputs queued"

    def _invalidate_transcript(self) -> None:
        self._cache_key = None


__all__ = [
    "ActiveModelResponse",
    "ActiveState",
    "ActiveToolBatch",
    "ActiveToolResult",
    "ChatUiState",
    "HistoryLine",
    "HistoryRole",
]
