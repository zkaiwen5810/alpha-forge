"""Reactive UI state over durable projections and ephemeral progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from alpha_forge.application.events import (
    ExitRequested,
    InputQueued,
    InputStarted,
    ModelOutputRecorded,
    PersistenceFailed,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    RequestFailed,
    SessionView,
    SessionViewChanged,
    StatusChanged,
    ToolResultRecorded,
    ToolStarted,
)
from alpha_forge.events import Event
from alpha_forge.projectors.ui_history import (
    UiCommandMessage,
    UiHistoryItem,
    UiModelOutput,
    UiPrompt,
    UiQueryFailure,
    UiSessionLink,
    UiToolResult,
)
from alpha_forge.providers.base import (
    ProviderOutput,
    ProviderOutputAccumulator,
    TokenUsage,
    ToolCall,
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
DEFAULT_UI_TOOL_RESULT_LINES = 20


@dataclass(frozen=True, slots=True)
class HistoryLine:
    role: HistoryRole
    text: str


@dataclass(frozen=True, slots=True)
class UiToolResultPreview:
    content: str
    truncated: bool


class UiToolResultPreviewStrategy(Protocol):
    def preview(self, content: str) -> UiToolResultPreview:
        """Bound presentation content without changing model context."""


class TailLinesUiToolResultPreview:
    def __init__(self, line_limit: int = DEFAULT_UI_TOOL_RESULT_LINES) -> None:
        if line_limit <= 0:
            raise ValueError("UI tool-result line limit must be positive")
        self.line_limit = line_limit

    def preview(self, content: str) -> UiToolResultPreview:
        lines = content.splitlines()
        if len(lines) <= self.line_limit:
            return UiToolResultPreview(content, False)
        return UiToolResultPreview("\n".join(lines[-self.line_limit :]), True)


@dataclass(slots=True)
class ActiveProviderResponse:
    prompt_event_id: str
    request_id: str
    accumulator: ProviderOutputAccumulator = field(
        default_factory=ProviderOutputAccumulator
    )
    output: ProviderOutput | None = None
    persistence_error: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveTool:
    model_output_event_id: str
    call: ToolCall
    persistence_error: str | None = None


type ActiveState = ActiveProviderResponse | ActiveTool


class ChatUiState:
    """Reduce application events into presentation-only state."""

    def __init__(
        self,
        view: SessionView,
        *,
        tool_result_preview: UiToolResultPreviewStrategy | None = None,
    ) -> None:
        self.view = view
        self.history_items: tuple[UiHistoryItem, ...] = view.items
        self.active: ActiveState | None = None
        self.status = "Ready"
        self.exiting = False
        self.persistence_error: str | None = None
        self.tool_result_preview = tool_result_preview or TailLinesUiToolResultPreview()
        self._queued_inputs: dict[str, str] = {}
        self._cache_revision: int | None = None
        self._transcript_cache: tuple[HistoryLine, ...] = ()

    @property
    def pending_inputs(self) -> list[str]:
        return list(self._queued_inputs.values())

    @property
    def pending_prompts(self) -> list[str]:
        return self.pending_inputs

    @property
    def has_unsaved_active(self) -> bool:
        return self.active is not None and self.active.persistence_error is not None

    def handle(self, event: Event) -> bool:
        if isinstance(event, SessionViewChanged):
            self.view = event.view
            self.history_items = event.view.items
            if event.reset_active:
                self.active = None
            self._cache_revision = None
            self.status = self._queue_status()
        elif isinstance(event, InputQueued):
            self._queued_inputs[event.item_id] = event.raw
            self.status = self._queue_status()
        elif isinstance(event, InputStarted):
            self._queued_inputs.pop(event.item_id, None)
            self.status = self._queue_status()
        elif isinstance(event, ProviderRequestStarted):
            self.active = ActiveProviderResponse(
                event.prompt_event_id,
                event.request_id,
            )
            self.status = "Streaming response"
        elif isinstance(event, ProviderDeltaReceived):
            active = self._active_provider(event.request_id)
            active.accumulator.apply(event.delta)
            self.status = "Streaming response"
        elif isinstance(event, ProviderResponseCompleted):
            active = self._active_provider(event.request_id)
            active.output = event.output
            self.status = "Saving response"
        elif isinstance(event, ModelOutputRecorded):
            self.active = None
            self.status = self._queue_status()
        elif isinstance(event, ToolStarted):
            self.active = ActiveTool(event.model_output_event_id, event.call)
            self.status = f"Running tool: {event.call.name}"
        elif isinstance(event, ToolResultRecorded):
            if (
                isinstance(self.active, ActiveTool)
                and self.active.call.call_id == event.call_id
            ):
                self.active = None
            self.status = self._queue_status()
        elif isinstance(event, PersistenceFailed):
            self.persistence_error = event.message
            if isinstance(self.active, ActiveProviderResponse):
                self.active.persistence_error = event.message
            elif isinstance(self.active, ActiveTool):
                self.active = ActiveTool(
                    self.active.model_output_event_id,
                    self.active.call,
                    event.message,
                )
            self.status = f"Cannot persist {event.stage}: {event.message}"
        elif isinstance(event, RequestFailed):
            self.active = None
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
        if self._cache_revision != self.view.revision:
            self._transcript_cache = tuple(self._render_transcript())
            self._cache_revision = self.view.revision
        return list(self._transcript_cache)

    def active_lines(self) -> list[HistoryLine]:
        if isinstance(self.active, ActiveProviderResponse):
            return self._render_active_provider(self.active)
        if isinstance(self.active, ActiveTool):
            lines = [
                HistoryLine(
                    "tool_call",
                    f"  Running tool [{self.active.call.name}]…",
                )
            ]
            if self.active.persistence_error:
                lines.extend(
                    self._labeled_lines(
                        "Error: ",
                        f"not persisted: {self.active.persistence_error}",
                        "error",
                    )
                )
            return lines
        return []

    def history_lines(self) -> list[HistoryLine]:
        return self.transcript_lines() + self.active_lines()

    def transcript_text(self) -> str:
        return "\n".join(line.text for line in self.transcript_lines())

    def active_text(self) -> str:
        return "\n".join(line.text for line in self.active_lines())

    def history_text(self) -> str:
        return "\n".join(line.text for line in self.history_lines())

    def render_pending(self) -> str:
        if not self.pending_inputs:
            return "No pending prompts."
        return "\n".join(
            f"{index}. {value}"
            for index, value in enumerate(self.pending_inputs, start=1)
        )

    def _render_transcript(self) -> list[HistoryLine]:
        results_by_output: dict[str, dict[str, UiToolResult]] = {}
        for item in self.history_items:
            if not isinstance(item, UiToolResult):
                continue
            results_by_output.setdefault(item.model_output_event_id, {})[
                item.call_id
            ] = item
        usage_output_event_id = self._usage_output_event_id()

        lines: list[HistoryLine] = []
        for item in self.history_items:
            if isinstance(item, UiPrompt):
                if lines:
                    lines.append(HistoryLine("spacer", ""))
                lines.extend(self._labeled_lines("You: ", item.content, "user"))
            elif isinstance(item, UiModelOutput):
                lines.extend(
                    self._render_model_output(
                        item,
                        results_by_output.get(item.output_event_id, {}),
                        show_usage=item.output_event_id == usage_output_event_id,
                    )
                )
            elif isinstance(item, UiToolResult):
                # Results are rendered beside their correlated calls above. The
                # durable projection remains flat and append-only.
                continue
            elif isinstance(item, UiCommandMessage):
                lines.extend(
                    self._labeled_lines(
                        "Notice: " if item.message.level == "notice" else "Error: ",
                        item.message.content,
                        item.message.level,
                    )
                )
            elif isinstance(item, UiSessionLink):
                lines.append(
                    HistoryLine(
                        "notice",
                        f"Notice: session {item.kind} from {item.source_session_id}",
                    )
                )
            elif isinstance(item, UiQueryFailure):
                lines.extend(
                    self._labeled_lines(
                        "Error: ",
                        f"request failed: {item.message}",
                        "error",
                    )
                )
        return lines or [HistoryLine("notice", "No messages yet.")]

    def _usage_output_event_id(self) -> str | None:
        latest_completion: tuple[int, str | None] | None = None
        for item in self.history_items:
            completion: tuple[int, str | None] | None = None
            if isinstance(item, UiModelOutput) and not item.tool_calls:
                completion = (item.sequence, item.output_event_id)
            elif isinstance(item, UiQueryFailure):
                completion = (item.sequence, None)
            if completion is not None and (
                latest_completion is None or completion[0] > latest_completion[0]
            ):
                latest_completion = completion
        return None if latest_completion is None else latest_completion[1]

    def _render_model_output(
        self,
        output: UiModelOutput,
        results_by_call: dict[str, UiToolResult],
        *,
        show_usage: bool,
    ) -> list[HistoryLine]:
        lines: list[HistoryLine] = []
        if output.reasoning:
            lines.extend(
                self._labeled_lines(
                    "  Assistant reasoning: ",
                    output.reasoning,
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
            result = results_by_call.get(call.call_id)
            if result is not None:
                lines.extend(self._render_tool_result(call, result))
        if output.text:
            lines.extend(
                self._labeled_lines(
                    "  Assistant note: " if output.tool_calls else "Assistant: ",
                    output.text,
                    "assistant_note" if output.tool_calls else "assistant",
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
        if show_usage and output.usage:
            lines.append(HistoryLine("token_usage", self._format_usage(output.usage)))
        return lines

    def _render_tool_result(
        self,
        call: ToolCall,
        result: UiToolResult,
    ) -> list[HistoryLine]:
        preview = self.tool_result_preview.preview(result.content)
        label = "Tool error" if result.status != "success" else "Tool result"
        if preview.truncated:
            label += " preview"
        if result.excluded_from_model:
            label += " (excluded from model context)"
        return self._labeled_lines(
            f"  {label} [{call.name}]: ",
            preview.content,
            "error" if result.status != "success" else "tool_result",
        )

    def _render_active_provider(
        self,
        active: ActiveProviderResponse,
    ) -> list[HistoryLine]:
        if active.output is not None:
            output = UiModelOutput(
                -1,
                "",
                active.prompt_event_id,
                active.output.output_text,
                active.output.reasoning,
                active.output.refusal,
                active.output.tool_calls,
                active.output.usage,
            )
            lines = self._render_model_output(
                output,
                {},
                show_usage=False,
            )
        else:
            preview = active.accumulator
            lines = []
            if preview.reasoning:
                lines.extend(
                    self._labeled_lines(
                        "  Assistant reasoning: ",
                        preview.reasoning,
                        "assistant_note",
                    )
                )
            for _, call in sorted(preview.tool_calls.items()):
                lines.extend(
                    self._labeled_lines(
                        f"  Tool call [{call.name or '…'}] (streaming): ",
                        call.arguments or "…",
                        "tool_call",
                    )
                )
            if preview.text:
                lines.extend(
                    self._labeled_lines(
                        ("  Assistant note: " if preview.tool_calls else "Assistant: "),
                        preview.text,
                        ("assistant_note" if preview.tool_calls else "assistant"),
                    )
                )
            if preview.refusal:
                lines.extend(
                    self._labeled_lines(
                        "Assistant refusal: ",
                        preview.refusal,
                        "assistant",
                    )
                )
        if active.persistence_error:
            lines.extend(
                self._labeled_lines(
                    "Error: ",
                    f"not persisted: {active.persistence_error}",
                    "error",
                )
            )
        return lines

    def _active_provider(self, request_id: str) -> ActiveProviderResponse:
        if (
            not isinstance(self.active, ActiveProviderResponse)
            or self.active.request_id != request_id
        ):
            raise RuntimeError("provider event does not match the active request")
        return self.active

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
    def _format_usage(usage: TokenUsage) -> str:
        parts = []
        if usage.total_tokens is not None:
            parts.append(f"Total tokens: {usage.total_tokens:,}")
        if usage.cached_tokens is not None:
            parts.append(f"Cached tokens: {usage.cached_tokens:,}")
        return " | ".join(parts)

    def _queue_status(self) -> str:
        if self.persistence_error is not None:
            return "Persistence failed; input processing stopped"
        if self.exiting:
            return "Exiting after queued inputs"
        if not self.pending_inputs:
            return "Ready"
        if len(self.pending_inputs) == 1:
            return "1 input queued"
        return f"{len(self.pending_inputs)} inputs queued"


__all__ = [
    "ActiveProviderResponse",
    "ActiveState",
    "ActiveTool",
    "ChatUiState",
    "DEFAULT_UI_TOOL_RESULT_LINES",
    "HistoryLine",
    "HistoryRole",
    "TailLinesUiToolResultPreview",
    "UiToolResultPreview",
    "UiToolResultPreviewStrategy",
]
