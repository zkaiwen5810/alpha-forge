"""UI-facing chat state and transcript rendering models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_forge.conversation import ToolMessage

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


@dataclass(frozen=True)
class ToolExchange:
    """UI projection of one tool call and its eventual model-facing result."""

    name: str
    arguments: str
    result: ToolMessage | None = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class IterationOutput:
    """Immutable snapshot of one model iteration as currently rendered."""

    assistant_text: str = ""
    tools: tuple[ToolExchange, ...] = ()
    tool_requesting: bool = False
    token_usage: TokenUsage | None = None


@dataclass(eq=False)
class ConversationTurnBlock:
    """Stable turn identity containing replaceable streaming snapshots."""

    prompt: str
    iterations: list[IterationOutput]
    error: str | None = None
    complete: bool = False


@dataclass(frozen=True)
class StandaloneBlock:
    role: Literal["notice", "error"]
    content: str


HistoryBlock = ConversationTurnBlock | StandaloneBlock


@dataclass(frozen=True)
class HistoryLine:
    role: HistoryRole
    text: str


class ChatUiState:
    """View model consumed by the terminal UI.

    This state deliberately does not own protocol conversation history. The
    controller publishes immutable iteration snapshots here, while
    ``Session`` remains the source of truth for messages sent to the LLM.
    """

    def __init__(self) -> None:
        self.blocks: list[HistoryBlock] = []
        self.pending_prompts: list[str] = []
        self.history_style_version = 0
        self.status = "Ready"

    def add_pending(self, prompt: str) -> None:
        self.pending_prompts.append(prompt)
        self.status = self._queue_status()

    def start_turn(self, prompt: str) -> ConversationTurnBlock:
        self._remove_pending(prompt)
        turn = ConversationTurnBlock(prompt=prompt, iterations=[])
        self.blocks.append(turn)
        self.status = "Streaming response"
        self._touch_history()
        return turn

    def set_iteration(
        self,
        turn: ConversationTurnBlock,
        index: int,
        output: IterationOutput,
    ) -> None:
        if not any(block is turn for block in self.blocks):
            return

        # Streaming may replace the latest snapshot many times, but it may not
        # skip iteration indexes; preserving that invariant keeps rendering
        # deterministic and catches controller ordering bugs early.
        if index < len(turn.iterations):
            turn.iterations[index] = output
        elif index == len(turn.iterations):
            turn.iterations.append(output)
        else:
            raise IndexError("iteration updates must be sequential")

        if output.tools:
            latest = output.tools[-1]
            self.status = (
                f"Running tool: {latest.name}"
                if latest.result is None
                else "Continuing response"
            )
        else:
            self.status = "Streaming response"
        self._touch_history()

    def finish_turn(self, turn: ConversationTurnBlock) -> None:
        if not any(block is turn for block in self.blocks):
            return
        turn.complete = True
        self.status = self._queue_status()
        self._touch_history()

    def fail_turn(self, turn: ConversationTurnBlock, message: str) -> None:
        if not any(block is turn for block in self.blocks):
            return
        turn.error = f"request failed: {message}"
        turn.complete = True
        self.status = "Request failed"
        self._touch_history()

    def add_notice(self, text: str) -> None:
        self.blocks.append(StandaloneBlock("notice", text))
        self._touch_history()

    def add_error(self, text: str) -> None:
        self.blocks.append(StandaloneBlock("error", text))
        self.status = text
        self._touch_history()

    def clear_history(self) -> None:
        self.blocks.clear()
        self.status = self._queue_status()
        self._touch_history()

    def request_exit(self) -> None:
        self.status = "Exiting after queued responses"

    def history_lines(self) -> list[HistoryLine]:
        if not self.blocks:
            return [HistoryLine("notice", "No messages yet.")]

        # Token usage is operational feedback rather than durable transcript
        # content, so only the latest iteration of the latest turn displays it.
        latest_turn = next(
            (
                block
                for block in reversed(self.blocks)
                if isinstance(block, ConversationTurnBlock)
            ),
            None,
        )
        lines: list[HistoryLine] = []
        for block_index, block in enumerate(self.blocks):
            if block_index:
                lines.append(HistoryLine("spacer", ""))
            if isinstance(block, ConversationTurnBlock):
                lines.extend(
                    self._render_turn(
                        block,
                        show_token_usage=block is latest_turn,
                    )
                )
            else:
                label = "Notice: " if block.role == "notice" else "Error: "
                lines.extend(self._labeled_lines(label, block.content, block.role))
        return lines

    def history_text(self) -> str:
        return "\n".join(line.text for line in self.history_lines())

    def _render_turn(
        self,
        turn: ConversationTurnBlock,
        *,
        show_token_usage: bool,
    ) -> list[HistoryLine]:
        lines = self._labeled_lines("You: ", turn.prompt, "user")
        for iteration_index, iteration in enumerate(turn.iterations):
            for tool in iteration.tools:
                lines.extend(
                    self._labeled_lines(
                        f"  Tool call [{tool.name}]: ",
                        tool.arguments,
                        "tool_call",
                    )
                )
                if tool.result is not None:
                    # Reuse the exact ToolMessage held in protocol history so
                    # the UI can never accidentally expose a persisted original
                    # while the model receives only its bounded preview.
                    label = "Tool error" if tool.result.failed else "Tool result"
                    if tool.result.preview is not None:
                        label += " preview"
                    role: HistoryRole = (
                        "error" if tool.result.failed else "tool_result"
                    )
                    lines.extend(
                        self._labeled_lines(
                            f"  {label} [{tool.name}]: ",
                            tool.result.content,
                            role,
                        )
                    )
            if iteration.assistant_text:
                if iteration.tool_requesting:
                    lines.extend(
                        self._labeled_lines(
                            "  Assistant note: ",
                            iteration.assistant_text,
                            "assistant_note",
                        )
                    )
                else:
                    lines.extend(
                        self._labeled_lines(
                            "Assistant: ",
                            iteration.assistant_text,
                            "assistant",
                        )
                    )
            if (
                show_token_usage
                and iteration_index == len(turn.iterations) - 1
                and iteration.token_usage is not None
            ):
                lines.append(
                    HistoryLine(
                        "token_usage",
                        self._format_token_usage(iteration.token_usage),
                    )
                )
        if turn.error is not None:
            lines.extend(self._labeled_lines("Error: ", turn.error, "error"))
        return lines

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
                percentage = round(
                    usage.cached_tokens / usage.prompt_tokens * 100
                )
                cache_text = f"{min(100, max(0, percentage))}% reused"
            parts.append(f"Prompt cache: {cache_text}")
        return " | ".join(parts)

    def render_pending(self) -> str:
        if not self.pending_prompts:
            return "No pending prompts."
        return "\n".join(
            f"{index}. {prompt}"
            for index, prompt in enumerate(self.pending_prompts, start=1)
        )

    def _remove_pending(self, prompt: str) -> None:
        try:
            self.pending_prompts.remove(prompt)
        except ValueError:
            return

    def _queue_status(self) -> str:
        if not self.pending_prompts:
            return "Ready"
        if len(self.pending_prompts) == 1:
            return "1 prompt queued"
        return f"{len(self.pending_prompts)} prompts queued"

    def _touch_history(self) -> None:
        self.history_style_version += 1
