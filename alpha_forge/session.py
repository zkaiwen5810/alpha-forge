"""Chat session state and streaming orchestration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from alpha_forge.chat import ChatClient, ChatStreamEvent
from alpha_forge.config import Config
from alpha_forge.conversation import Conversation, ToolCall
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext
from alpha_forge.tools import ToolError, ToolRegistry, load_builtin_tools


DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."
MAX_TOOL_ITERATIONS = 10

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
Redraw = Callable[[], None]
ExitRequest = Callable[[int], None]


@dataclass(frozen=True)
class WorkItem:
    prompt: str


@dataclass(frozen=True)
class ToolExchange:
    name: str
    arguments: str
    result: str | None = None
    failed: bool = False


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class IterationOutput:
    assistant_text: str = ""
    tools: tuple[ToolExchange, ...] = ()
    tool_requesting: bool = False
    token_usage: TokenUsage | None = None


@dataclass(eq=False)
class ConversationTurnBlock:
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


@dataclass
class _PendingToolCall:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class ChatUiState:
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
        turn.complete = True
        self.status = self._queue_status()
        self._touch_history()

    def fail_turn(self, turn: ConversationTurnBlock, message: str) -> None:
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
        self._touch_history()

    def request_exit(self) -> None:
        self.status = "Exiting after queued responses"

    def history_lines(self) -> list[HistoryLine]:
        if not self.blocks:
            return [HistoryLine("notice", "No messages yet.")]

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
                    label = "Tool error" if tool.failed else "Tool result"
                    role: HistoryRole = "error" if tool.failed else "tool_result"
                    lines.extend(
                        self._labeled_lines(
                            f"  {label} [{tool.name}]: ",
                            tool.result,
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


class ChatReplController:
    def __init__(
        self,
        config: Config,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat: ChatClient | None = None,
        command_handler: SlashCommandHandler | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.conversation = Conversation(system_prompt=system_prompt)
        self.chat = chat if chat is not None else ChatClient(config)
        self.command_handler = (
            command_handler if command_handler is not None else SlashCommandHandler()
        )
        self.tool_registry = (
            tool_registry if tool_registry is not None else load_builtin_tools()
        )
        self.state = ChatUiState()
        self.queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        self.exiting = False
        # UI lifecycle hooks. They default to no-ops so the controller can be
        # unit-tested headlessly, then TerminalChatUi installs prompt-toolkit
        # callbacks that redraw or exit the full-screen app.
        self.request_redraw: Redraw = lambda: None
        self.request_app_exit: ExitRequest = lambda _exit_code: None

    def submit(self, user_input: str) -> None:
        if self.exiting:
            return

        text = user_input.strip()
        if not text:
            return

        if text.startswith("/"):
            self._handle_command(text)
            return

        self.state.add_pending(user_input)
        self.queue.put_nowait(WorkItem(user_input))
        self.request_redraw()

    def request_exit(self) -> None:
        if self.exiting:
            return
        self.exiting = True
        self.state.request_exit()
        self.queue.put_nowait(None)
        self.request_redraw()

    async def consume(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    self.request_app_exit(0)
                    return
                await self._stream_response(item)
            finally:
                self.queue.task_done()

    async def _stream_response(self, item: WorkItem) -> None:
        turn = self.state.start_turn(item.prompt)
        self.request_redraw()

        self.conversation.add_user(item.prompt)
        try:
            for iteration_index in range(MAX_TOOL_ITERATIONS):
                iteration, tool_calls = await self._stream_iteration(
                    turn,
                    iteration_index,
                )
                if not tool_calls:
                    self._finish_assistant_turn(
                        turn,
                        iteration_index,
                        iteration,
                    )
                    return

                self.conversation.add_assistant(
                    iteration.assistant_text or None,
                    tool_calls=tool_calls,
                )
                await self._run_tool_calls(
                    turn,
                    iteration_index,
                    iteration,
                    tool_calls,
                )
            raise RuntimeError(
                f"tool iteration limit reached ({MAX_TOOL_ITERATIONS})"
            )
        except Exception as exc:
            self.state.fail_turn(
                turn,
                str(exc) or type(exc).__name__,
            )
            self.request_redraw()
            return

    async def _stream_iteration(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
    ) -> tuple[IterationOutput, tuple[ToolCall, ...]]:
        iteration = IterationOutput()
        pending_calls: dict[int, _PendingToolCall] = {}

        async for event in self.chat.stream_response(
            self.conversation.messages,
            tools=self.tool_registry.definitions(),
        ):
            if event.type == "text_delta":
                iteration = self._add_text_delta(
                    turn,
                    iteration_index,
                    iteration,
                    event.text,
                )
            elif event.type == "tool_call_delta":
                iteration = self._add_tool_call_delta(
                    turn,
                    iteration_index,
                    iteration,
                    pending_calls,
                    event,
                )
            elif (
                event.prompt_tokens is not None
                or event.cached_tokens is not None
                or event.total_tokens is not None
            ):
                iteration = replace(
                    iteration,
                    token_usage=TokenUsage(
                        prompt_tokens=event.prompt_tokens,
                        cached_tokens=event.cached_tokens,
                        total_tokens=event.total_tokens,
                    ),
                )
                self._publish_iteration(turn, iteration_index, iteration)

        return iteration, self._build_tool_calls(pending_calls)

    def _add_text_delta(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
        iteration: IterationOutput,
        text: str,
    ) -> IterationOutput:
        updated = replace(
            iteration,
            assistant_text=iteration.assistant_text + text,
        )
        self._publish_iteration(turn, iteration_index, updated)
        return updated

    def _add_tool_call_delta(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
        iteration: IterationOutput,
        pending_calls: dict[int, _PendingToolCall],
        event: ChatStreamEvent,
    ) -> IterationOutput:
        if event.index is None:
            raise RuntimeError("tool-call delta is missing its index")

        updated = iteration
        if not updated.tool_requesting:
            updated = replace(updated, tool_requesting=True)
            self._publish_iteration(turn, iteration_index, updated)

        pending = pending_calls.setdefault(event.index, _PendingToolCall())
        pending.call_id += event.call_id
        pending.name += event.name
        pending.arguments += event.arguments
        return updated

    @staticmethod
    def _build_tool_calls(
        pending_calls: dict[int, _PendingToolCall],
    ) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(
                id=pending.call_id,
                name=pending.name,
                arguments=pending.arguments,
            )
            for _, pending in sorted(pending_calls.items())
        )

    async def _run_tool_calls(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
        iteration: IterationOutput,
        tool_calls: tuple[ToolCall, ...],
    ) -> IterationOutput:
        updated = iteration
        for tool_call in tool_calls:
            exchange = ToolExchange(
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
            updated = replace(
                updated,
                tools=(*updated.tools, exchange),
            )
            self._publish_iteration(turn, iteration_index, updated)
            await asyncio.sleep(0)

            result, failed = self._execute_tool_call(tool_call)
            self.conversation.add_tool(result, tool_call_id=tool_call.id)
            completed_exchange = replace(
                exchange,
                result=result,
                failed=failed,
            )
            updated = replace(
                updated,
                tools=(
                    *updated.tools[:-1],
                    completed_exchange,
                ),
            )
            self._publish_iteration(turn, iteration_index, updated)
        return updated

    def _finish_assistant_turn(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
        iteration: IterationOutput,
    ) -> None:
        self._publish_iteration(turn, iteration_index, iteration)
        self.conversation.add_assistant(iteration.assistant_text)
        self.state.finish_turn(turn)
        self.request_redraw()

    def _publish_iteration(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
        iteration: IterationOutput,
    ) -> None:
        self.state.set_iteration(turn, iteration_index, iteration)
        self.request_redraw()

    def _execute_tool_call(self, tool_call: ToolCall) -> tuple[str, bool]:
        try:
            arguments = json.loads(tool_call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to a JSON object")
            return self.tool_registry.execute(tool_call.name, arguments), False
        except (json.JSONDecodeError, ValueError, ToolError) as exc:
            return f"error: {exc}", True

    def _handle_command(self, text: str) -> None:
        command_name = text.split(maxsplit=1)[0]
        if command_name == "/clear":
            self.state.clear_history()

        command_result = self.command_handler.handle(
            text,
            CommandContext(
                config=self.config,
                conversation=self.conversation,
                chat=self.chat,
                print_text=self._add_notice,
            ),
        )
        if command_result.exit_requested:
            self.request_exit()
            return
        if command_result.handled:
            self.request_redraw()
            return

        self.state.add_error(f"unknown command: {command_name}")
        self.request_redraw()

    def _add_notice(self, text: str) -> None:
        self.state.add_notice(text)
        self.request_redraw()
