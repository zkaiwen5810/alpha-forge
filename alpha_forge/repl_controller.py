"""REPL queueing, model streaming, tool orchestration, and commands."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, replace

from alpha_forge.chat import ChatClient, ChatStreamEvent
from alpha_forge.config import Config
from alpha_forge.conversation import Conversation, ToolCall
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext
from alpha_forge.tool_results import RawToolResult, ToolResultManager
from alpha_forge.tools import ToolError, ToolRegistry, load_builtin_tools
from alpha_forge.ui_state import (
    ChatUiState,
    ConversationTurnBlock,
    IterationOutput,
    TokenUsage,
    ToolExchange,
)

DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."
MAX_TOOL_ITERATIONS = 10

Redraw = Callable[[], None]
ExitRequest = Callable[[int], None]


@dataclass(frozen=True)
class WorkItem:
    prompt: str


@dataclass
class _PendingToolCall:
    """Accumulator for one streamed tool call, keyed by provider index."""

    call_id: str = ""
    name: str = ""
    arguments: str = ""


class ChatReplController:
    """Coordinate protocol history, background work, tools, and UI snapshots.

    The controller owns application flow but not terminal rendering. It writes
    view-ready snapshots to ``ui_state`` and exposes lifecycle callbacks that a
    concrete UI installs when it starts.
    """

    def __init__(
        self,
        config: Config,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat: ChatClient | None = None,
        command_handler: SlashCommandHandler | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_result_manager: ToolResultManager | None = None,
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
        self.tool_result_manager = (
            tool_result_manager
            if tool_result_manager is not None
            else ToolResultManager()
        )
        self.ui_state = ChatUiState()

        # The queue lets prompt submission remain synchronous and responsive;
        # one consumer serializes model turns so each request sees a complete,
        # ordered protocol history from all earlier requests.
        self.queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        self.exiting = False

        # UI lifecycle hooks default to no-ops for headless tests. A concrete
        # UI later installs redraw and application-exit callbacks without
        # becoming a dependency of the controller.
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

        self.ui_state.add_pending(user_input)
        self.queue.put_nowait(WorkItem(user_input))
        self.request_redraw()

    def request_exit(self) -> None:
        if self.exiting:
            return
        self.exiting = True
        self.ui_state.request_exit()
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
        turn = self.ui_state.start_turn(item.prompt)
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
            self.ui_state.fail_turn(
                turn,
                str(exc) or type(exc).__name__,
            )
            self.request_redraw()

    async def _stream_iteration(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
    ) -> tuple[IterationOutput, tuple[ToolCall, ...]]:
        iteration = IterationOutput()

        # Providers stream tool-call fields as fragments. The stable numeric
        # index, rather than fragment arrival order, identifies which call each
        # delta extends when several calls are requested in one iteration.
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
        first_exchange_index = len(updated.tools)
        raw_results: list[RawToolResult] = []
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
            raw_results.append(
                RawToolResult(
                    tool_call_id=tool_call.id,
                    content=result,
                    failed=failed,
                )
            )

        # Aggregate budgeting requires every raw result from this iteration.
        # Nothing is committed to protocol history until persistence and
        # preview construction succeed, which prevents a failed write from
        # leaving an assistant tool-call message without matching tool outputs.
        tool_messages = self.tool_result_manager.process(tuple(raw_results))
        self.conversation.add_assistant(
            iteration.assistant_text or None,
            tool_calls=tool_calls,
        )
        for tool_message in tool_messages:
            self.conversation.add_tool(tool_message)

        # Publish the same ToolMessage objects that were committed above. This
        # keeps model and UI content identical while allowing results to become
        # visible in call order after the batch budget has been decided.
        for offset, tool_message in enumerate(tool_messages):
            exchange_index = first_exchange_index + offset
            completed_exchange = replace(
                updated.tools[exchange_index],
                result=tool_message,
            )
            tools = list(updated.tools)
            tools[exchange_index] = completed_exchange
            updated = replace(
                updated,
                tools=tuple(tools),
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
        self.ui_state.finish_turn(turn)
        self.request_redraw()

    def _publish_iteration(
        self,
        turn: ConversationTurnBlock,
        iteration_index: int,
        iteration: IterationOutput,
    ) -> None:
        self.ui_state.set_iteration(turn, iteration_index, iteration)
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
            self.ui_state.clear_history()

            # Conversation clearing and persistence rotation form one user
            # boundary: old files remain inspectable, but future previews must
            # never be written into the cleared conversation's session.
            self.tool_result_manager.rotate_session()

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

        self.ui_state.add_error(f"unknown command: {command_name}")
        self.request_redraw()

    def _add_notice(self, text: str) -> None:
        self.ui_state.add_notice(text)
        self.request_redraw()
