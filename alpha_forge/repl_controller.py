"""REPL orchestration across session, ephemeral UI, tools, and commands."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config
from alpha_forge.events import Event, EventRouter
from alpha_forge.models import ToolCall
from alpha_forge.session import DEFAULT_SYSTEM_PROMPT, Session
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import (
    CommandContext,
    CommandOutcome,
)
from alpha_forge.streaming import StreamCompleted
from alpha_forge.system_events import (
    AssistantMessageAdded,
    AssistantMessageAddFailed,
    ExitRequested,
    ModelResponseStarted,
    RequestFailed,
    SessionSelected,
    StatusChanged,
    ToolBatchStarted,
    ToolResultsAddFailed,
    ToolResultsFinalized,
    ToolResultsUpdated,
    ToolStarted,
    TranscriptUpdated,
)
from alpha_forge.tools import (
    Tool,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    load_builtin_tools,
)
from alpha_forge.transcript import (
    CommandMessage,
    ModelOutput,
    TranscriptError,
)
from alpha_forge.ui_state import ChatUiState

MAX_TOOL_ROUNDS = 10

Redraw = Callable[[], None]
ExitRequest = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class WorkItem:
    prompt: str
    turn_id: str | None = None


class AssistantMessageAddError(RuntimeError):
    """Raised after a completed response could not be added to Session."""


class ToolResultsAddError(RuntimeError):
    """Raised after completed tool data could not be added to Session."""


@dataclass(slots=True)
class _SessionAssistantConsumer:
    session: Session
    turn_id: str
    events: EventRouter
    output: ModelOutput | None = None

    def handle(self, event: StreamCompleted) -> None:
        try:
            self.output = self.session.add_assistant_message(
                turn_id=self.turn_id,
                response=event.response,
            )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self.events.publish(AssistantMessageAddFailed(message))
            raise AssistantMessageAddError(message) from exc
        self.events.publish(
            AssistantMessageAdded(
                self.output,
                self.session.head_turn_id,
            )
        )

    def result(self) -> ModelOutput:
        if self.output is None:
            raise RuntimeError("model stream ended without a completed response")
        return self.output


class ChatReplController:
    """Route completed activities through Session and deltas through UI state."""

    def __init__(
        self,
        config: Config,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat: ChatClient | None = None,
        command_handler: SlashCommandHandler | None = None,
        tool_registry: ToolRegistry | None = None,
        session: Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or Session(system_prompt=system_prompt)
        self.chat = chat if chat is not None else ChatClient(config)
        self.command_handler = (
            command_handler if command_handler is not None else SlashCommandHandler()
        )
        self.tool_registry = (
            tool_registry if tool_registry is not None else load_builtin_tools()
        )
        self._register_transcript_result_reader()
        self.request_redraw: Redraw = lambda: None
        self.request_app_exit: ExitRequest = lambda _exit_code: None
        self.ui_state = ChatUiState(
            self.session.transcript,
            head_turn_id=self.session.head_turn_id,
        )
        self.events = EventRouter()
        self.events.subscribe(Event, self._apply_ui_event)
        self.queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        self.exiting = False
        self._active_turn_id: str | None = None

    def _apply_ui_event(self, event: Event) -> None:
        if self.ui_state.handle(event):
            self.request_redraw()

    def submit(self, user_input: str) -> None:
        if self.exiting:
            return
        text = user_input.strip()
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(user_input)
            return
        if self.ui_state.has_unsaved_active:
            self.events.publish(
                StatusChanged("Cannot continue while completed output is not added")
            )
            return
        try:
            turn_id = self.session.submit_user(user_input)
        except Exception as exc:
            self.events.publish(StatusChanged(f"Cannot add prompt: {exc}"))
            return
        self.queue.put_nowait(WorkItem(user_input, turn_id))
        self.events.publish(TranscriptUpdated(self.session.head_turn_id))

    def request_exit(self) -> None:
        if self.exiting:
            return
        self.exiting = True
        self.events.publish(ExitRequested())
        self.queue.put_nowait(None)

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
        session = self.session
        turn_id = item.turn_id or session.submit_user(item.prompt)
        self._active_turn_id = turn_id

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                output = await self._stream_model_response(
                    session,
                    turn_id,
                )
                if not output.tool_calls:
                    return

                self.events.publish(
                    ToolBatchStarted(
                        turn_id,
                        output.output_id,
                        output.tool_calls,
                    )
                )
                await self._run_tool_calls(
                    session,
                    output.output_id,
                    output.tool_calls,
                )
            raise RuntimeError(f"tool round limit reached ({MAX_TOOL_ROUNDS})")
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            try:
                session.fail_turn(turn_id, message)
            except Exception:
                pass
            if not isinstance(
                exc,
                (AssistantMessageAddError, ToolResultsAddError),
            ):
                self.events.publish(RequestFailed(message))
        finally:
            self._active_turn_id = None

    async def _stream_model_response(
        self,
        session: Session,
        turn_id: str,
    ) -> ModelOutput:
        consumer = _SessionAssistantConsumer(
            session,
            turn_id,
            self.events,
        )
        with self.events.subscribe(StreamCompleted, consumer.handle):
            self.events.publish(ModelResponseStarted(turn_id))
            async for event in self.chat.stream_response(
                session.messages_for_turn(turn_id),
                tools=self.tool_registry.definitions(),
            ):
                self.events.publish(event)
        return consumer.result()

    async def _run_tool_calls(
        self,
        session: Session,
        output_id: str,
        tool_calls: tuple[ToolCall, ...],
    ) -> None:
        completed = []
        for tool_call in tool_calls:
            self.events.publish(ToolStarted(tool_call.id))
            await asyncio.sleep(0)
            content, failed = self._execute_tool_call(tool_call)
            try:
                result = session.add_tool_result(
                    output_id=output_id,
                    call_id=tool_call.id,
                    content=content,
                    failed=failed,
                )
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                self.events.publish(ToolResultsAddFailed(message))
                raise ToolResultsAddError(message) from exc
            completed.append(result)
            decisions = session.provisional_tool_decisions(tuple(completed))
            self.events.publish(
                ToolResultsUpdated(
                    tuple(completed),
                    decisions,
                )
            )
            await asyncio.sleep(0)

        try:
            session.finalize_tool_results(
                output_id=output_id,
                results=tuple(completed),
            )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self.events.publish(ToolResultsAddFailed(message))
            raise ToolResultsAddError(message) from exc
        self.events.publish(ToolResultsFinalized(session.head_turn_id))

    def _execute_tool_call(self, tool_call: ToolCall) -> tuple[str, bool]:
        try:
            arguments = json.loads(tool_call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to a JSON object")
            return self.tool_registry.execute(tool_call.name, arguments), False
        except (json.JSONDecodeError, ValueError, ToolError) as exc:
            return f"error: {exc}", True

    def _handle_command(self, text: str) -> None:
        source = self.session
        parsed = self.command_handler.parse(text)
        try:
            command = source.add_command(
                raw=parsed.raw,
                name=parsed.name,
                arguments=parsed.arguments,
            )
        except Exception as exc:
            self.events.publish(StatusChanged(f"Cannot add command: {exc}"))
            return

        try:
            outcome = self.command_handler.execute(
                parsed,
                CommandContext(config=self.config, chat=self.chat),
            )
        except Exception as exc:
            outcome = CommandOutcome(
                status="error",
                messages=(
                    CommandMessage(
                        f"command failed: {exc}",
                        "error",
                    ),
                ),
            )

        if outcome.action in ("clear", "resume"):
            outcome = self._switch_session(
                source,
                command.command_id,
                outcome,
                resume_path=(parsed.arguments if outcome.action == "resume" else None),
            )
        save_error = self._record_command_outcome(
            source,
            command.command_id,
            outcome,
        )

        if outcome.action == "exit" and outcome.status == "success":
            self.request_exit()
            return
        self.events.publish(TranscriptUpdated(self.session.head_turn_id))
        if save_error is not None:
            self.events.publish(StatusChanged(save_error))
        elif outcome.status == "error" and outcome.messages:
            self.events.publish(StatusChanged(outcome.messages[-1].content))

    def _switch_session(
        self,
        source: Session,
        command_id: str,
        outcome: CommandOutcome,
        *,
        resume_path: str | None,
    ) -> CommandOutcome:
        if not self._can_switch_session():
            action = "resume" if resume_path is not None else "clear"
            return CommandOutcome(
                status="error",
                messages=(
                    CommandMessage(
                        f"cannot {action} while a response is active or "
                        "prompts are queued",
                        "error",
                    ),
                ),
            )
        try:
            if resume_path is None:
                destination = source.fresh()
                pending = []
                kind = "clear"
            else:
                destination = Session.resume(Path(resume_path))
                pending = destination.recover_unfinished_turns()
                kind = "resume"
            destination.add_session_transition(
                kind=kind,
                source_session_id=source.session_id,
                source_command_id=command_id,
            )
        except (OSError, ValueError, TranscriptError) as exc:
            return CommandOutcome(
                status="error",
                messages=(
                    CommandMessage(
                        f"cannot {outcome.action} transcript: {exc}",
                        "error",
                    ),
                ),
            )

        self.session = destination
        self.events.publish(
            SessionSelected(
                destination.transcript,
                destination.head_turn_id,
            )
        )
        for turn in pending:
            self.queue.put_nowait(WorkItem(turn.content, turn.turn_id))
        return outcome

    def _record_command_outcome(
        self,
        source: Session,
        command_id: str,
        outcome: CommandOutcome,
    ) -> str | None:
        try:
            source.add_command_result(
                command_id,
                status=outcome.status,
                messages=outcome.messages,
            )
        except Exception as exc:
            return f"Cannot add command result: {exc}"
        return None

    def _can_switch_session(self) -> bool:
        return (
            self._active_turn_id is None
            and self.queue.empty()
            and not self.ui_state.has_unsaved_active
        )

    def _register_transcript_result_reader(self) -> None:
        try:
            self.tool_registry.get("tool_result_reader")
            return
        except ToolNotFoundError:
            pass
        self.tool_registry.register(
            Tool(
                name="tool_result_reader",
                function=self._read_transcript_result,
                description="Read a complete result stored in this transcript.",
                prompt=(
                    "Read a raw tool result referenced by transcript_ref. "
                    "Use offset and limit to page through large results."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "result_id": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 16_000,
                        },
                    },
                    "required": ["result_id"],
                    "additionalProperties": False,
                },
            )
        )

    def _read_transcript_result(self, arguments: Mapping[str, object]) -> str:
        result_id = arguments.get("result_id")
        if not isinstance(result_id, str) or not result_id:
            raise ValueError("result_id must be a non-empty string")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 16_000)
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError("offset must be an integer")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        return self.session.read_tool_result(
            result_id,
            offset=offset,
            limit=limit,
        )


__all__ = ["MAX_TOOL_ROUNDS", "ChatReplController", "WorkItem"]
