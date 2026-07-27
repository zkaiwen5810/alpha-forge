"""FIFO REPL orchestration over stateless query and stateful session services."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config
from alpha_forge.events import EventRouter
from alpha_forge.prompt_editor import PromptEditor, ToolResultPromptEditor
from alpha_forge.query import (
    MAX_TOOL_ROUNDS,
    ModelDeltaReceived,
    ModelRoundCompleted,
    ModelRoundStarted,
    QueryEngine,
    QueryEvent,
    QueryRequest,
    ToolBatchStarted as QueryToolBatchStarted,
    ToolExecutionStarted,
    ToolResultProduced,
    ToolResultsEdited,
)
from alpha_forge.session import DEFAULT_SYSTEM_PROMPT, Session
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext, CommandOutcome
from alpha_forge.system_events import (
    AssistantMessageAdded,
    ExitReady,
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
    ToolResultRecorded,
    ToolStarted,
)
from alpha_forge.tool_execution import ToolExecutor
from alpha_forge.tools import (
    Tool,
    ToolNotFoundError,
    ToolRegistry,
    load_builtin_tools,
)
from alpha_forge.transcript import CommandMessage, TranscriptError
from alpha_forge.ui_history import UiHistoryProjector


@dataclass(frozen=True, slots=True)
class PromptInput:
    item_id: str
    content: str
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommandInput:
    item_id: str
    raw: str


@dataclass(frozen=True, slots=True)
class ShutdownInput:
    item_id: str


type UserInput = PromptInput | CommandInput
type QueueItem = UserInput | ShutdownInput


class QueryPersistenceError(RuntimeError):
    """Raised when a durable query fact cannot be committed."""


class ChatReplController:
    """Serialize user inputs and commit query facts before publishing history."""

    def __init__(
        self,
        config: Config,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat: ChatClient | None = None,
        command_handler: SlashCommandHandler | None = None,
        tool_registry: ToolRegistry | None = None,
        session: Session | None = None,
        prompt_editor: PromptEditor | None = None,
        query: QueryEngine | None = None,
    ) -> None:
        self.config = config
        self.session = session or Session(system_prompt=system_prompt)
        self.chat = chat if chat is not None else ChatClient(config)
        self.command_handler = command_handler or SlashCommandHandler()
        self.tool_registry = tool_registry or load_builtin_tools()
        self.prompt_editor = prompt_editor or ToolResultPromptEditor()
        self.query = query or QueryEngine(
            self.chat,
            prompt_editor=self.prompt_editor,
        )
        self.events = EventRouter()
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._recovery: deque[PromptInput] = deque()
        self._accepting = True
        self._shutdown_enqueued = False
        self._persistence_halted = False
        self._active_turn_id: str | None = None

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def initial_view(self) -> SessionView:
        return self._session_view()

    def submit(self, user_input: str) -> None:
        if not self._accepting:
            return
        text = user_input.strip()
        if not text:
            return
        item = self._parse_input(user_input)
        self.queue.put_nowait(item)
        raw = item.raw if isinstance(item, CommandInput) else item.content
        self.events.publish(InputQueued(item.item_id, raw))
        if isinstance(item, CommandInput) and (
            text.split(maxsplit=1)[0] in ("/exit", "/quit")
        ):
            self._accepting = False
            self._shutdown_enqueued = True
            self.events.publish(ExitRequested())

    def request_exit(self) -> None:
        if self._shutdown_enqueued:
            return
        self._accepting = False
        self._shutdown_enqueued = True
        item = ShutdownInput(uuid4().hex)
        self.queue.put_nowait(item)
        self.events.publish(ExitRequested())

    async def consume(self) -> None:
        while True:
            from_queue = not self._recovery
            item: QueueItem
            if self._recovery:
                item = self._recovery.popleft()
            else:
                item = await self.queue.get()
            try:
                self.events.publish(InputStarted(item.item_id))
                if isinstance(item, ShutdownInput):
                    self.events.publish(ExitReady())
                    return
                if self._persistence_halted:
                    if (
                        isinstance(item, CommandInput)
                        and item.raw.strip().split(maxsplit=1)[0]
                        in ("/exit", "/quit")
                    ):
                        self.events.publish(ExitReady())
                        return
                    continue
                should_exit = await self.handle_input(item)
                if should_exit:
                    self.events.publish(ExitReady())
                    return
            finally:
                if from_queue:
                    self.queue.task_done()

    async def handle_input(self, item: UserInput) -> bool:
        """Handle one dequeued input through the shared persistence pipeline."""
        if isinstance(item, PromptInput):
            await self._handle_prompt(item.content, turn_id=item.turn_id)
            return False
        command_name = item.raw.strip().split(maxsplit=1)[0]
        handled_exit = await self._handle_command(item.raw)
        return handled_exit or command_name in ("/exit", "/quit")

    async def _handle_prompt(
        self,
        content: str,
        *,
        turn_id: str | None = None,
    ) -> None:
        session = self.session
        try:
            resolved_turn = turn_id or session.submit_user(content)
        except Exception as exc:
            self._halt_for_persistence_failure("user input", exc)
            return
        self._active_turn_id = resolved_turn
        self._publish_view()
        registry = self._query_registry(session)
        request = QueryRequest(
            messages=tuple(session.query_messages_for_turn(resolved_turn)),
            tool_definitions=tuple(registry.definitions()),
            tool_executor=ToolExecutor(registry),
        )
        try:
            async with aclosing(self.query.run(request)) as query_events:
                async for event in query_events:
                    self._handle_query_event(session, resolved_turn, event)
        except QueryPersistenceError:
            return
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            try:
                session.fail_turn(resolved_turn, message)
                self._publish_view()
            except Exception as persistence_exc:
                self._halt_for_persistence_failure(
                    "request failure",
                    persistence_exc,
                )
            self.events.publish(RequestFailed(message))
        finally:
            self._active_turn_id = None

    def _handle_query_event(
        self,
        session: Session,
        turn_id: str,
        event: QueryEvent,
    ) -> None:
        if isinstance(event, ModelRoundStarted):
            self.events.publish(ModelResponseStarted(turn_id, event.output_id))
            return
        if isinstance(event, ModelDeltaReceived):
            self.events.publish(event.delta)
            return
        if isinstance(event, ModelRoundCompleted):
            self.events.publish(
                ModelResponseCompleted(event.output_id, event.response)
            )
            self._commit_query_event(session, turn_id, event, "model output")
            self._publish_view()
            self.events.publish(AssistantMessageAdded(event.output_id))
            return
        if isinstance(event, QueryToolBatchStarted):
            self.events.publish(
                ToolBatchStarted(
                    turn_id,
                    event.output_id,
                    event.calls,
                    event.results,
                )
            )
            return
        if isinstance(event, ToolExecutionStarted):
            self.events.publish(ToolStarted(event.call.id))
            return
        if isinstance(event, ToolResultProduced):
            self._commit_query_event(session, turn_id, event, "raw tool result")
            self.events.publish(ToolResultRecorded(event.result))
            return
        if isinstance(event, ToolResultsEdited):
            self._commit_query_event(session, turn_id, event, "prompt edit")
            self._publish_view(reset_active=True)

    def _commit_query_event(
        self,
        session: Session,
        turn_id: str,
        event: QueryEvent,
        stage: str,
    ) -> None:
        try:
            session.apply_query_event(turn_id=turn_id, event=event)
        except Exception as exc:
            self._halt_for_persistence_failure(stage, exc)
            raise QueryPersistenceError(str(exc)) from exc

    async def _handle_command(self, text: str) -> bool:
        source = self.session
        parsed = self.command_handler.parse(text)
        try:
            command = source.add_command(
                raw=parsed.raw,
                name=parsed.name,
                arguments=parsed.arguments,
            )
        except Exception as exc:
            self._halt_for_persistence_failure("command", exc)
            return False
        self._publish_view()

        try:
            outcome = await asyncio.to_thread(
                self.command_handler.execute,
                parsed,
                CommandContext(
                    current_model=self.config.model,
                    model_catalog=self.chat,
                ),
            )
        except Exception as exc:
            outcome = CommandOutcome(
                status="error",
                messages=(CommandMessage(f"command failed: {exc}", "error"),),
            )

        if outcome.action in ("clear", "resume"):
            outcome = self._switch_session(
                source,
                command.command_id,
                outcome,
                resume_path=(
                    parsed.arguments if outcome.action == "resume" else None
                ),
            )

        try:
            source.add_command_result(
                command.command_id,
                status=outcome.status,
                messages=outcome.messages,
            )
        except Exception as exc:
            self._halt_for_persistence_failure("command result", exc)
        else:
            if self.session is source:
                self._publish_view()

        if outcome.status == "error" and outcome.messages:
            self.events.publish(StatusChanged(outcome.messages[-1].content))
        return outcome.action == "exit" and outcome.status == "success"

    def _halt_for_persistence_failure(
        self,
        stage: str,
        error: Exception,
    ) -> None:
        self._accepting = False
        self._persistence_halted = True
        message = str(error) or type(error).__name__
        self.events.publish(PersistenceFailed(stage, message))

    def _switch_session(
        self,
        source: Session,
        command_id: str,
        outcome: CommandOutcome,
        *,
        resume_path: str | None,
    ) -> CommandOutcome:
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
        self._recovery.extend(
            PromptInput(uuid4().hex, turn.content, turn.turn_id)
            for turn in pending
        )
        self._publish_view(reset_active=True)
        return outcome

    def _query_registry(self, session: Session) -> ToolRegistry:
        registry = self.tool_registry.copy()
        try:
            registry.get("tool_result_reader")
        except ToolNotFoundError:
            registry.register(self._transcript_result_reader(session))
        return registry

    @staticmethod
    def _transcript_result_reader(session: Session) -> Tool:
        def read(arguments: Mapping[str, object]) -> str:
            result_id = arguments.get("result_id")
            if not isinstance(result_id, str) or not result_id:
                raise ValueError("result_id must be a non-empty string")
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 16_000)
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ValueError("offset must be an integer")
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            return session.read_tool_result(
                result_id,
                offset=offset,
                limit=limit,
            )

        return Tool(
            name="tool_result_reader",
            function=read,
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

    def _session_view(self) -> SessionView:
        return SessionView(
            session_id=self.session.session_id,
            revision=self.session.transcript.revision,
            head_turn_id=self.session.head_turn_id,
            items=tuple(
                UiHistoryProjector(self.session.transcript).items(
                    head_turn_id=self.session.head_turn_id
                )
            ),
        )

    @staticmethod
    def _parse_input(raw: str) -> UserInput:
        item_id = uuid4().hex
        if raw.strip().startswith("/"):
            return CommandInput(item_id, raw)
        return PromptInput(item_id, raw)

    def _publish_view(self, *, reset_active: bool = False) -> None:
        self.events.publish(
            SessionViewChanged(
                self._session_view(),
                reset_active=reset_active,
            )
        )


__all__ = [
    "MAX_TOOL_ROUNDS",
    "ChatReplController",
    "CommandInput",
    "PromptInput",
    "QueryPersistenceError",
    "ShutdownInput",
]
