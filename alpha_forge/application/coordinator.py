"""FIFO application coordinator for input, persistence, query, and UI flow."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from alpha_forge.application.events import (
    ExitReady,
    ExitRequested,
    InputQueued,
    InputStarted,
    ModelOutputRecorded,
    PersistenceFailed,
    ProviderDeltaReceived as ApplicationProviderDeltaReceived,
    ProviderRequestStarted as ApplicationProviderRequestStarted,
    ProviderResponseCompleted as ApplicationProviderResponseCompleted,
    RequestFailed,
    SessionView,
    SessionViewChanged,
    StatusChanged,
    ToolResultRecorded,
    ToolStarted,
)
from alpha_forge.config import Config
from alpha_forge.context.pipeline import ContextPipeline
from alpha_forge.context.tool_result_budget import ToolResultBudgetPolicy
from alpha_forge.events import EventRouter
from alpha_forge.projectors.session_state import OpenQuery
from alpha_forge.providers.base import ModelProvider
from alpha_forge.providers.openai_chat import OpenAIChatAdapter
from alpha_forge.query import (
    CommitModelOutput,
    CommitToolResult,
    ContextPrepared,
    ModelOutputCommitted,
    PendingToolContinuation,
    PrepareContext,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    QueryCompleted,
    QueryEffect,
    QueryEngine,
    QueryExecutionError,
    QueryRequest,
    ToolExecutionStarted,
    ToolResultCommitted,
)
from alpha_forge.sessions import DEFAULT_SYSTEM_PROMPT, Session
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext, CommandOutcome
from alpha_forge.tools import (
    ToolExecutor,
    ToolNotFoundError,
    ToolRegistry,
    load_builtin_tools,
)
from alpha_forge.transcript import (
    CommandMessage,
    QueryFailureStage,
    TranscriptError,
    TranscriptPersistenceError,
)


@dataclass(frozen=True, slots=True)
class PromptInput:
    item_id: str
    content: str


@dataclass(frozen=True, slots=True)
class CommandInput:
    item_id: str
    raw: str


@dataclass(frozen=True, slots=True)
class RecoveryInput:
    item_id: str
    continuation: OpenQuery


@dataclass(frozen=True, slots=True)
class ShutdownInput:
    item_id: str


type UserInput = PromptInput | CommandInput
type QueueItem = UserInput | ShutdownInput


class ApplicationCoordinator:
    """Own data flow; domain modules own their state and transformations."""

    def __init__(
        self,
        config: Config,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        provider: ModelProvider | None = None,
        command_handler: SlashCommandHandler | None = None,
        tool_registry: ToolRegistry | None = None,
        session: Session | None = None,
        context_pipeline: ContextPipeline | None = None,
        query: QueryEngine | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or OpenAIChatAdapter(config)
        self.command_handler = command_handler or SlashCommandHandler()
        self.tool_registry = tool_registry or load_builtin_tools()
        self.session = session or Session.create(system_prompt=system_prompt)
        self.context_pipeline = context_pipeline or ContextPipeline(
            (ToolResultBudgetPolicy(),)
        )
        self.query = query or QueryEngine(self.provider)
        self.events = EventRouter()
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._recovery: deque[RecoveryInput] = deque()
        self._accepting = True
        self._shutdown_enqueued = False
        self._persistence_halted = False
        self._schedule_recovery(self.session)

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def initial_view(self) -> SessionView:
        return self._session_view()

    def submit(self, user_input: str) -> None:
        if not self._accepting or not user_input.strip():
            return
        item = self._parse_input(user_input)
        self.queue.put_nowait(item)
        raw = item.raw if isinstance(item, CommandInput) else item.content
        self.events.publish(InputQueued(item.item_id, raw))
        if isinstance(item, CommandInput) and (
            item.raw.strip().split(maxsplit=1)[0] in ("/exit", "/quit")
        ):
            self._accepting = False
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
        try:
            while True:
                from_queue = not self._recovery
                item: QueueItem | RecoveryInput
                if self._recovery:
                    item = self._recovery.popleft()
                else:
                    item = await self.queue.get()
                try:
                    self.events.publish(InputStarted(item.item_id))
                    if isinstance(item, ShutdownInput):
                        self._shutdown_enqueued = True
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
                    if isinstance(item, RecoveryInput):
                        await self._run_query(item.continuation)
                        continue
                    if isinstance(item, PromptInput):
                        await self._handle_prompt(item.content)
                        continue
                    if await self._handle_command(item.raw):
                        self._shutdown_enqueued = True
                        self.events.publish(ExitReady())
                        return
                finally:
                    if from_queue:
                        self.queue.task_done()
        finally:
            self.session.close()

    async def _handle_prompt(self, content: str) -> None:
        try:
            self.session.accept_prompt(content)
        except Exception as exc:
            self._halt_for_persistence_failure("user input", exc)
            return
        self._publish_view()
        continuation = self.session.open_query()
        if continuation is None:
            self._record_request_failure(
                prompt_event_id=None,
                stage="internal",
                message="accepted prompt did not create an open query",
            )
            return
        await self._run_query(continuation)

    async def _run_query(self, continuation: OpenQuery) -> None:
        registry = self._query_registry(self.session)
        request = QueryRequest(
            prompt_event_id=continuation.prompt_event_id,
            pending_tool_continuation=(
                PendingToolContinuation(
                    continuation.pending_tool_batch.model_output_event_id,
                    continuation.pending_tool_batch.missing_calls,
                )
                if continuation.pending_tool_batch is not None
                else None
            ),
            completed_tool_rounds=continuation.completed_tool_rounds,
            tool_specs=registry.specs(),
            tool_executor=ToolExecutor(registry),
        )
        feedback = None
        try:
            async with aclosing(self.query.run(request)) as query_events:
                while True:
                    try:
                        event = await query_events.asend(feedback)
                    except StopAsyncIteration:
                        break
                    feedback = None
                    if isinstance(event, QueryEffect):
                        feedback = self._handle_query_effect(event)
                    else:
                        self._publish_query_progress(event)
        except TranscriptPersistenceError as exc:
            self._halt_for_persistence_failure("query", exc)
        except QueryExecutionError as exc:
            self._record_request_failure(
                prompt_event_id=continuation.prompt_event_id,
                stage=exc.stage,
                message=str(exc),
            )
        except Exception as exc:
            self._record_request_failure(
                prompt_event_id=continuation.prompt_event_id,
                stage="internal",
                message=str(exc) or type(exc).__name__,
            )

    def _handle_query_effect(self, effect: QueryEffect):
        if isinstance(effect, PrepareContext):
            before = self.session.revision
            try:
                snapshot = self.session.prepare_context(self.context_pipeline)
            except TranscriptPersistenceError:
                raise
            except Exception as exc:
                raise QueryExecutionError(
                    "context",
                    str(exc) or type(exc).__name__,
                ) from exc
            if self.session.revision != before:
                self._publish_view()
            return ContextPrepared(snapshot)
        if isinstance(effect, CommitModelOutput):
            record = self.session.record_model_output(
                effect.prompt_event_id,
                effect.output,
            )
            self._publish_view(reset_active=True)
            self.events.publish(ModelOutputRecorded(record.event_id))
            return ModelOutputCommitted(record.event_id, self.session.revision)
        if isinstance(effect, CommitToolResult):
            record = self.session.record_tool_result(
                model_output_event_id=effect.model_output_event_id,
                call_id=effect.call_id,
                status=effect.status,
                content=effect.content,
            )
            self._publish_view(reset_active=True)
            self.events.publish(
                ToolResultRecorded(
                    record.event_id,
                    effect.model_output_event_id,
                    effect.call_id,
                )
            )
            return ToolResultCommitted(record.event_id, self.session.revision)
        raise QueryExecutionError(
            "internal",
            f"unsupported query effect: {type(effect).__name__}",
        )

    def _publish_query_progress(self, event: object) -> None:
        if isinstance(event, ProviderRequestStarted):
            self.events.publish(
                ApplicationProviderRequestStarted(
                    event.prompt_event_id,
                    event.request_id,
                )
            )
        elif isinstance(event, ProviderDeltaReceived):
            self.events.publish(
                ApplicationProviderDeltaReceived(event.request_id, event.delta)
            )
        elif isinstance(event, ProviderResponseCompleted):
            self.events.publish(
                ApplicationProviderResponseCompleted(
                    event.request_id,
                    event.output,
                )
            )
        elif isinstance(event, ToolExecutionStarted):
            self.events.publish(
                ToolStarted(event.model_output_event_id, event.call)
            )
        elif isinstance(event, QueryCompleted):
            self.events.publish(StatusChanged("Ready"))

    async def _handle_command(self, text: str) -> bool:
        source = self.session
        parsed = self.command_handler.parse(text)
        try:
            command_record = source.accept_command(
                text=parsed.raw,
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
                    model_catalog=self.provider,
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
                command_record.event_id,
                outcome,
                resume_path=(
                    parsed.arguments if outcome.action == "resume" else None
                ),
            )

        try:
            source.complete_command(
                command_record.event_id,
                status=outcome.status,
                messages=outcome.messages,
            )
        except Exception as exc:
            self._halt_for_persistence_failure("command result", exc)
            if self.session is not source:
                source.close()
        else:
            if self.session is source:
                self._publish_view()
            else:
                source.close()

        if outcome.status == "error" and outcome.messages:
            self.events.publish(StatusChanged(outcome.messages[-1].content))
        return outcome.action == "exit" and outcome.status == "success"

    def _switch_session(
        self,
        source: Session,
        command_event_id: str,
        outcome: CommandOutcome,
        *,
        resume_path: str | None,
    ) -> CommandOutcome:
        destination: Session | None = None
        try:
            if resume_path is None:
                destination = source.fresh()
                kind = "clear"
            else:
                destination = Session.resume(Path(resume_path))
                kind = "resume"
            destination.link(
                kind=kind,
                source_session_id=source.session_id,
                source_command_event_id=command_event_id,
            )
        except (OSError, ValueError, TranscriptError) as exc:
            if destination is not None:
                destination.close()
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
        self._schedule_recovery(destination)
        self._publish_view(reset_active=True)
        return outcome

    def _query_registry(self, session: Session) -> ToolRegistry:
        registry = self.tool_registry.copy()
        try:
            registry.get("tool_result_reader")
        except ToolNotFoundError:
            registry.register(session.tool_result_reader())
        return registry

    def _record_request_failure(
        self,
        *,
        prompt_event_id: str | None,
        stage: QueryFailureStage,
        message: str,
    ) -> None:
        if prompt_event_id is not None:
            try:
                self.session.fail_query(
                    prompt_event_id,
                    stage=stage,
                    message=message,
                )
                self._publish_view(reset_active=True)
            except Exception as exc:
                self._halt_for_persistence_failure("request failure", exc)
                return
        self.events.publish(RequestFailed(message))

    def _halt_for_persistence_failure(
        self,
        stage: str,
        error: Exception,
    ) -> None:
        self._accepting = False
        self._persistence_halted = True
        message = str(error) or type(error).__name__
        self.events.publish(PersistenceFailed(stage, message))

    def _schedule_recovery(self, session: Session) -> None:
        continuation = session.open_query()
        if continuation is not None:
            self._recovery.append(
                RecoveryInput(uuid4().hex, continuation)
            )

    def _session_view(self) -> SessionView:
        return SessionView(
            self.session.session_id,
            self.session.revision,
            self.session.ui_history(),
        )

    def _publish_view(self, *, reset_active: bool = False) -> None:
        self.events.publish(
            SessionViewChanged(self._session_view(), reset_active)
        )

    @staticmethod
    def _parse_input(raw: str) -> UserInput:
        item_id = uuid4().hex
        if raw.strip().startswith("/"):
            return CommandInput(item_id, raw)
        return PromptInput(item_id, raw)


__all__ = [
    "ApplicationCoordinator",
    "CommandInput",
    "PromptInput",
    "RecoveryInput",
    "ShutdownInput",
]
