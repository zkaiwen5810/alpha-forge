"""FIFO application coordinator for input, persistence, query, and UI flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from alpha_forge.application.events import (
    ExitReady,
    ExitRequested,
    InputQueued,
    InputStarted,
    PersistenceFailed,
    RequestFailed,
    SessionView,
    StatusChanged,
)
from alpha_forge.application.permissions import PermissionBroker
from alpha_forge.application.query_runner import QueryRunner
from alpha_forge.application.views import publish_session_view, session_view
from alpha_forge.config import Config
from alpha_forge.context.pipeline import ContextPipeline
from alpha_forge.context.tool_result_budget import ToolResultBudgetPolicy
from alpha_forge.events import EventRouter
from alpha_forge.hooks import (
    Hook,
    HookRegistry,
    PermissionAction,
    PreToolExecution,
    match_tool_names,
)
from alpha_forge.providers.base import ModelProvider
from alpha_forge.providers.openai_chat import OpenAIChatAdapter
from alpha_forge.query import (
    QueryEngine,
    QueryExecutionError,
)
from alpha_forge.sessions import Session
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext, CommandOutcome
from alpha_forge.tools import (
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
class ShutdownInput:
    item_id: str


type UserInput = PromptInput | CommandInput
type QueueItem = UserInput | ShutdownInput

DEFAULT_PERMISSION_TOOLS = ("bash", "file_writer")


class ApplicationCoordinator:
    """Own data flow; domain modules own their state and transformations."""

    def __init__(
        self,
        config: Config,
        *,
        provider: ModelProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        session: Session | None = None,
        context_pipeline: ContextPipeline | None = None,
        query_engine: QueryEngine | None = None,
        hook_registry: HookRegistry | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or OpenAIChatAdapter(config)
        self.command_handler = SlashCommandHandler()
        self.session = session or Session.create()
        self.event_router = EventRouter()
        self._permission_broker = PermissionBroker(self.event_router)
        self.hook_registry = hook_registry or HookRegistry()
        self.hook_registry.register(
            Hook(
                match_tool_names(*DEFAULT_PERMISSION_TOOLS),
                PermissionAction(self.request_tool_permission),
            )
        )
        self._query_runner = QueryRunner(
            query_engine=query_engine or QueryEngine(self.provider),
            context_pipeline=context_pipeline
            or ContextPipeline((ToolResultBudgetPolicy(),)),
            tool_registry=tool_registry or load_builtin_tools(),
            hook_registry=self.hook_registry,
            event_router=self.event_router,
        )
        self._input_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._accepting = True
        self._shutdown_enqueued = False
        self._persistence_halted = False

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def session_view(self) -> SessionView:
        return session_view(self.session)

    def submit(self, user_input: str) -> None:
        if not self._accepting or not user_input.strip():
            return
        item = self._parse_input(user_input)
        self._input_queue.put_nowait(item)
        raw = item.raw if isinstance(item, CommandInput) else item.content
        self.event_router.publish(InputQueued(item.item_id, raw))
        if isinstance(item, CommandInput) and (
            item.raw.strip().split(maxsplit=1)[0] in ("/exit", "/quit")
        ):
            self._accepting = False
            self.event_router.publish(ExitRequested())

    def request_exit(self) -> None:
        if self._shutdown_enqueued:
            return
        self._accepting = False
        self._permission_broker.deny_pending()
        self._shutdown_enqueued = True
        item = ShutdownInput(uuid4().hex)
        self._input_queue.put_nowait(item)
        self.event_router.publish(ExitRequested())

    async def request_tool_permission(self, event: PreToolExecution) -> bool:
        """Publish an ephemeral approval request and await its resolution."""
        return await self._permission_broker.request(event)

    def resolve_tool_permission(self, request_id: str, allowed: bool) -> bool:
        """Resolve the matching pending approval exactly once."""
        return self._permission_broker.resolve(request_id, allowed)

    async def consume(self) -> None:
        try:
            revision = self.session.revision
            try:
                self.session.interrupt_open_query()
            except Exception as exc:  # noqa: BLE001
                self._halt_for_persistence_failure("session activation", exc)
            finally:
                if self.session.revision != revision:
                    self._publish_view(reset_active=True)
            while True:
                item = await self._input_queue.get()
                try:
                    self.event_router.publish(InputStarted(item.item_id))
                    if isinstance(item, ShutdownInput):
                        self._shutdown_enqueued = True
                        self.event_router.publish(ExitReady())
                        return
                    if self._persistence_halted:
                        if isinstance(item, CommandInput) and item.raw.strip().split(
                            maxsplit=1
                        )[0] in ("/exit", "/quit"):
                            self.event_router.publish(ExitReady())
                            return
                        continue
                    if isinstance(item, PromptInput):
                        await self._handle_prompt(item.content)
                        continue
                    if await self._handle_command(item.raw):
                        self._shutdown_enqueued = True
                        self.event_router.publish(ExitReady())
                        return
                finally:
                    self._input_queue.task_done()
        finally:
            self.session.close()

    async def _handle_prompt(self, content: str) -> None:
        try:
            prompt = self.session.accept_prompt(content)
        except Exception as exc:  # noqa: BLE001
            self._halt_for_persistence_failure("user input", exc)
            return
        self._publish_view()
        await self._run_query(prompt.event_id)

    async def _run_query(self, prompt_event_id: str) -> None:
        try:
            request = self._query_runner.prepare_request(self.session, prompt_event_id)
            await self._query_runner.run(self.session, request)
        except TranscriptPersistenceError as exc:
            self._halt_for_persistence_failure("query", exc)
        except QueryExecutionError as exc:
            self._record_request_failure(
                prompt_event_id=prompt_event_id,
                stage=exc.stage,
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            self._record_request_failure(
                prompt_event_id=prompt_event_id,
                stage="internal",
                message=str(exc) or type(exc).__name__,
            )

    async def _handle_command(self, text: str) -> bool:
        source = self.session
        parsed = self.command_handler.parse(text)
        try:
            command_record = source.accept_command(
                text=parsed.raw,
                name=parsed.name,
                arguments=parsed.arguments,
            )
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            outcome = CommandOutcome(
                status="error",
                messages=(CommandMessage(f"command failed: {exc}", "error"),),
            )

        if outcome.action in ("clear", "resume"):
            outcome = self._switch_session(
                source,
                command_record.event_id,
                outcome,
                resume_path=(parsed.arguments if outcome.action == "resume" else None),
            )

        try:
            source.complete_command(
                command_record.event_id,
                status=outcome.status,
                messages=outcome.messages,
            )
        except Exception as exc:  # noqa: BLE001
            self._halt_for_persistence_failure("command result", exc)
            if self.session is not source:
                source.close()
        else:
            if self.session is source:
                self._publish_view()
            else:
                source.close()

        if outcome.status == "error" and outcome.messages:
            self.event_router.publish(StatusChanged(outcome.messages[-1].content))
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
            destination.interrupt_open_query()
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
        self._publish_view(reset_active=True)
        return outcome

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
            except Exception as exc:  # noqa: BLE001
                self._halt_for_persistence_failure("request failure", exc)
                return
        self.event_router.publish(RequestFailed(message))

    def _halt_for_persistence_failure(
        self,
        stage: str,
        error: Exception,
    ) -> None:
        self._accepting = False
        self._persistence_halted = True
        message = str(error) or type(error).__name__
        self.event_router.publish(PersistenceFailed(stage, message))

    def _publish_view(self, *, reset_active: bool = False) -> None:
        publish_session_view(self.event_router, self.session, reset_active=reset_active)

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
    "ShutdownInput",
]
