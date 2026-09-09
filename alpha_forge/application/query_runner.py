"""Execute a query against one session and acknowledge durable effects."""

from contextlib import aclosing

from alpha_forge.application.events import (
    ModelOutputRecorded,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    StatusChanged,
    ToolResultRecorded,
    ToolStarted,
)
from alpha_forge.application.views import publish_session_view
from alpha_forge.context.pipeline import ContextPipeline
from alpha_forge.events import EventRouter
from alpha_forge.hooks import HookRegistry
from alpha_forge.projectors.session_state import OpenQuery
from alpha_forge.query import (
    CommitModelOutput,
    CommitToolResult,
    ContextPrepared,
    ModelOutputCommitted,
    PendingIntermediateRound,
    PrepareContext,
    QueryCompleted,
    QueryEffect,
    QueryEngine,
    QueryExecutionError,
    QueryRequest,
    ToolExecutionStarted,
    ToolResultCommitted,
)
from alpha_forge.query.protocol import QueryFeedback
from alpha_forge.sessions import Session
from alpha_forge.sessions.tool_result_reader import ToolResultReader
from alpha_forge.tools import ToolExecutor, ToolNotFoundError, ToolRegistry
from alpha_forge.transcript import TranscriptPersistenceError


class QueryRunner:
    def __init__(
        self,
        query_engine: QueryEngine,
        context_pipeline: ContextPipeline,
        tool_registry: ToolRegistry,
        hook_registry: HookRegistry,
        event_router: EventRouter,
    ) -> None:
        self.query_engine = query_engine
        self.context_pipeline = context_pipeline
        self.tool_registry = tool_registry
        self.hook_registry = hook_registry
        self.event_router = event_router

    def prepare_request(
        self, session: Session, continuation: OpenQuery
    ) -> QueryRequest:
        registry = self._query_registry(session)
        return QueryRequest(
            prompt_event_id=continuation.prompt_event_id,
            pending_intermediate_round=(
                PendingIntermediateRound(
                    continuation.pending_intermediate_round.model_output_event_id,
                    continuation.pending_intermediate_round.missing_calls,
                )
                if continuation.pending_intermediate_round is not None
                else None
            ),
            completed_intermediate_rounds=(continuation.completed_intermediate_rounds),
            tool_specs=registry.specs(),
            tool_executor=ToolExecutor(registry, self.hook_registry),
        )

    async def run(self, session: Session, request: QueryRequest) -> None:
        feedback: QueryFeedback | None = None
        async with aclosing(self.query_engine.run(request)) as query_events:
            while True:
                try:
                    event = await query_events.asend(feedback)
                except StopAsyncIteration:
                    break
                feedback = None
                if isinstance(event, QueryEffect):
                    feedback = self._handle_query_effect(session, event)
                else:
                    self._publish_query_progress(event)

    def _handle_query_effect(
        self, session: Session, effect: QueryEffect
    ) -> QueryFeedback:
        if isinstance(effect, PrepareContext):
            previous_revision = session.revision
            try:
                snapshot = session.prepare_context(self.context_pipeline)
            except TranscriptPersistenceError:
                raise
            except Exception as exc:
                raise QueryExecutionError(
                    "context",
                    str(exc) or type(exc).__name__,
                ) from exc
            if session.revision != previous_revision:
                publish_session_view(self.event_router, session)
            return ContextPrepared(snapshot)
        if isinstance(effect, CommitModelOutput):
            record = session.record_model_output(
                effect.prompt_event_id,
                effect.output,
            )
            publish_session_view(self.event_router, session, reset_active=True)
            self.event_router.publish(ModelOutputRecorded(record.event_id))
            return ModelOutputCommitted(record.event_id, session.revision)
        if isinstance(effect, CommitToolResult):
            record = session.record_tool_result(
                model_output_event_id=effect.model_output_event_id,
                call_id=effect.call_id,
                status=effect.status,
                content=effect.content,
            )
            publish_session_view(self.event_router, session, reset_active=True)
            self.event_router.publish(
                ToolResultRecorded(
                    record.event_id,
                    effect.model_output_event_id,
                    effect.call_id,
                )
            )
            return ToolResultCommitted(record.event_id, session.revision)
        raise QueryExecutionError(
            "internal",
            f"unsupported query effect: {type(effect).__name__}",
        )

    def _publish_query_progress(self, event: object) -> None:
        if isinstance(event, ProviderRequestStarted):
            self.event_router.publish(event)
        elif isinstance(event, ProviderDeltaReceived):
            self.event_router.publish(event)
        elif isinstance(event, ProviderResponseCompleted):
            self.event_router.publish(event)
        elif isinstance(event, ToolExecutionStarted):
            self.event_router.publish(
                ToolStarted(event.model_output_event_id, event.call)
            )
        elif isinstance(event, QueryCompleted):
            self.event_router.publish(StatusChanged("Ready"))

    def _query_registry(self, session: Session) -> ToolRegistry:
        registry = self.tool_registry.copy()
        try:
            registry.get("tool_result_reader")
        except ToolNotFoundError:
            registry.register(ToolResultReader(session.transcript).as_tool())
        return registry
