"""Execute a query against one session and acknowledge durable effects."""

from contextlib import aclosing

from alpha_forge.application.events import (
    ModelOutputRecorded,
    ResponseStreamCompleted,
    ResponseStreamStarted,
    ResponseStreamUpdated,
    StatusChanged,
    ToolCallProcessingStarted,
    ToolResultRecorded,
)
from alpha_forge.application.router import ApplicationEventRouter
from alpha_forge.application.views import publish_session_view
from alpha_forge.context.pipeline import ContextPipeline
from alpha_forge.hooks import HookRegistry
from alpha_forge.query import (
    CommitModelOutput,
    CommitToolResult,
    ContextPrepared,
    ModelOutputCommitted,
    PrepareContext,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    QueryCompleted,
    QueryEffect,
    QueryEngine,
    QueryExecutionError,
    QueryRequest,
    ToolResultCommitted,
)
from alpha_forge.query import (
    ToolCallProcessingStarted as QueryToolCallProcessingStarted,
)
from alpha_forge.query.protocol import QueryFeedback, QueryProgress
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
        event_router: ApplicationEventRouter,
    ) -> None:
        self.query_engine = query_engine
        self.context_pipeline = context_pipeline
        self.tool_registry = tool_registry
        self.hook_registry = hook_registry
        self.event_router = event_router

    def prepare_request(
        self, session: Session, prompt_event_id: str
    ) -> QueryRequest:
        registry = self._query_registry(session)
        return QueryRequest(
            prompt_event_id=prompt_event_id,
            tool_specs=registry.specs(),
            tool_executor=ToolExecutor(registry, self.hook_registry),
        )

    async def run(self, session: Session, request: QueryRequest) -> None:
        feedback: QueryFeedback | None = None
        async with aclosing(self.query_engine.run(request)) as query_messages:
            while True:
                try:
                    message = await query_messages.asend(feedback)
                except StopAsyncIteration:
                    break
                feedback = None
                if isinstance(message, QueryEffect):
                    feedback = self._handle_query_effect(session, message)
                elif isinstance(message, QueryProgress):
                    self._publish_query_progress(message)
                else:
                    raise QueryExecutionError(
                        "internal",
                        f"unsupported query message: {type(message).__name__}",
                    )

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

    def _publish_query_progress(self, event: QueryProgress) -> None:
        if isinstance(event, ProviderRequestStarted):
            self.event_router.publish(
                ResponseStreamStarted(event.prompt_event_id, event.request_id)
            )
        elif isinstance(event, ProviderDeltaReceived):
            self.event_router.publish(ResponseStreamUpdated(event.request_id, event.delta))
        elif isinstance(event, ProviderResponseCompleted):
            self.event_router.publish(
                ResponseStreamCompleted(event.request_id, event.output)
            )
        elif isinstance(event, QueryToolCallProcessingStarted):
            self.event_router.publish(
                ToolCallProcessingStarted(event.model_output_event_id, event.call)
            )
        elif isinstance(event, QueryCompleted):
            self.event_router.publish(StatusChanged("Ready"))
        else:
            raise QueryExecutionError(
                "internal",
                f"unsupported query progress: {type(event).__name__}",
            )

    def _query_registry(self, session: Session) -> ToolRegistry:
        registry = self.tool_registry.copy()
        try:
            registry.get("tool_result_reader")
        except ToolNotFoundError:
            registry.register(ToolResultReader(session.transcript).as_tool())
        return registry
