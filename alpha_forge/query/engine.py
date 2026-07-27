"""Stateless multi-round query engine driven by committed feedback."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from uuid import uuid4

from alpha_forge.providers.base import ModelProvider, StreamCompleted
from alpha_forge.query.protocol import (
    CommitModelOutput,
    CommitToolResult,
    ContextPrepared,
    ModelOutputCommitted,
    PrepareContext,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    QueryCompleted,
    QueryExecutionError,
    QueryFeedback,
    QueryRequest,
    QueryStreamEvent,
    ToolExecutionStarted,
    ToolResultCommitted,
)

MAX_TOOL_ROUNDS = 10
INTERRUPTED_TOOL_RESULT = (
    "Tool execution was interrupted before a durable result was recorded."
)


class QueryEngine:
    """Request effects without retaining a second copy of model context."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")
        self.provider = provider
        self.max_tool_rounds = max_tool_rounds

    async def run(
        self,
        request: QueryRequest,
    ) -> AsyncGenerator[QueryStreamEvent, QueryFeedback | None]:
        tool_rounds = request.completed_tool_rounds

        pending = request.pending_tool_continuation
        if pending is not None:
            recovery_revision = 0
            for call in pending.missing_calls:
                feedback = yield CommitToolResult(
                    pending.model_output_event_id,
                    call.call_id,
                    "interrupted",
                    INTERRUPTED_TOOL_RESULT,
                )
                committed_result = _expect(
                    feedback,
                    ToolResultCommitted,
                    "tool-result commit",
                )
                if (
                    not committed_result.result_event_id
                    or committed_result.revision <= recovery_revision
                ):
                    raise QueryExecutionError(
                        "internal",
                        "recovery commit feedback is not monotonic",
                    )
                recovery_revision = committed_result.revision
            tool_rounds += 1

        while True:
            if tool_rounds >= self.max_tool_rounds:
                raise QueryExecutionError(
                    "tool_round_limit",
                    f"tool round limit reached ({self.max_tool_rounds})",
                )

            feedback = yield PrepareContext(request.prompt_event_id)
            prepared = _expect(feedback, ContextPrepared, "context preparation")
            if prepared.snapshot.revision < 1:
                raise QueryExecutionError(
                    "internal",
                    "context feedback has an invalid transcript revision",
                )

            request_id = uuid4().hex
            yield ProviderRequestStarted(request.prompt_event_id, request_id)
            completed = None
            try:
                stream = self.provider.stream(
                    prepared.snapshot,
                    tools=request.tool_specs,
                )
                async with aclosing(stream) as provider_events:
                    async for event in provider_events:
                        if completed is not None:
                            raise RuntimeError(
                                "provider emitted data after stream completion"
                            )
                        if isinstance(event, StreamCompleted):
                            completed = event.output
                        else:
                            yield ProviderDeltaReceived(request_id, event)
            except QueryExecutionError:
                raise
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                raise QueryExecutionError("provider", message) from exc

            if completed is None:
                raise QueryExecutionError(
                    "provider",
                    "provider stream ended without a completed output",
                )
            if not completed.items:
                raise QueryExecutionError(
                    "provider",
                    "provider completed without any output items",
                )
            yield ProviderResponseCompleted(request_id, completed)

            feedback = yield CommitModelOutput(
                request.prompt_event_id,
                completed,
            )
            committed = _expect(
                feedback,
                ModelOutputCommitted,
                "model-output commit",
            )
            if (
                not committed.output_event_id
                or committed.revision != prepared.snapshot.revision + 1
            ):
                raise QueryExecutionError(
                    "internal",
                    "model-output commit feedback has an unexpected revision",
                )
            output_event_id = committed.output_event_id

            if not completed.tool_calls:
                yield QueryCompleted(request.prompt_event_id, output_event_id)
                return

            committed_revision = committed.revision
            for call in completed.tool_calls:
                yield ToolExecutionStarted(output_event_id, call)
                try:
                    result = await request.tool_executor.execute(call)
                except Exception as exc:
                    result_status = "error"
                    result_content = f"error: {str(exc) or type(exc).__name__}"
                else:
                    result_status = result.status
                    result_content = result.content
                feedback = yield CommitToolResult(
                    output_event_id,
                    call.call_id,
                    result_status,
                    result_content,
                )
                committed_result = _expect(
                    feedback,
                    ToolResultCommitted,
                    "tool-result commit",
                )
                if (
                    not committed_result.result_event_id
                    or committed_result.revision != committed_revision + 1
                ):
                    raise QueryExecutionError(
                        "internal",
                        "tool-result commit feedback has an unexpected revision",
                    )
                committed_revision = committed_result.revision
            tool_rounds += 1


def _expect[FeedbackType: QueryFeedback](
    feedback: QueryFeedback | None,
    expected: type[FeedbackType],
    action: str,
) -> FeedbackType:
    if not isinstance(feedback, expected):
        received = "none" if feedback is None else type(feedback).__name__
        raise QueryExecutionError(
            "internal",
            f"{action} requires {expected.__name__} feedback, got {received}",
        )
    return feedback


__all__ = [
    "INTERRUPTED_TOOL_RESULT",
    "MAX_TOOL_ROUNDS",
    "QueryEngine",
]
