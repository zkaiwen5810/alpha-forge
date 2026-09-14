"""Session-scoped transcript commands and projections."""

from __future__ import annotations

from pathlib import Path

from alpha_forge.context.models import ModelContextSnapshot
from alpha_forge.context.pipeline import ContextPipeline
from alpha_forge.projectors.model_context import ModelContextProjector
from alpha_forge.projectors.ui_history import (
    UiHistoryItem,
    UiHistoryProjector,
)
from alpha_forge.providers.base import ProviderOutput
from alpha_forge.transcript.events import (
    CommandCompleted,
    CommandMessage,
    CommandStatus,
    InputAccepted,
    ModelOutput,
    QueryFailed,
    QueryFailureStage,
    SessionLinked,
    SessionLinkKind,
    ToolResult,
    ToolResultStatus,
    TranscriptEvent,
)
from alpha_forge.transcript.records import TranscriptRecord
from alpha_forge.transcript.store import TranscriptStore
from alpha_forge.transcript.validation import tool_calls

DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."
INTERRUPTED_TOOL_RESULT = (
    "No durable tool result was recorded. Execution outcome is unknown; "
    "the action may have happened before the session was interrupted."
)


class Session:
    """The only application service allowed to append transcript events."""

    def __init__(self, transcript: TranscriptStore) -> None:
        self._transcript = transcript

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        session_id: str | None = None,
        transcript_path: Path | None = None,
        in_memory: bool = False,
    ) -> Session:
        if in_memory:
            store = TranscriptStore.in_memory(
                instructions=system_prompt,
                session_id=session_id,
            )
        else:
            store = TranscriptStore.create(
                instructions=system_prompt,
                session_id=session_id,
                path=transcript_path,
            )
        return cls(store)

    @classmethod
    def resume(cls, path: Path) -> Session:
        return cls(TranscriptStore.resume(path))

    @property
    def transcript(self) -> TranscriptStore:
        """Underlying store for inspection and readers; append through Session methods."""

        return self._transcript

    @property
    def session_id(self) -> str:
        return self._transcript.session_id

    @property
    def transcript_path(self) -> Path | None:
        return self._transcript.path

    @property
    def revision(self) -> int:
        return self._transcript.revision

    @property
    def instructions(self) -> str | None:
        return self._transcript.instructions

    def accept_prompt(self, text: str) -> TranscriptRecord:
        return self._commit(InputAccepted("prompt", text))

    def accept_command(
        self,
        *,
        text: str,
        name: str,
        arguments: str,
    ) -> TranscriptRecord:
        return self._commit(
            InputAccepted("command", text, name, arguments)
        )

    def complete_command(
        self,
        command_event_id: str,
        *,
        status: CommandStatus,
        messages: tuple[CommandMessage, ...],
    ) -> TranscriptRecord:
        if status not in ("success", "error"):
            raise ValueError(f"invalid command status: {status}")
        return self._commit(CommandCompleted(command_event_id, status, messages))

    def link(
        self,
        *,
        kind: SessionLinkKind,
        source_session_id: str,
        source_command_event_id: str,
    ) -> TranscriptRecord:
        if kind not in ("clear", "resume"):
            raise ValueError(f"invalid session link kind: {kind}")
        return self._commit(
            SessionLinked(kind, source_session_id, source_command_event_id)
        )

    def record_model_output(
        self,
        prompt_event_id: str,
        output: ProviderOutput,
    ) -> TranscriptRecord:
        return self._commit(
            ModelOutput(
                prompt_event_id,
                output.items,
                output.finish_reason,
                output.usage,
            )
        )

    def record_tool_result(
        self,
        *,
        model_output_event_id: str,
        call_id: str,
        status: ToolResultStatus,
        content: str,
    ) -> TranscriptRecord:
        return self._commit(
            ToolResult(
                model_output_event_id,
                call_id,
                status,
                content,
            )
        )

    def fail_query(
        self,
        prompt_event_id: str,
        *,
        stage: QueryFailureStage,
        message: str,
    ) -> TranscriptRecord:
        return self._commit(QueryFailed(prompt_event_id, stage, message))

    def prepare_context(
        self,
        pipeline: ContextPipeline,
    ) -> ModelContextSnapshot:
        projector = ModelContextProjector(self._transcript)
        return pipeline.prepare(
            project=lambda: projector.project(require_complete=True),
            commit=self._commit,
        )

    def interrupt_open_query(self) -> None:
        """Close abandoned work on activation without executing tools or a model.

        Each append is durable, so a later activation can finish this operation
        after a crash without duplicating results or the terminal event.
        """
        state = self._transcript.state
        prompt_id = state.active_prompt_event_id
        if prompt_id is None:
            return
        output_ids = state.outputs_by_prompt.get(prompt_id, [])
        if output_ids:
            output_id = output_ids[-1]
            recorded = state.results_by_output[output_id]
            for call in tool_calls(state.outputs[output_id]):
                if call.call_id not in recorded:
                    self.record_tool_result(
                        model_output_event_id=output_id,
                        call_id=call.call_id,
                        status="interrupted",
                        content=INTERRUPTED_TOOL_RESULT,
                    )
        self.fail_query(
            prompt_id,
            stage="interrupted",
            message="Previous request was interrupted. Send a message to continue.",
        )

    def ui_history(self) -> tuple[UiHistoryItem, ...]:
        return tuple(UiHistoryProjector(self._transcript).items())

    def fresh(self) -> Session:
        return Session.create(
            system_prompt=self.instructions,
            in_memory=self.transcript_path is None,
        )

    def close(self) -> None:
        self._transcript.close()

    def _commit(self, event: TranscriptEvent) -> TranscriptRecord:
        return self._transcript.append(
            event,
            expected_revision=self._transcript.revision,
        )


__all__ = ["DEFAULT_SYSTEM_PROMPT", "Session"]
