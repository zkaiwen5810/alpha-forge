"""Model-facing projection of committed transcript records."""

from __future__ import annotations

from alpha_forge.context.models import (
    ModelContextSnapshot,
    ModelOutputContext,
    SystemMessage,
    ToolResultContext,
    UserMessage,
)
from alpha_forge.transcript.events import InputAccepted, ModelOutput, ToolResult
from alpha_forge.transcript.records import TranscriptCorruptError
from alpha_forge.transcript.representations import render_tool_result
from alpha_forge.transcript.store import TranscriptStore


class ModelContextProjector:
    def __init__(self, transcript: TranscriptStore) -> None:
        self.transcript = transcript

    def project(self, *, require_complete: bool = True) -> ModelContextSnapshot:
        items = []
        if self.transcript.instructions:
            items.append(SystemMessage(self.transcript.instructions))

        state = self.transcript.state
        for record in self.transcript.records:
            event = record.event
            if isinstance(event, InputAccepted) and event.kind == "prompt":
                items.append(UserMessage(record.event_id, event.text))
            elif isinstance(event, ModelOutput):
                if not state.exchange_visibility[record.event_id]:
                    continue
                items.append(
                    ModelOutputContext(
                        record.event_id,
                        event.prompt_event_id,
                        event.items,
                    )
                )
            elif isinstance(event, ToolResult):
                if not state.exchange_visibility[event.model_output_event_id]:
                    continue
                representation = state.result_representations[record.event_id]
                items.append(
                    ToolResultContext(
                        result_event_id=record.event_id,
                        model_output_event_id=event.model_output_event_id,
                        call_id=event.call_id,
                        status=event.status,
                        content=render_tool_result(
                            record.event_id,
                            event.content,
                            representation,
                        ),
                        original_chars=len(event.content),
                        representation=representation,
                    )
                )

        snapshot = ModelContextSnapshot(self.transcript.revision, tuple(items))
        if require_complete:
            self._validate_provider_protocol(snapshot)
        return snapshot

    @staticmethod
    def _validate_provider_protocol(snapshot: ModelContextSnapshot) -> None:
        outputs = {
            item.output_event_id: item
            for item in snapshot.items
            if isinstance(item, ModelOutputContext)
        }
        results: dict[str, dict[str, ToolResultContext]] = {}
        for item in snapshot.items:
            if isinstance(item, ToolResultContext):
                output = outputs.get(item.model_output_event_id)
                if output is None:
                    raise TranscriptCorruptError(
                        "model context contains an orphaned tool result"
                    )
                results.setdefault(item.model_output_event_id, {})[
                    item.call_id
                ] = item
        for output_id, output in outputs.items():
            calls = output.tool_calls
            if not calls:
                continue
            by_call = results.get(output_id, {})
            if [call.call_id for call in calls] != list(by_call):
                raise TranscriptCorruptError(
                    "model context contains an incomplete or reordered tool exchange"
                )


__all__ = ["ModelContextProjector"]
