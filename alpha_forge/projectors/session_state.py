"""Recovery and continuation projection for a linear transcript."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_forge.providers.base import ToolCall
from alpha_forge.transcript.store import TranscriptStore
from alpha_forge.transcript.validation import tool_calls


@dataclass(frozen=True, slots=True)
class PendingIntermediateRound:
    model_output_event_id: str
    missing_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class OpenQuery:
    prompt_event_id: str
    pending_intermediate_round: PendingIntermediateRound | None
    completed_intermediate_rounds: int


class SessionStateProjector:
    def __init__(self, transcript: TranscriptStore) -> None:
        self.transcript = transcript

    def open_query(self) -> OpenQuery | None:
        state = self.transcript.state
        prompt_id = state.active_prompt_event_id
        if prompt_id is None:
            return None
        output_ids = state.outputs_by_prompt.get(prompt_id, [])
        if not output_ids:
            return OpenQuery(prompt_id, None, 0)
        output_id = output_ids[-1]
        output = state.outputs[output_id]
        calls = tool_calls(output)
        if not calls:
            return None
        recorded = state.results_by_output[output_id]
        missing = tuple(
            call for call in calls if call.call_id not in recorded
        )
        return OpenQuery(
            prompt_id,
            PendingIntermediateRound(
                output_id,
                missing,
            ),
            sum(
                1
                for candidate_id in output_ids
                if tool_calls(state.outputs[candidate_id])
                and state.output_complete(candidate_id)
                and candidate_id != output_id
            ),
        )


__all__ = ["OpenQuery", "PendingIntermediateRound", "SessionStateProjector"]
