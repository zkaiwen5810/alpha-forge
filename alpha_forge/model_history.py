"""Model-facing projection of durable transcript activities."""

from __future__ import annotations

from alpha_forge.model_messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
)
from alpha_forge.model_messages import (
    UserMessage as ModelUserMessage,
)
from alpha_forge.models import RawToolResult
from alpha_forge.prompt_editor import ToolResultPromptEditor
from alpha_forge.transcript import (
    ModelOutput,
    ToolResult,
    ToolResultEdit,
    Transcript,
    UserMessage,
)


class ModelHistoryProjector:
    """Build valid OpenAI messages for one selected transcript branch."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript

    def messages(self, *, head_turn_id: str | None = None) -> list[Message]:
        return list(
            self._project(
                head_turn_id=head_turn_id,
                include_unfinished=False,
            )
        )

    def query_messages(
        self,
        *,
        head_turn_id: str | None = None,
    ) -> list[Message]:
        return list(
            self._project(
                head_turn_id=head_turn_id,
                include_unfinished=True,
            )
        )

    def _project(
        self,
        *,
        head_turn_id: str | None,
        include_unfinished: bool,
    ) -> tuple[Message, ...]:
        messages: list[Message] = []
        if self.transcript.system_prompt:
            messages.append(SystemMessage(self.transcript.system_prompt))

        turns = self._ancestry(head_turn_id)
        outputs_by_turn: dict[str, list[ModelOutput]] = {}
        results: dict[tuple[str, str], ToolResult] = {}
        edits: dict[str, ToolResultEdit] = {}
        for event in self.transcript.events:
            if isinstance(event, ModelOutput):
                outputs_by_turn.setdefault(event.turn_id, []).append(event)
            elif isinstance(event, ToolResult):
                results[(event.output_id, event.call_id)] = event
            elif isinstance(event, ToolResultEdit):
                edits[event.output_id] = event

        for turn in turns:
            messages.append(ModelUserMessage(turn.content))
            for output in outputs_by_turn.get(turn.turn_id, ()):
                if output.tool_calls:
                    edit = edits.get(output.output_id)
                    if edit is None:
                        if not include_unfinished:
                            return tuple(messages)
                        messages.append(self._assistant_message(output))
                        resolved = [
                            results.get((output.output_id, call.id))
                            for call in output.tool_calls
                        ]
                        messages.extend(
                            ToolMessage(
                                result.content,
                                result.call_id,
                                result.failed,
                                result_id=result.result_id,
                                raw=True,
                            )
                            for result in resolved
                            if result is not None
                        )
                        return tuple(messages)
                    resolved = [
                        results.get((output.output_id, decision.call_id))
                        for decision in edit.decisions
                    ]
                    if any(result is None for result in resolved):
                        break
                    messages.append(self._assistant_message(output))
                    for decision, result in zip(
                        edit.decisions,
                        resolved,
                        strict=True,
                    ):
                        assert result is not None
                        messages.append(
                            ToolMessage(
                                ToolResultPromptEditor.render(
                                    RawToolResult(
                                        result.result_id,
                                        result.call_id,
                                        result.content,
                                        result.failed,
                                    ),
                                    decision,
                                ),
                                result.call_id,
                                result.failed,
                                result_id=result.result_id,
                            )
                        )
                    continue

                messages.append(self._assistant_message(output))
                break
        return tuple(messages)

    @staticmethod
    def _assistant_message(output: ModelOutput) -> AssistantMessage:
        return AssistantMessage(
            content=output.content,
            tool_calls=output.tool_calls,
            reasoning_content=output.reasoning_content,
            refusal=output.refusal,
            output_id=output.output_id,
        )

    def _ancestry(self, head_turn_id: str | None) -> list[UserMessage]:
        turns = {
            event.turn_id: event
            for event in self.transcript.events
            if isinstance(event, UserMessage)
        }
        if not turns:
            return []
        head = head_turn_id or next(reversed(turns))
        if head not in turns:
            raise KeyError(head)
        ancestry: list[UserMessage] = []
        cursor: str | None = head
        while cursor is not None:
            turn = turns[cursor]
            ancestry.append(turn)
            cursor = turn.parent_turn_id
        ancestry.reverse()
        return ancestry


__all__ = ["ModelHistoryProjector"]
