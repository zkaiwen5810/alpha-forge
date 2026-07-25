"""Protocol-aware session services over an append-only transcript."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from alpha_forge.model_history import ModelHistoryProjector
from alpha_forge.model_messages import Message
from alpha_forge.streaming import ModelResponse
from alpha_forge.tool_results import (
    MAX_TOOL_RESULT_CHARS,
    TranscriptToolResultLimiter,
)
from alpha_forge.transcript import (
    Command,
    CommandMessage,
    CommandResult,
    CommandStatus,
    ModelOutput,
    SessionTransition,
    ToolLimitDecision,
    ToolResult,
    ToolResultLimit,
    Transcript,
    TransitionKind,
    TurnFailure,
    UserMessage,
)

DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."


@dataclass(frozen=True, slots=True)
class PendingTurn:
    turn_id: str
    content: str


class _UseCurrentHead:
    pass


_USE_CURRENT_HEAD = _UseCurrentHead()


class Session:
    """Own transcript writes and enforce the model/tool conversation protocol."""

    def __init__(
        self,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        tool_result_limiter: TranscriptToolResultLimiter | None = None,
        session_id: str | None = None,
        transcript: Transcript | None = None,
        transcript_path: Path | None = None,
    ) -> None:
        if transcript is not None and (
            session_id is not None or transcript_path is not None
        ):
            raise ValueError(
                "transcript cannot be combined with session_id or transcript_path"
            )
        self.transcript = transcript or Transcript.create(
            system_prompt=system_prompt,
            session_id=session_id,
            path=transcript_path,
        )
        self._limiter = tool_result_limiter or TranscriptToolResultLimiter()
        self._head_turn_id = self._latest_turn_id()

    @classmethod
    def resume(
        cls,
        path: Path,
        *,
        tool_result_limiter: TranscriptToolResultLimiter | None = None,
    ) -> Session:
        return cls(
            transcript=Transcript.resume(path),
            tool_result_limiter=tool_result_limiter,
        )

    @property
    def session_id(self) -> str:
        return self.transcript.session_id

    @property
    def transcript_path(self) -> Path | None:
        return self.transcript.path

    @property
    def head_turn_id(self) -> str | None:
        return self._head_turn_id

    @property
    def messages(self) -> list[Message]:
        return ModelHistoryProjector(self.transcript).messages(
            head_turn_id=self._head_turn_id
        )

    def messages_for_turn(self, turn_id: str) -> list[Message]:
        return ModelHistoryProjector(self.transcript).messages(head_turn_id=turn_id)

    def submit_user(
        self,
        content: str,
        *,
        turn_id: str | None = None,
        parent_turn_id: str | None | _UseCurrentHead = _USE_CURRENT_HEAD,
    ) -> str:
        resolved = turn_id or uuid4().hex
        parent = (
            self._head_turn_id
            if isinstance(parent_turn_id, _UseCurrentHead)
            else parent_turn_id
        )
        self.transcript.append(UserMessage(resolved, parent, content))
        self._head_turn_id = resolved
        return resolved

    def select_head(self, turn_id: str | None) -> None:
        if turn_id is not None and self._user_message(turn_id) is None:
            raise KeyError(turn_id)
        self._head_turn_id = turn_id

    def add_assistant_message(
        self,
        *,
        turn_id: str,
        response: ModelResponse,
        output_id: str | None = None,
    ) -> ModelOutput:
        self._assert_model_output_allowed(turn_id)
        output = ModelOutput(
            output_id=output_id or uuid4().hex,
            turn_id=turn_id,
            content=response.content,
            tool_calls=response.tool_calls,
            reasoning_content=response.reasoning_content,
            refusal=response.refusal,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )
        self.transcript.append(output)
        return output

    def add_tool_result(
        self,
        *,
        output_id: str,
        call_id: str,
        content: str,
        failed: bool,
        result_id: str | None = None,
    ) -> ToolResult:
        output = self._model_output(output_id)
        if output is None:
            raise ValueError(f"unknown model output: {output_id}")
        if not output.tool_calls:
            raise RuntimeError("model output did not request tools")
        if self._tool_limit(output_id) is not None:
            raise RuntimeError("tool-result batch is already finalized")
        result = ToolResult(
            result_id=result_id or uuid4().hex,
            output_id=output_id,
            call_id=call_id,
            content=content,
            failed=failed,
        )
        self.transcript.append(result)
        return result

    def finalize_tool_results(
        self,
        *,
        output_id: str,
        results: tuple[ToolResult, ...],
    ) -> ToolResultLimit:
        output = self._model_output(output_id)
        if output is None:
            raise ValueError(f"unknown model output: {output_id}")
        expected = [call.id for call in output.tool_calls]
        actual = [result.call_id for result in results]
        if actual != expected:
            raise RuntimeError(
                "tool results must cover every call in model-output order"
            )
        if any(result.output_id != output_id for result in results):
            raise RuntimeError("tool results belong to another model output")
        event = self._limiter.apply(
            output_id=output_id,
            results=results,
        )
        self.transcript.append(event)
        return event

    def provisional_tool_decisions(
        self,
        results: tuple[ToolResult, ...],
    ) -> tuple[ToolLimitDecision, ...]:
        return self._limiter.decide(results)

    def fail_turn(self, turn_id: str, message: str) -> TurnFailure:
        if self._user_message(turn_id) is None:
            raise ValueError(f"unknown turn: {turn_id}")
        if self._turn_failed(turn_id):
            raise RuntimeError("turn failure is already recorded")
        event = TurnFailure(turn_id, message)
        self.transcript.append(event)
        return event

    def add_command(
        self,
        *,
        raw: str,
        name: str,
        arguments: str,
        command_id: str | None = None,
    ) -> Command:
        event = Command(
            command_id or uuid4().hex,
            self._head_turn_id,
            raw,
            name,
            arguments,
        )
        self.transcript.append(event)
        return event

    def add_command_result(
        self,
        command_id: str,
        *,
        status: CommandStatus,
        messages: tuple[CommandMessage, ...] = (),
    ) -> CommandResult:
        event = CommandResult(command_id, status, messages)
        self.transcript.append(event)
        return event

    def add_session_transition(
        self,
        *,
        kind: TransitionKind,
        source_session_id: str,
        source_command_id: str,
    ) -> SessionTransition:
        event = SessionTransition(
            kind,
            source_session_id,
            source_command_id,
        )
        self.transcript.append(event)
        return event

    def recover_unfinished_turns(self) -> list[PendingTurn]:
        """Repair interrupted tool batches and return branch work to requeue."""
        pending: list[PendingTurn] = []
        for turn in self._ancestry(self._head_turn_id):
            if self._turn_failed(turn.turn_id):
                continue
            outputs = self._outputs_for_turn(turn.turn_id)
            if not outputs:
                pending.append(PendingTurn(turn.turn_id, turn.content))
                continue
            latest = outputs[-1]
            if not latest.tool_calls:
                continue
            if self._tool_limit(latest.output_id) is None:
                results = self._results_for_output(latest.output_id)
                by_call = {result.call_id: result for result in results}
                ordered: list[ToolResult] = []
                for call in latest.tool_calls:
                    result = by_call.get(call.id)
                    if result is None:
                        result = self.add_tool_result(
                            output_id=latest.output_id,
                            call_id=call.id,
                            content=(
                                "error: tool execution was interrupted before "
                                "its result was durably recorded"
                            ),
                            failed=True,
                        )
                    ordered.append(result)
                self.finalize_tool_results(
                    output_id=latest.output_id,
                    results=tuple(ordered),
                )
            pending.append(PendingTurn(turn.turn_id, turn.content))
        return pending

    def read_tool_result(
        self,
        result_id: str,
        *,
        offset: int = 0,
        limit: int = MAX_TOOL_RESULT_CHARS,
    ) -> str:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit <= 0 or limit > MAX_TOOL_RESULT_CHARS:
            raise ValueError(f"limit must be between 1 and {MAX_TOOL_RESULT_CHARS}")
        try:
            result = self.transcript.result(result_id)
        except KeyError as exc:
            raise ValueError(f"unknown transcript result: {result_id}") from exc
        chunk = result.content[offset : offset + limit]
        next_offset = offset + len(chunk)
        eof = next_offset >= len(result.content)
        return (
            f"{chunk}\n"
            f"[alpha-forge transcript-result]\n"
            f"result_id: {result_id}\n"
            f"next_offset: {next_offset}\n"
            f"eof: {str(eof).lower()}"
        )

    def fresh(self) -> Session:
        return Session(
            system_prompt=self.transcript.system_prompt,
            tool_result_limiter=TranscriptToolResultLimiter(
                individual_limit=self._limiter.individual_limit,
                aggregate_limit=self._limiter.aggregate_limit,
            ),
        )

    def _assert_model_output_allowed(self, turn_id: str) -> None:
        if self._user_message(turn_id) is None:
            raise ValueError(f"unknown turn: {turn_id}")
        if self._turn_failed(turn_id):
            raise RuntimeError("cannot append output to a failed turn")
        outputs = self._outputs_for_turn(turn_id)
        if not outputs:
            return
        latest = outputs[-1]
        if not latest.tool_calls:
            raise RuntimeError("turn already has a terminal model output")
        if self._tool_limit(latest.output_id) is None:
            raise RuntimeError(
                "cannot continue before all requested tools are finalized"
            )

    def _latest_turn_id(self) -> str | None:
        latest = None
        for event in self.transcript.events:
            if isinstance(event, UserMessage):
                latest = event.turn_id
        return latest

    def _user_message(self, turn_id: str) -> UserMessage | None:
        return next(
            (
                event
                for event in self.transcript.events
                if isinstance(event, UserMessage) and event.turn_id == turn_id
            ),
            None,
        )

    def _model_output(self, output_id: str) -> ModelOutput | None:
        return next(
            (
                event
                for event in self.transcript.events
                if isinstance(event, ModelOutput) and event.output_id == output_id
            ),
            None,
        )

    def _outputs_for_turn(self, turn_id: str) -> list[ModelOutput]:
        return [
            event
            for event in self.transcript.events
            if isinstance(event, ModelOutput) and event.turn_id == turn_id
        ]

    def _results_for_output(self, output_id: str) -> list[ToolResult]:
        return [
            event
            for event in self.transcript.events
            if isinstance(event, ToolResult) and event.output_id == output_id
        ]

    def _tool_limit(self, output_id: str) -> ToolResultLimit | None:
        return next(
            (
                event
                for event in self.transcript.events
                if isinstance(event, ToolResultLimit) and event.output_id == output_id
            ),
            None,
        )

    def _turn_failed(self, turn_id: str) -> bool:
        return any(
            isinstance(event, TurnFailure) and event.turn_id == turn_id
            for event in self.transcript.events
        )

    def _ancestry(self, head_turn_id: str | None) -> list[UserMessage]:
        if head_turn_id is None:
            return []
        turns = {
            event.turn_id: event
            for event in self.transcript.events
            if isinstance(event, UserMessage)
        }
        result: list[UserMessage] = []
        cursor: str | None = head_turn_id
        while cursor is not None:
            turn = turns[cursor]
            result.append(turn)
            cursor = turn.parent_turn_id
        result.reverse()
        return result


__all__ = ["DEFAULT_SYSTEM_PROMPT", "PendingTurn", "Session"]
