"""Pure, session-agnostic editing immediately before model requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from alpha_forge.model_messages import AssistantMessage, Message, ToolMessage
from alpha_forge.models import (
    EditedToolResult,
    PromptEdit,
    PromptEditDecision,
    PromptEditReason,
    RawToolResult,
    ToolCall,
)

MAX_TOOL_RESULT_CHARS = 16_000
MAX_TOOL_RESULTS_CHARS = 32_000
INTERRUPTED_TOOL_RESULT_CONTENT = (
    "error: tool execution was interrupted before its result was durably recorded"
)


class PromptEditBudgetError(RuntimeError):
    """Raised when required preview metadata cannot fit in its budget."""


def _decide_tool_results(
    results: tuple[RawToolResult, ...],
    *,
    individual_limit: int,
    aggregate_limit: int,
) -> tuple[PromptEditDecision, ...]:
    desired = [
        min(len(result.content), individual_limit) for result in results
    ]
    allocated = _water_fill(desired, aggregate_limit)
    decisions: list[PromptEditDecision] = []
    for result, desired_length, allocated_length in zip(
        results,
        desired,
        allocated,
        strict=True,
    ):
        reason: PromptEditReason | None = None
        if len(result.content) > allocated_length:
            reason = _preview_reason(
                original_length=len(result.content),
                desired_length=desired_length,
                allocated_length=allocated_length,
            )
            _render_tool_result(
                result,
                PromptEditDecision(
                    result.result_id,
                    result.call_id,
                    allocated_length,
                    reason,
                ),
            )
        decisions.append(
            PromptEditDecision(
                result.result_id,
                result.call_id,
                allocated_length,
                reason,
            )
        )
    return tuple(decisions)


def _render_tool_result(
    result: RawToolResult,
    decision: PromptEditDecision,
) -> str:
    if (
        result.result_id != decision.result_id
        or result.call_id != decision.call_id
    ):
        raise ValueError("prompt edit decision does not match tool result")
    if decision.allocated_chars < 0:
        raise ValueError("allocated characters cannot be negative")
    if decision.reason is None:
        if decision.allocated_chars != len(result.content):
            raise ValueError(
                "unedited result allocation must equal its content length"
            )
        return result.content
    if decision.allocated_chars >= len(result.content):
        raise ValueError(
            "preview allocation must be shorter than the raw result"
        )
    prefix = (
        "[alpha-forge tool-result-preview]\n"
        "truncated: true\n"
        f"reason: {decision.reason}\n"
        f"original_chars: {len(result.content)}\n"
        f"transcript_ref: {json.dumps(result.result_id)}\n"
        "--- preview head ---\n"
    )
    middle = "\n--- content omitted ---\n"
    suffix = "\n--- preview tail ---\n"
    overhead = len(prefix) + len(middle) + len(suffix)
    if overhead > decision.allocated_chars:
        raise PromptEditBudgetError(
            "tool-result budget is too small for required preview metadata "
            f"({decision.allocated_chars} available, {overhead} required)"
        )
    excerpt_chars = decision.allocated_chars - overhead
    head_chars = (excerpt_chars + 1) // 2
    tail_chars = excerpt_chars // 2
    tail = result.content[-tail_chars:] if tail_chars else ""
    return prefix + result.content[:head_chars] + middle + suffix + tail


def _water_fill(caps: list[int], total: int) -> list[int]:
    if sum(caps) <= total:
        return list(caps)
    allocations = [0] * len(caps)
    remaining = total
    pending = sorted(range(len(caps)), key=caps.__getitem__)
    while pending:
        share, remainder = divmod(remaining, len(pending))
        smallest = pending[0]
        if caps[smallest] <= share:
            allocations[smallest] = caps[smallest]
            remaining -= caps[smallest]
            pending.pop(0)
            continue
        for position, index in enumerate(pending):
            allocations[index] = share + (1 if position < remainder else 0)
        break
    return allocations


def _preview_reason(
    *,
    original_length: int,
    desired_length: int,
    allocated_length: int,
) -> PromptEditReason:
    individual = original_length > desired_length
    aggregate = allocated_length < desired_length
    if individual and aggregate:
        return "individual_and_aggregate_limits"
    if individual:
        return "individual_limit"
    return "aggregate_limit"


@dataclass(frozen=True, slots=True)
class PromptDraft:
    """Canonical application messages awaiting pre-request policies."""

    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class ToolBatchPromptEdit:
    """Effects produced while finalizing one raw tool-message tail."""

    output_id: str
    calls: tuple[ToolCall, ...]
    existing_results: tuple[RawToolResult, ...]
    synthesized_results: tuple[RawToolResult, ...]
    prompt_edit: PromptEdit

    @property
    def results(self) -> tuple[RawToolResult, ...]:
        by_call = {
            result.call_id: result
            for result in self.existing_results + self.synthesized_results
        }
        return tuple(by_call[call.id] for call in self.calls)


@dataclass(frozen=True, slots=True)
class EditedPrompt:
    """One outgoing model prompt and any replayable tool-result edit."""

    messages: tuple[Message, ...]
    tool_batch_edit: ToolBatchPromptEdit | None = None


class PromptEditor(Protocol):
    """Strategy applied once at the pre-request boundary of each iteration."""

    def edit(self, draft: PromptDraft) -> EditedPrompt:
        """Return the exact messages to send for one model request."""


@dataclass(frozen=True, slots=True)
class _RawToolBatch:
    assistant_index: int
    output_id: str
    calls: tuple[ToolCall, ...]
    results: tuple[RawToolResult, ...]


@dataclass(frozen=True, slots=True)
class _MissingResultPolicyOutcome:
    messages: tuple[Message, ...]
    batch: _RawToolBatch | None = None
    existing_results: tuple[RawToolResult, ...] = ()
    synthesized_results: tuple[RawToolResult, ...] = ()


def _open_tool_batch(messages: tuple[Message, ...]) -> _RawToolBatch | None:
    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], AssistantMessage)
        ),
        None,
    )
    if assistant_index is None:
        return None
    assistant = messages[assistant_index]
    assert isinstance(assistant, AssistantMessage)
    if not assistant.tool_calls:
        return None
    trailing = messages[assistant_index + 1 :]
    if any(not isinstance(message, ToolMessage) for message in trailing):
        return None
    tool_messages = tuple(
        message for message in trailing if isinstance(message, ToolMessage)
    )
    if tool_messages and not any(message.raw for message in tool_messages):
        if [message.tool_call_id for message in tool_messages] != [
            call.id for call in assistant.tool_calls
        ]:
            raise ValueError(
                "finalized tool messages must match tool-call order"
            )
        return None
    if any(not message.raw for message in tool_messages):
        raise ValueError("raw and finalized tool messages cannot be mixed")
    if assistant.output_id is None:
        raise ValueError("unfinished tool exchange requires an output ID")

    calls_by_id = {call.id: call for call in assistant.tool_calls}
    results_by_call: dict[str, RawToolResult] = {}
    for message in tool_messages:
        if message.tool_call_id not in calls_by_id:
            raise ValueError(
                f"raw tool message references unknown call: "
                f"{message.tool_call_id}"
            )
        if message.tool_call_id in results_by_call:
            raise ValueError(
                f"duplicate raw tool message: {message.tool_call_id}"
            )
        if message.result_id is None:
            raise ValueError("raw tool message requires a result ID")
        results_by_call[message.tool_call_id] = RawToolResult(
            message.result_id,
            message.tool_call_id,
            message.content,
            message.failed,
        )
    return _RawToolBatch(
        assistant_index,
        assistant.output_id,
        assistant.tool_calls,
        tuple(
            results_by_call[call.id]
            for call in assistant.tool_calls
            if call.id in results_by_call
        ),
    )


class MissingToolResultPolicy:
    """Complete an interrupted raw tool-message tail without rerunning tools."""

    def edit(self, draft: PromptDraft) -> _MissingResultPolicyOutcome:
        batch = _open_tool_batch(draft.messages)
        if batch is None:
            return _MissingResultPolicyOutcome(draft.messages)
        existing_by_call = {
            result.call_id: result for result in batch.results
        }
        synthesized = tuple(
            RawToolResult(
                uuid4().hex,
                call.id,
                INTERRUPTED_TOOL_RESULT_CONTENT,
                True,
            )
            for call in batch.calls
            if call.id not in existing_by_call
        )
        all_by_call = {
            result.call_id: result
            for result in batch.results + synthesized
        }
        ordered_results = tuple(all_by_call[call.id] for call in batch.calls)
        messages = draft.messages[: batch.assistant_index + 1] + tuple(
            ToolMessage(
                result.content,
                result.call_id,
                result.failed,
                result_id=result.result_id,
                raw=True,
            )
            for result in ordered_results
        )
        return _MissingResultPolicyOutcome(
            messages,
            _RawToolBatch(
                batch.assistant_index,
                batch.output_id,
                batch.calls,
                ordered_results,
            ),
            batch.results,
            synthesized,
        )


class ToolResultPromptEditor:
    """Add deterministic bounded tool results to an outgoing prompt."""

    policy_version = "head_tail_v1"

    def __init__(
        self,
        *,
        individual_limit: int = MAX_TOOL_RESULT_CHARS,
        aggregate_limit: int = MAX_TOOL_RESULTS_CHARS,
        missing_result_policy: MissingToolResultPolicy | None = None,
    ) -> None:
        if individual_limit <= 0:
            raise ValueError("individual tool-result limit must be positive")
        if aggregate_limit <= 0:
            raise ValueError("aggregate tool-result limit must be positive")
        self.individual_limit = individual_limit
        self.aggregate_limit = aggregate_limit
        self.missing_result_policy = (
            missing_result_policy or MissingToolResultPolicy()
        )

    def edit(self, draft: PromptDraft) -> EditedPrompt:
        completed = self.missing_result_policy.edit(draft)
        batch = completed.batch
        if batch is None:
            return EditedPrompt(completed.messages)
        prompt_edit = self.edit_results(batch.results)
        edited_results = self.render_edit(batch.results, prompt_edit)
        return EditedPrompt(
            messages=completed.messages[: batch.assistant_index + 1]
            + tuple(
                ToolMessage(
                    result.content,
                    tool_call_id=result.call_id,
                    failed=result.failed,
                    result_id=result.result_id,
                )
                for result in edited_results
            ),
            tool_batch_edit=ToolBatchPromptEdit(
                batch.output_id,
                batch.calls,
                completed.existing_results,
                completed.synthesized_results,
                prompt_edit,
            ),
        )

    def edit_results(self, results: tuple[RawToolResult, ...]) -> PromptEdit:
        decisions = self.decide(results)
        return PromptEdit(
            policy_version=self.policy_version,
            individual_limit=self.individual_limit,
            aggregate_limit=self.aggregate_limit,
            decisions=decisions,
        )

    def decide(
        self,
        results: tuple[RawToolResult, ...],
    ) -> tuple[PromptEditDecision, ...]:
        return _decide_tool_results(
            results,
            individual_limit=self.individual_limit,
            aggregate_limit=self.aggregate_limit,
        )

    def render_edit(
        self,
        results: tuple[RawToolResult, ...],
        edit: PromptEdit,
    ) -> tuple[EditedToolResult, ...]:
        if [
            (result.result_id, result.call_id) for result in results
        ] != [
            (decision.result_id, decision.call_id)
            for decision in edit.decisions
        ]:
            raise ValueError("prompt edit decisions must match tool-result order")
        return tuple(
            EditedToolResult(
                result_id=result.result_id,
                call_id=result.call_id,
                content=self.render(result, decision),
                failed=result.failed,
                previewed=decision.reason is not None,
            )
            for result, decision in zip(results, edit.decisions, strict=True)
        )

    @staticmethod
    def render(
        result: RawToolResult,
        decision: PromptEditDecision,
    ) -> str:
        return _render_tool_result(result, decision)


__all__ = [
    "EditedPrompt",
    "INTERRUPTED_TOOL_RESULT_CONTENT",
    "MAX_TOOL_RESULTS_CHARS",
    "MAX_TOOL_RESULT_CHARS",
    "MissingToolResultPolicy",
    "PromptDraft",
    "PromptEditBudgetError",
    "PromptEditor",
    "ToolBatchPromptEdit",
    "ToolResultPromptEditor",
]
