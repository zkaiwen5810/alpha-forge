"""Pure, session-agnostic editing of tool results for model prompts."""

from __future__ import annotations

import json

from alpha_forge.models import (
    EditedToolResult,
    PromptEdit,
    PromptEditDecision,
    PromptEditReason,
    RawToolResult,
)

MAX_TOOL_RESULT_CHARS = 16_000
MAX_TOOL_RESULTS_CHARS = 32_000


class PromptEditBudgetError(RuntimeError):
    """Raised when required preview metadata cannot fit in its budget."""


class ToolResultPromptEditor:
    """Create deterministic bounded representations of raw tool results."""

    policy_version = "head_tail_v1"

    def __init__(
        self,
        *,
        individual_limit: int = MAX_TOOL_RESULT_CHARS,
        aggregate_limit: int = MAX_TOOL_RESULTS_CHARS,
    ) -> None:
        if individual_limit <= 0:
            raise ValueError("individual tool-result limit must be positive")
        if aggregate_limit <= 0:
            raise ValueError("aggregate tool-result limit must be positive")
        self.individual_limit = individual_limit
        self.aggregate_limit = aggregate_limit

    def edit(self, results: tuple[RawToolResult, ...]) -> PromptEdit:
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
        desired = [
            min(len(result.content), self.individual_limit) for result in results
        ]
        allocated = self._water_fill(desired, self.aggregate_limit)
        decisions: list[PromptEditDecision] = []
        for result, desired_length, allocated_length in zip(
            results,
            desired,
            allocated,
            strict=True,
        ):
            reason: PromptEditReason | None = None
            if len(result.content) > allocated_length:
                reason = self._preview_reason(
                    original_length=len(result.content),
                    desired_length=desired_length,
                    allocated_length=allocated_length,
                )
                self.render(
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

    @staticmethod
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

    @staticmethod
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


__all__ = [
    "MAX_TOOL_RESULTS_CHARS",
    "MAX_TOOL_RESULT_CHARS",
    "PromptEditBudgetError",
    "ToolResultPromptEditor",
]
