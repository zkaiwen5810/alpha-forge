"""Pure, versioned projection policy for transcript-held tool results."""

from __future__ import annotations

import json

from alpha_forge.transcript import (
    PreviewReason,
    ToolLimitDecision,
    ToolResult,
    ToolResultLimit,
)

MAX_TOOL_RESULT_CHARS = 16_000
MAX_TOOL_RESULTS_CHARS = 32_000


class ToolResultBudgetError(RuntimeError):
    """Raised when required preview metadata cannot fit in its budget."""


class TranscriptToolResultLimiter:
    """Record and replay deterministic model/UI result projections."""

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

    def apply(
        self,
        *,
        output_id: str,
        results: tuple[ToolResult, ...],
    ) -> ToolResultLimit:
        decisions = self.decide(results)
        return ToolResultLimit(
            output_id=output_id,
            policy_version="head_tail_v1",
            individual_limit=self.individual_limit,
            aggregate_limit=self.aggregate_limit,
            decisions=decisions,
        )

    def decide(
        self,
        results: tuple[ToolResult, ...],
    ) -> tuple[ToolLimitDecision, ...]:
        """Return deterministic preview decisions without creating an event."""
        desired = [
            min(len(result.content), self.individual_limit) for result in results
        ]
        allocated = self._water_fill(desired, self.aggregate_limit)
        decisions: list[ToolLimitDecision] = []
        for result, desired_length, allocated_length in zip(
            results,
            desired,
            allocated,
            strict=True,
        ):
            reason: PreviewReason | None = None
            if len(result.content) > allocated_length:
                reason = self._preview_reason(
                    original_length=len(result.content),
                    desired_length=desired_length,
                    allocated_length=allocated_length,
                )
                self.render(
                    result,
                    ToolLimitDecision(
                        result.result_id,
                        result.call_id,
                        allocated_length,
                        reason,
                    ),
                )
            decisions.append(
                ToolLimitDecision(
                    result.result_id,
                    result.call_id,
                    allocated_length,
                    reason,
                )
            )
        return tuple(decisions)

    @staticmethod
    def render(
        result: ToolResult,
        decision: ToolLimitDecision,
    ) -> str:
        if decision.reason is None:
            return result.content
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
            raise ToolResultBudgetError(
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
    ) -> PreviewReason:
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
    "ToolResultBudgetError",
    "TranscriptToolResultLimiter",
]
