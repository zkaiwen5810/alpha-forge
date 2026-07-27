"""Versioned tool-result representation policy."""

from __future__ import annotations

from alpha_forge.context.models import (
    ModelContextSnapshot,
    ModelOutputContext,
    ToolResultContext,
)
from alpha_forge.context.policies import ContextPolicyDecision
from alpha_forge.transcript.events import (
    HeadTailPreview,
    OriginalRepresentation,
    PolicyInvocation,
    SetToolResultRepresentation,
    ToolPreviewReason,
    ToolResultRepresentation,
)
from alpha_forge.transcript.representations import (
    preview_frame,
)

MAX_TOOL_RESULT_CHARS = 16_000
MAX_TOOL_RESULTS_CHARS = 32_000


class ToolResultBudgetError(RuntimeError):
    """Raised when required preview metadata cannot fit in a configured budget."""


class ToolResultBudgetPolicy:
    """Bound the latest complete tool exchange before a provider request."""

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

    def evaluate(
        self,
        snapshot: ModelContextSnapshot,
    ) -> ContextPolicyDecision:
        output, results = _latest_tool_exchange(snapshot)
        policy = PolicyInvocation(
            "tool_result_budget",
            1,
            {
                "individual_limit": self.individual_limit,
                "aggregate_limit": self.aggregate_limit,
            },
        )
        if output is None:
            return ContextPolicyDecision(policy)
        desired = [
            min(result.original_chars, self.individual_limit)
            for result in results
        ]
        allocations = _water_fill(desired, self.aggregate_limit)
        operations = []
        for result, desired_length, allocation in zip(
            results,
            desired,
            allocations,
            strict=True,
        ):
            target = _representation(
                result,
                desired_length=desired_length,
                allocated_chars=allocation,
            )
            if target != result.representation:
                operations.append(
                    SetToolResultRepresentation(
                        result.result_event_id,
                        target,
                    )
                )
        return ContextPolicyDecision(policy, tuple(operations))


def _latest_tool_exchange(
    snapshot: ModelContextSnapshot,
) -> tuple[ModelOutputContext | None, tuple[ToolResultContext, ...]]:
    latest: ModelOutputContext | None = None
    for item in snapshot.items:
        if isinstance(item, ModelOutputContext) and item.tool_calls:
            latest = item
    if latest is None:
        return None, ()
    results_by_call = {
        item.call_id: item
        for item in snapshot.items
        if isinstance(item, ToolResultContext)
        and item.model_output_event_id == latest.output_event_id
    }
    if any(call.call_id not in results_by_call for call in latest.tool_calls):
        return None, ()
    return latest, tuple(
        results_by_call[call.call_id] for call in latest.tool_calls
    )


def _representation(
    result: ToolResultContext,
    *,
    desired_length: int,
    allocated_chars: int,
) -> ToolResultRepresentation:
    if allocated_chars == result.original_chars:
        return OriginalRepresentation()
    reason = _preview_reason(
        original_length=result.original_chars,
        desired_length=desired_length,
        allocated_length=allocated_chars,
    )
    prefix, middle, suffix = preview_frame(
        result.result_event_id,
        result.original_chars,
        reason,
    )
    overhead = len(prefix) + len(middle) + len(suffix)
    if overhead > allocated_chars:
        raise ToolResultBudgetError(
            "tool-result budget is too small for required preview metadata "
            f"({allocated_chars} available, {overhead} required)"
        )
    excerpt_chars = allocated_chars - overhead
    head_chars = (excerpt_chars + 1) // 2
    tail_chars = excerpt_chars // 2
    return HeadTailPreview(
        original_chars=result.original_chars,
        rendered_chars=allocated_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
        reason=reason,
    )


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
) -> ToolPreviewReason:
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
    "ToolResultBudgetPolicy",
]
