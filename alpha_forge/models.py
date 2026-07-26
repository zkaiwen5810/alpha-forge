"""Small completed domain values shared across architectural layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptEditReason = Literal[
    "individual_limit",
    "aggregate_limit",
    "individual_and_aggregate_limits",
]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One complete function call requested by a model output."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class RawToolResult:
    """One complete application-side tool result before prompt editing."""

    result_id: str
    call_id: str
    content: str
    failed: bool = False


@dataclass(frozen=True, slots=True)
class PromptEditDecision:
    """How one raw tool result is represented in a model prompt."""

    result_id: str
    call_id: str
    allocated_chars: int
    reason: PromptEditReason | None = None


@dataclass(frozen=True, slots=True)
class PromptEdit:
    """Versioned, replayable prompt-edit policy output for one tool batch."""

    policy_version: Literal["head_tail_v1"]
    individual_limit: int
    aggregate_limit: int
    decisions: tuple[PromptEditDecision, ...]


@dataclass(frozen=True, slots=True)
class EditedToolResult:
    """Bounded tool result ready to be placed in a model prompt or UI."""

    result_id: str
    call_id: str
    content: str
    failed: bool
    previewed: bool


__all__ = [
    "EditedToolResult",
    "PromptEdit",
    "PromptEditDecision",
    "PromptEditReason",
    "RawToolResult",
    "TokenUsage",
    "ToolCall",
]
