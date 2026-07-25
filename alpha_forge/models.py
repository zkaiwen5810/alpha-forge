"""Small completed domain values shared across architectural layers."""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["TokenUsage", "ToolCall"]
