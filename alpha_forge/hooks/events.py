"""Immutable interception context passed to awaited hook actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alpha_forge.json_values import FrozenJsonObject


@dataclass(frozen=True, slots=True, kw_only=True)
class PreToolExecution:
    """A validated tool call immediately before its handler is invoked."""

    call_id: str
    tool_name: str
    tool_input: FrozenJsonObject
    lifecycle: Literal["PreToolExecution"] = field(
        init=False,
        default="PreToolExecution",
    )


type HookContext = PreToolExecution


__all__ = ["HookContext", "PreToolExecution"]
