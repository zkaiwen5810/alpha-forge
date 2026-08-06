"""Permission action for pre-tool-execution hooks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from alpha_forge.hooks.events import PreToolExecution

type PermissionRequester = Callable[[PreToolExecution], Awaitable[bool]]


class PermissionDeniedError(RuntimeError):
    """Raised when a permission action rejects a tool call."""


@dataclass(frozen=True, slots=True)
class PermissionAction:
    requester: PermissionRequester

    async def __call__(self, event: PreToolExecution) -> None:
        if not await self.requester(event):
            # Hook actions cannot return a decision to the executor. Raising is
            # the permission hook's denial signal and prevents tool invocation.
            raise PermissionDeniedError(
                f"permission denied for tool {event.tool_name!r}"
            )


__all__ = [
    "PermissionAction",
    "PermissionDeniedError",
    "PermissionRequester",
]
