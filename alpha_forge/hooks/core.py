"""Ordered lifecycle hook registration and dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from alpha_forge.hooks.events import LifecycleEvent
from alpha_forge.hooks.matcher import HookMatcher

# Hook actions are side-effect-only guards/observers. Dispatch deliberately has no
# return-value channel; an action must raise to abort the intercepted operation.
type HookAction[EventType: LifecycleEvent] = Callable[[EventType], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Hook[EventType: LifecycleEvent]:
    matcher: HookMatcher[EventType]
    action: HookAction[EventType]


class HookRegistry:
    """Dispatch matching hooks serially in registration order."""

    def __init__(self, hooks: Iterable[Hook[Any]] | None = None) -> None:
        self._hooks: list[Hook[Any]] = list(hooks or ())

    def register[EventType: LifecycleEvent](self, hook: Hook[EventType]) -> None:
        self._hooks.append(cast(Hook[Any], hook))

    async def dispatch(self, event: LifecycleEvent) -> None:
        for hook in tuple(self._hooks):
            if hook.matcher.matches(event):
                # Do not collect action results. Exceptions propagate so a guard
                # hook can stop the lifecycle before the caller continues.
                await hook.action(event)


__all__ = ["Hook", "HookAction", "HookRegistry"]
