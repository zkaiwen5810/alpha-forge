"""Typed conditions deciding whether hooks apply to lifecycle events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from alpha_forge.hooks.events import LifecycleEvent, PreToolExecution


@dataclass(frozen=True, slots=True)
class HookMatcher[EventType: LifecycleEvent]:
    """Match one lifecycle event type and an event-specific predicate."""

    event_type: type[EventType]
    predicate: Callable[[EventType], bool]

    def matches(self, event: LifecycleEvent) -> bool:
        return isinstance(event, self.event_type) and self.predicate(
            cast(EventType, event)
        )


def match_lifecycle[EventType: LifecycleEvent](
    event_type: type[EventType],
    predicate: Callable[[EventType], bool] | None = None,
) -> HookMatcher[EventType]:
    """Build a matcher for one lifecycle and an optional typed condition."""

    return HookMatcher(event_type, predicate or (lambda _event: True))


def match_tool_names(*tool_names: str) -> HookMatcher[PreToolExecution]:
    """Match pre-execution events for exact canonical tool names."""

    names = frozenset(tool_names)
    if not names or any(not name for name in names):
        raise ValueError("tool-name matcher requires non-empty names")
    return match_lifecycle(
        PreToolExecution,
        lambda event: event.tool_name in names,
    )


__all__ = ["HookMatcher", "match_lifecycle", "match_tool_names"]
