"""Typed synchronous event publication with deterministic fan-out."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast


class Event:
    """Marker base for events published inside the application."""


@dataclass(slots=True)
class _Subscription[EventType: Event]:
    router: EventRouter
    event_type: type[EventType]
    handler: Callable[[EventType], None]
    active: bool = True

    def unsubscribe(self) -> None:
        if self.active:
            self.router._unsubscribe(self)
            self.active = False

    def __enter__(self) -> _Subscription[EventType]:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.unsubscribe()


class EventRouter:
    """Publish events immediately to matching handlers in registration order."""

    def __init__(self) -> None:
        self._subscriptions: list[_Subscription[Event]] = []

    def subscribe[EventType: Event](
        self,
        event_type: type[EventType],
        handler: Callable[[EventType], None],
    ) -> _Subscription[EventType]:
        subscription = _Subscription(self, event_type, handler)
        self._subscriptions.append(cast(_Subscription[Event], subscription))
        return subscription

    def publish(self, event: Event) -> None:
        # Snapshotting permits handlers to subscribe, unsubscribe, or publish
        # nested events without disturbing the current delivery pass.
        for subscription in tuple(self._subscriptions):
            if subscription.active and isinstance(
                event,
                subscription.event_type,
            ):
                subscription.handler(event)

    def _unsubscribe[EventType: Event](
        self,
        subscription: _Subscription[EventType],
    ) -> None:
        try:
            self._subscriptions.remove(cast(_Subscription[Event], subscription))
        except ValueError:
            pass


__all__ = ["Event", "EventRouter"]
