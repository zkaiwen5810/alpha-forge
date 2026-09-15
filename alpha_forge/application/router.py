"""Typed synchronous event publication with deterministic fan-out."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from alpha_forge.application.events import ApplicationEvent


@dataclass(slots=True)
class _Subscription[EventType: ApplicationEvent]:
    router: ApplicationEventRouter
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


class ApplicationEventRouter:
    """Publish events immediately to matching handlers in registration order."""

    def __init__(self) -> None:
        self._subscriptions: list[_Subscription[ApplicationEvent]] = []

    def subscribe[EventType: ApplicationEvent](
        self,
        event_type: type[EventType],
        handler: Callable[[EventType], None],
    ) -> _Subscription[EventType]:
        if not issubclass(event_type, ApplicationEvent):
            raise TypeError("subscriptions require an ApplicationEvent type")
        subscription = _Subscription(self, event_type, handler)
        self._subscriptions.append(cast(_Subscription[ApplicationEvent], subscription))
        return subscription

    def publish(self, event: ApplicationEvent) -> None:
        if not isinstance(event, ApplicationEvent):
            raise TypeError("only ApplicationEvent notifications can be published")
        # Snapshotting permits handlers to subscribe, unsubscribe, or publish
        # nested events without disturbing the current delivery pass.
        for subscription in tuple(self._subscriptions):
            if subscription.active and isinstance(
                event,
                subscription.event_type,
            ):
                subscription.handler(event)

    def _unsubscribe[EventType: ApplicationEvent](
        self,
        subscription: _Subscription[EventType],
    ) -> None:
        try:
            self._subscriptions.remove(cast(_Subscription[ApplicationEvent], subscription))
        except ValueError:
            pass


__all__ = ["ApplicationEventRouter"]
