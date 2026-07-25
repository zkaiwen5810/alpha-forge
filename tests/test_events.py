import unittest
from dataclasses import dataclass

from alpha_forge.events import Event, EventRouter


@dataclass(frozen=True, slots=True)
class First(Event):
    value: str


@dataclass(frozen=True, slots=True)
class Second(Event):
    value: str


class EventRouterTests(unittest.TestCase):
    def test_selective_delivery_preserves_registration_order(self) -> None:
        router = EventRouter()
        received: list[str] = []
        router.subscribe(Event, lambda event: received.append("all"))
        router.subscribe(First, lambda event: received.append(event.value))
        router.subscribe(Second, lambda event: received.append(event.value))

        router.publish(First("first"))

        self.assertEqual(received, ["all", "first"])

    def test_nested_publication_is_immediate_and_ordered(self) -> None:
        router = EventRouter()
        received: list[str] = []

        def publish_second(_event: First) -> None:
            received.append("first:start")
            router.publish(Second("nested"))
            received.append("first:end")

        router.subscribe(First, publish_second)
        router.subscribe(Second, lambda event: received.append(event.value))

        router.publish(First("outer"))

        self.assertEqual(
            received,
            ["first:start", "nested", "first:end"],
        )

    def test_subscription_context_unsubscribes(self) -> None:
        router = EventRouter()
        received: list[str] = []

        with router.subscribe(
            First,
            lambda event: received.append(event.value),
        ):
            router.publish(First("inside"))
        router.publish(First("outside"))

        self.assertEqual(received, ["inside"])


if __name__ == "__main__":
    unittest.main()
