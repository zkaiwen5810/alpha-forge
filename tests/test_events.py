import unittest
from dataclasses import dataclass

from alpha_forge.application.events import ApplicationEvent
from alpha_forge.application.router import ApplicationEventRouter
from alpha_forge.hooks import PreToolExecution
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.query import PrepareContext, ProviderRequestStarted
from alpha_forge.transcript import InputAccepted


@dataclass(frozen=True, slots=True)
class First(ApplicationEvent):
    value: str


@dataclass(frozen=True, slots=True)
class Second(ApplicationEvent):
    value: str


class EventRouterTests(unittest.TestCase):
    def test_router_rejects_other_message_families(self) -> None:
        router = ApplicationEventRouter()
        received = []
        router.subscribe(ApplicationEvent, received.append)
        for message in (
            PrepareContext("prompt"),
            ProviderRequestStarted("prompt", "request"),
            InputAccepted("prompt", "hello"),
            PreToolExecution(
                call_id="call", tool_name="bash", tool_input=FrozenJsonObject({})
            ),
        ):
            with self.subTest(message=type(message).__name__):
                with self.assertRaises(TypeError):
                    router.publish(message)
                with self.assertRaises(TypeError):
                    router.subscribe(type(message), received.append)
        self.assertEqual(received, [])

    def test_subscriber_failure_propagates_and_stops_delivery(self) -> None:
        router = ApplicationEventRouter()
        received = []

        def fail(_event):
            raise RuntimeError("subscriber failed")

        router.subscribe(First, fail)
        router.subscribe(First, received.append)
        with self.assertRaisesRegex(RuntimeError, "subscriber failed"):
            router.publish(First("value"))
        self.assertEqual(received, [])

    def test_selective_delivery_preserves_registration_order(self) -> None:
        router = ApplicationEventRouter()
        received: list[str] = []
        router.subscribe(ApplicationEvent, lambda event: received.append("all"))
        router.subscribe(First, lambda event: received.append(event.value))
        router.subscribe(Second, lambda event: received.append(event.value))

        router.publish(First("first"))

        self.assertEqual(received, ["all", "first"])

    def test_nested_publication_is_immediate_and_ordered(self) -> None:
        router = ApplicationEventRouter()
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
        router = ApplicationEventRouter()
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
