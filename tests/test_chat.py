import asyncio
import unittest
from types import SimpleNamespace

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config
from alpha_forge.conversation import Conversation


class FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="assistant reply")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeModels:
    def __init__(self) -> None:
        self.response = SimpleNamespace(
            data=[
                SimpleNamespace(id="model-b"),
                SimpleNamespace(id="model-a"),
            ]
        )

    def list(self):
        return self.response


class FakeAsyncChunkStream:
    """Async iterator of synthetic chat-completion chunks."""

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces
        self._index = 0

    def __aiter__(self) -> "FakeAsyncChunkStream":
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._index >= len(self._pieces):
            raise StopAsyncIteration
        piece = self._pieces[self._index]
        self._index += 1
        delta = SimpleNamespace(content=piece) if piece else SimpleNamespace(content=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeAsyncCompletions:
    def __init__(self) -> None:
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        return FakeAsyncChunkStream(["Hel", "lo, ", "world"])


class ChatClientTests(unittest.TestCase):
    def test_sends_chat_completions_request_with_explicit_history(self) -> None:
        completions = FakeCompletions()
        openai_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = Config(api_key="sk-test", model="gpt-test")
        client = ChatClient(config, client=openai_client)
        conversation = Conversation(system_prompt="system")
        conversation.add_user("hello")

        reply = client.complete(conversation.messages)

        self.assertEqual(reply, "assistant reply")
        self.assertEqual(completions.request["model"], "gpt-test")
        self.assertEqual(
            completions.request["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertNotIn("tools", completions.request)
        self.assertNotIn("previous_response_id", completions.request)

    def test_list_models_returns_sorted_ids(self) -> None:
        openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions()),
            models=FakeModels(),
        )
        config = Config(api_key="sk-test", model="model-a")
        client = ChatClient(config, client=openai_client)

        self.assertEqual(client.list_models(), ["model-a", "model-b"])

    def test_stream_yields_delta_content_in_order_and_requests_stream(self) -> None:
        completions = FakeAsyncCompletions()
        async_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = Config(api_key="sk-test", model="gpt-test")
        client = ChatClient(config, async_client=async_client)
        conversation = Conversation(system_prompt="system")
        conversation.add_user("hello")

        async def _collect() -> list[str]:
            return [chunk async for chunk in client.stream(conversation.messages)]

        pieces = asyncio.run(_collect())

        self.assertEqual(pieces, ["Hel", "lo, ", "world"])
        self.assertEqual(completions.request["model"], "gpt-test")
        self.assertTrue(completions.request["stream"])
        self.assertEqual(
            completions.request["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
        )
