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
