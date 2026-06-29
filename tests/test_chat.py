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


class FakeToolAsyncCompletions:
    def __init__(self) -> None:
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Working",
                            tool_calls=[],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="calculator",
                                        arguments='{"expression":',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='"2+2"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
        ]

        async def _stream():  # type: ignore[no-untyped-def]
            for chunk in chunks:
                yield chunk

        return _stream()


class ChatClientTests(unittest.TestCase):
    def test_client_kwargs_include_configured_timeout(self) -> None:
        config = Config(api_key="sk-test", timeout=12.5)

        self.assertEqual(
            ChatClient._client_kwargs(config),
            {"api_key": "sk-test", "timeout": 12.5},
        )

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

    def test_stream_response_yields_fragmented_tool_call_deltas(self) -> None:
        completions = FakeToolAsyncCompletions()
        async_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = Config(api_key="sk-test", model="gpt-test")
        client = ChatClient(config, async_client=async_client)
        conversation = Conversation()
        conversation.add_user("calculate")
        tools = [{"type": "function", "function": {"name": "calculator"}}]

        async def _collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in client.stream_response(
                    conversation.messages,
                    tools=tools,
                )
            ]

        events = asyncio.run(_collect())

        self.assertEqual(
            [event.type for event in events],
            ["text_delta", "tool_call_delta", "tool_call_delta"],
        )
        self.assertEqual(events[0].text, "Working")
        self.assertEqual(events[1].call_id, "call-1")
        self.assertEqual(events[1].name, "calculator")
        self.assertEqual(events[2].arguments, '"2+2"}')
        self.assertEqual(completions.request["tools"], tools)
        self.assertTrue(completions.request["stream"])
