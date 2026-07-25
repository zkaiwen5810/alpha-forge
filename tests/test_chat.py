import asyncio
import unittest
from types import SimpleNamespace

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config
from alpha_forge.model_messages import SystemMessage, ToolCall, UserMessage
from alpha_forge.streaming import (
    ModelResponse,
    StreamCompleted,
    TokenUsage,
)


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
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    total_tokens=125,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=75),
                ),
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
        reply = client.complete([SystemMessage("system"), UserMessage("hello")])

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
        tools = [{"type": "function", "function": {"name": "calculator"}}]

        async def _collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in client.stream_response(
                    [UserMessage("calculate")],
                    tools=tools,
                )
            ]

        events = asyncio.run(_collect())

        self.assertEqual(
            [event.type for event in events],
            [
                "text_delta",
                "tool_call_delta",
                "tool_call_delta",
                "usage",
                "completed",
            ],
        )
        self.assertEqual(events[0].text, "Working")
        self.assertEqual(events[1].call_id, "call-1")
        self.assertEqual(events[1].name, "calculator")
        self.assertEqual(events[2].arguments, '"2+2"}')
        self.assertEqual(events[3].usage.prompt_tokens, 100)
        self.assertEqual(events[3].usage.cached_tokens, 75)
        self.assertEqual(events[3].usage.total_tokens, 125)
        self.assertEqual(
            events[4].response,
            ModelResponse(
                content="Working",
                tool_calls=(
                    ToolCall(
                        "call-1",
                        "calculator",
                        '{"expression":"2+2"}',
                    ),
                ),
                usage=TokenUsage(100, 75, 125),
            ),
        )
        self.assertEqual(completions.request["tools"], tools)
        self.assertTrue(completions.request["stream"])
        self.assertEqual(
            completions.request["stream_options"],
            {"include_usage": True},
        )

    def test_stream_response_preserves_chunk_content_order_and_completes_last(
        self,
    ) -> None:
        class MixedContentCompletions:
            async def create(self, **_kwargs):  # type: ignore[no-untyped-def]
                mixed = SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="answer",
                                reasoning_content="reason",
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call-1",
                                        function=SimpleNamespace(
                                            name="calculator",
                                            arguments='{"expression":"2+2"}',
                                        ),
                                    )
                                ],
                                refusal="refusal",
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=None,
                )
                usage = SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        total_tokens=20,
                        prompt_tokens_details=None,
                    ),
                )

                async def _stream():  # type: ignore[no-untyped-def]
                    yield mixed
                    yield usage

                return _stream()

        client = ChatClient(
            Config(api_key="sk-test"),
            async_client=SimpleNamespace(
                chat=SimpleNamespace(
                    completions=MixedContentCompletions(),
                )
            ),
        )

        async def _collect():  # type: ignore[no-untyped-def]
            return [event async for event in client.stream_response([], tools=[])]

        events = asyncio.run(_collect())

        self.assertEqual(
            [event.type for event in events],
            [
                "text_delta",
                "reasoning_delta",
                "tool_call_delta",
                "refusal_delta",
                "usage",
                "completed",
            ],
        )
        self.assertEqual(events[0].text, "answer")
        self.assertEqual(events[1].text, "reason")
        self.assertEqual(events[3].text, "refusal")
        self.assertEqual(
            events[-1].response,
            ModelResponse(
                content="answer",
                tool_calls=(
                    ToolCall(
                        "call-1",
                        "calculator",
                        '{"expression":"2+2"}',
                    ),
                ),
                reasoning_content="reason",
                refusal="refusal",
                finish_reason="tool_calls",
                usage=TokenUsage(10, None, 20),
            ),
        )

    def test_stream_response_keeps_total_without_cached_token_details(self) -> None:
        class UsageWithoutCacheCompletions:
            async def create(self, **_kwargs):  # type: ignore[no-untyped-def]
                async def _stream():  # type: ignore[no-untyped-def]
                    yield SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(
                            prompt_tokens=20,
                            total_tokens=30,
                            prompt_tokens_details=None,
                        ),
                    )

                return _stream()

        async_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=UsageWithoutCacheCompletions(),
            )
        )
        client = ChatClient(
            Config(api_key="sk-test"),
            async_client=async_client,
        )

        async def _collect():  # type: ignore[no-untyped-def]
            return [event async for event in client.stream_response([], tools=[])]

        events = asyncio.run(_collect())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].usage.prompt_tokens, 20)
        self.assertIsNone(events[0].usage.cached_tokens)
        self.assertEqual(events[0].usage.total_tokens, 30)
        self.assertEqual(
            events[1],
            StreamCompleted(
                ModelResponse(
                    content=None,
                    usage=TokenUsage(20, None, 30),
                )
            ),
        )

    def test_stream_response_accepts_gateway_usage_aliases_and_dicts(self) -> None:
        class CompatibleGatewayCompletions:
            async def create(self, **_kwargs):  # type: ignore[no-untyped-def]
                async def _stream():  # type: ignore[no-untyped-def]
                    yield SimpleNamespace(
                        choices=[],
                        usage={
                            "prompt_tokens": 80,
                            "prompt_cache_hit_tokens": 40,
                            "total_tokens": 100,
                        },
                    )

                return _stream()

        async_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=CompatibleGatewayCompletions(),
            )
        )
        client = ChatClient(
            Config(api_key="sk-test"),
            async_client=async_client,
        )

        async def _collect():  # type: ignore[no-untyped-def]
            return [event async for event in client.stream_response([], tools=[])]

        events = asyncio.run(_collect())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].usage.prompt_tokens, 80)
        self.assertEqual(events[0].usage.cached_tokens, 40)
        self.assertEqual(events[0].usage.total_tokens, 100)
        self.assertEqual(
            events[1],
            StreamCompleted(
                ModelResponse(
                    content=None,
                    usage=TokenUsage(80, 40, 100),
                )
            ),
        )
