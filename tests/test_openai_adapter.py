import asyncio
import unittest
from types import SimpleNamespace

from alpha_forge.config import Config
from alpha_forge.context import (
    ModelContextSnapshot,
    ModelOutputContext,
    SystemMessage,
    ToolResultContext,
    UserMessage,
)
from alpha_forge.providers import (
    OutputText,
    StreamCompleted,
    TextDelta,
    TokenUsage,
    ToolCall,
)
from alpha_forge.providers.openai_chat import OpenAIChatAdapter
from alpha_forge.tools import ToolSpec
from alpha_forge.transcript import OriginalRepresentation


class FakeModels:
    def list(self):
        return SimpleNamespace(
            data=[SimpleNamespace(id="model-b"), SimpleNamespace(id="model-a")]
        )


class FakeAsyncCompletions:
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
                            reasoning_content=None,
                            refusal=None,
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
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            refusal=None,
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
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=25,
                    total_tokens=125,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=75),
                ),
            ),
        ]

        async def stream():
            for chunk in chunks:
                yield chunk

        return stream()


class OpenAIChatAdapterTests(unittest.TestCase):
    def _adapter(self, completions: object) -> OpenAIChatAdapter:
        sync = SimpleNamespace(
            models=FakeModels(),
            chat=SimpleNamespace(completions=SimpleNamespace()),
        )
        async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        return OpenAIChatAdapter(
            Config(api_key="sk-test", model="gpt-test"),
            client=sync,
            async_client=async_client,
        )

    def test_client_kwargs_use_generic_base_url_only_when_configured(self) -> None:
        self.assertEqual(
            OpenAIChatAdapter._client_kwargs(
                Config("key", timeout=12.5)
            ),
            {"api_key": "key", "timeout": 12.5},
        )
        self.assertEqual(
            OpenAIChatAdapter._client_kwargs(
                Config("key", base_url="http://gateway/v1", timeout=2)
            )["base_url"],
            "http://gateway/v1",
        )

    def test_stream_translates_context_tools_and_completed_output(self) -> None:
        completions = FakeAsyncCompletions()
        adapter = self._adapter(completions)
        context = ModelContextSnapshot(
            7,
            (
                SystemMessage("system"),
                UserMessage("prompt-id", "calculate"),
                ModelOutputContext(
                    "output-id",
                    "prompt-id",
                    (ToolCall("old-call", "calculator", "{}"),),
                ),
                ToolResultContext(
                    "result-id",
                    "output-id",
                    "old-call",
                    "success",
                    "4",
                    1,
                    OriginalRepresentation(),
                ),
            ),
        )
        spec = ToolSpec(
            "calculator",
            "Calculate.",
            {"type": "object"},
        )

        async def collect():
            return [
                event
                async for event in adapter.stream(context, tools=(spec,))
            ]

        events = asyncio.run(collect())

        self.assertIsInstance(events[0], TextDelta)
        self.assertIsInstance(events[-1], StreamCompleted)
        self.assertEqual(
            events[-1].output.tool_calls,
            (ToolCall("call-1", "calculator", '{"expression":"2+2"}'),),
        )
        self.assertEqual(
            events[-1].output.usage,
            TokenUsage(100, 75, 25, 125),
        )
        self.assertEqual(
            completions.request["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "calculate"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "old-call",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "4", "tool_call_id": "old-call"},
            ],
        )
        self.assertEqual(
            completions.request["tools"][0]["function"]["name"],
            "calculator",
        )
        self.assertNotIn("strict", completions.request["tools"][0]["function"])

    def test_list_models_returns_sorted_ids(self) -> None:
        adapter = self._adapter(FakeAsyncCompletions())
        self.assertEqual(adapter.list_models(), ["model-a", "model-b"])


if __name__ == "__main__":
    unittest.main()
