import asyncio
import unittest

from alpha_forge.model_messages import UserMessage
from alpha_forge.models import ToolCall
from alpha_forge.prompt_editor import ToolResultPromptEditor
from alpha_forge.query import (
    ModelDeltaReceived,
    ModelRoundCompleted,
    QueryCompleted,
    QueryEngine,
    QueryRequest,
    ToolResultProduced,
    ToolResultsEdited,
)
from alpha_forge.streaming import ModelResponse, StreamCompleted, TextDelta
from alpha_forge.tool_execution import ExecutedToolResult


class ScriptedModel:
    def __init__(self, responses):  # type: ignore[no-untyped-def]
        self.responses = list(responses)
        self.requests: list[list[dict[str, object]]] = []

    async def stream_response(self, messages, *, tools):  # type: ignore[no-untyped-def]
        self.requests.append([message.to_openai() for message in messages])
        for event in self.responses.pop(0):
            yield event


class RecordingExecutor:
    def __init__(self, content: str = "result") -> None:
        self.content = content
        self.calls: list[ToolCall] = []

    async def execute(self, call: ToolCall) -> ExecutedToolResult:
        self.calls.append(call)
        return ExecutedToolResult(call.id, self.content)


class QueryEngineTests(unittest.TestCase):
    def test_terminal_response_forwards_deltas_and_completes(self) -> None:
        model = ScriptedModel(
            [[TextDelta("hello"), StreamCompleted(ModelResponse("hello"))]]
        )
        query = QueryEngine(model)  # type: ignore[arg-type]

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(UserMessage("hi"),),
                        tool_definitions=(),
                        tool_executor=RecordingExecutor(),
                    )
                )
            ]

        events = asyncio.run(collect())

        self.assertTrue(any(isinstance(e, ModelDeltaReceived) for e in events))
        self.assertTrue(any(isinstance(e, ModelRoundCompleted) for e in events))
        self.assertIsInstance(events[-1], QueryCompleted)
        self.assertEqual(model.requests[0], [{"role": "user", "content": "hi"}])

    def test_tool_results_are_edited_before_the_next_model_round(self) -> None:
        call = ToolCall("call", "echo", '{"text":"hi"}')
        model = ScriptedModel(
            [
                [StreamCompleted(ModelResponse(None, (call,)))],
                [StreamCompleted(ModelResponse("done"))],
            ]
        )
        executor = RecordingExecutor("HEAD" + "x" * 500 + "TAIL")
        query = QueryEngine(
            model,  # type: ignore[arg-type]
            prompt_editor=ToolResultPromptEditor(
                individual_limit=300,
                aggregate_limit=300,
            ),
        )

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(UserMessage("run"),),
                        tool_definitions=(
                            {"type": "function", "function": {"name": "echo"}},
                        ),
                        tool_executor=executor,
                    )
                )
            ]

        events = asyncio.run(collect())
        raw = next(e for e in events if isinstance(e, ToolResultProduced))
        edited = next(e for e in events if isinstance(e, ToolResultsEdited))

        self.assertEqual(executor.calls, [call])
        self.assertGreater(len(raw.result.content), 300)
        self.assertEqual(len(edited.results[0].content), 300)
        self.assertIn("transcript_ref", edited.results[0].content)
        self.assertEqual(model.requests[1][-2]["role"], "assistant")
        self.assertEqual(model.requests[1][-1]["role"], "tool")
        self.assertEqual(
            model.requests[1][-1]["content"],
            edited.results[0].content,
        )

    def test_stream_without_completion_fails(self) -> None:
        query = QueryEngine(ScriptedModel([[TextDelta("partial")]]))  # type: ignore[arg-type]

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(UserMessage("hi"),),
                        tool_definitions=(),
                        tool_executor=RecordingExecutor(),
                    )
                )
            ]

        with self.assertRaisesRegex(RuntimeError, "without a completed response"):
            asyncio.run(collect())

    def test_closing_query_closes_active_model_stream(self) -> None:
        class ClosableModel:
            def __init__(self) -> None:
                self.closed = False

            async def stream_response(
                self, messages, *, tools  # type: ignore[no-untyped-def]
            ):
                try:
                    yield TextDelta("partial")
                    await asyncio.Event().wait()
                finally:
                    self.closed = True

        model = ClosableModel()
        query = QueryEngine(model)  # type: ignore[arg-type]

        async def close_early() -> None:
            events = query.run(
                QueryRequest(
                    messages=(UserMessage("hi"),),
                    tool_definitions=(),
                    tool_executor=RecordingExecutor(),
                )
            )
            await anext(events)
            await anext(events)
            await events.aclose()

        asyncio.run(close_early())

        self.assertTrue(model.closed)

    def test_multiple_calls_execute_and_return_in_model_order(self) -> None:
        calls = (
            ToolCall("one", "echo", "{}"),
            ToolCall("two", "echo", "{}"),
        )
        model = ScriptedModel(
            [
                [StreamCompleted(ModelResponse(None, calls))],
                [StreamCompleted(ModelResponse("done"))],
            ]
        )
        executor = RecordingExecutor()
        query = QueryEngine(model)  # type: ignore[arg-type]

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(UserMessage("run"),),
                        tool_definitions=(),
                        tool_executor=executor,
                    )
                )
            ]

        asyncio.run(collect())

        self.assertEqual(executor.calls, list(calls))
        self.assertEqual(
            [message["tool_call_id"] for message in model.requests[1][-2:]],
            ["one", "two"],
        )

    def test_tool_round_limit_is_terminal(self) -> None:
        call = ToolCall("call", "echo", "{}")
        query = QueryEngine(
            ScriptedModel(
                [
                    [StreamCompleted(ModelResponse(None, (call,)))],
                    [StreamCompleted(ModelResponse(None, (call,)))],
                ]
            ),  # type: ignore[arg-type]
            max_tool_rounds=2,
        )

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(UserMessage("run"),),
                        tool_definitions=(),
                        tool_executor=RecordingExecutor(),
                    )
                )
            ]

        with self.assertRaisesRegex(RuntimeError, "tool round limit reached"):
            asyncio.run(collect())


if __name__ == "__main__":
    unittest.main()
