import asyncio
import unittest

from alpha_forge.model_messages import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)
from alpha_forge.models import ToolCall
from alpha_forge.prompt_editor import (
    EditedPrompt,
    PromptDraft,
    ToolResultPromptEditor,
)
from alpha_forge.query import (
    ModelDeltaReceived,
    ModelRoundCompleted,
    ModelRoundStarted,
    QueryCompleted,
    QueryEngine,
    QueryRequest,
    ToolBatchStarted,
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
        model_result = model.requests[1][-1]["content"]
        assert isinstance(model_result, str)
        self.assertEqual(len(model_result), 300)
        self.assertIn("transcript_ref", model_result)
        self.assertEqual(model.requests[1][-2]["role"], "assistant")
        self.assertEqual(model.requests[1][-1]["role"], "tool")
        edit_position = events.index(edited)
        round_starts = [
            index
            for index, event in enumerate(events)
            if isinstance(event, ModelRoundStarted)
        ]
        self.assertLess(edit_position, round_starts[1])

    def test_editor_runs_once_per_iteration_only_before_client_call(self) -> None:
        trace: list[str] = []
        call = ToolCall("call", "echo", "{}")

        class TracedModel(ScriptedModel):
            async def stream_response(  # type: ignore[no-untyped-def]
                self,
                messages,
                *,
                tools,
            ):
                trace.append("client")
                async for event in super().stream_response(
                    messages,
                    tools=tools,
                ):
                    yield event

        class TracedEditor:
            def __init__(self) -> None:
                self.delegate = ToolResultPromptEditor()

            def edit(self, draft: PromptDraft):  # type: ignore[no-untyped-def]
                trace.append(
                    "edit:pending"
                    if any(
                        isinstance(message, ToolMessage) and message.raw
                        for message in draft.messages
                    )
                    else "edit:empty"
                )
                return self.delegate.edit(draft)

        class TracedExecutor(RecordingExecutor):
            async def execute(self, requested: ToolCall) -> ExecutedToolResult:
                trace.append("tool")
                return await super().execute(requested)

        query = QueryEngine(
            TracedModel(
                [
                    [StreamCompleted(ModelResponse(None, (call,)))],
                    [StreamCompleted(ModelResponse("done"))],
                ]
            ),  # type: ignore[arg-type]
            prompt_editor=TracedEditor(),
        )

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(UserMessage("run"),),
                        tool_definitions=(),
                        tool_executor=TracedExecutor(),
                    )
                )
            ]

        asyncio.run(collect())

        self.assertEqual(
            trace,
            [
                "edit:empty",
                "client",
                "tool",
                "edit:pending",
                "client",
            ],
        )

    def test_initial_unfinished_messages_synthesize_only_missing_results(
        self,
    ) -> None:
        calls = (
            ToolCall("one", "tool", "{}"),
            ToolCall("two", "tool", "{}"),
        )
        model = ScriptedModel(
            [[StreamCompleted(ModelResponse("recovered"))]]
        )
        executor = RecordingExecutor()
        query = QueryEngine(model)  # type: ignore[arg-type]

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(
                            UserMessage("run"),
                            AssistantMessage(
                                None,
                                calls,
                                output_id="prior-output",
                            ),
                            ToolMessage(
                                "first",
                                "one",
                                result_id="result-one",
                                raw=True,
                            ),
                        ),
                        tool_definitions=(),
                        tool_executor=executor,
                    )
                )
            ]

        events = asyncio.run(collect())
        started = next(
            event for event in events if isinstance(event, ToolBatchStarted)
        )
        produced = [
            event
            for event in events
            if isinstance(event, ToolResultProduced)
        ]

        self.assertEqual(
            [result.call_id for result in started.results],
            ["one"],
        )
        self.assertEqual([event.result.call_id for event in produced], ["two"])
        self.assertTrue(produced[0].result.failed)
        self.assertEqual(executor.calls, [])
        self.assertEqual(
            [message["role"] for message in model.requests[0]],
            ["user", "assistant", "tool", "tool"],
        )

    def test_raw_tool_messages_cannot_reach_model_client(self) -> None:
        call = ToolCall("call", "tool", "{}")
        model = ScriptedModel(
            [[StreamCompleted(ModelResponse("must not run"))]]
        )

        class NoOpEditor:
            def edit(self, draft: PromptDraft) -> EditedPrompt:
                return EditedPrompt(draft.messages)

        query = QueryEngine(  # type: ignore[arg-type]
            model,
            prompt_editor=NoOpEditor(),
        )

        async def collect():  # type: ignore[no-untyped-def]
            return [
                event
                async for event in query.run(
                    QueryRequest(
                        messages=(
                            UserMessage("run"),
                            AssistantMessage(
                                None,
                                (call,),
                                output_id="output",
                            ),
                            ToolMessage(
                                "raw",
                                "call",
                                result_id="result",
                                raw=True,
                            ),
                        ),
                        tool_definitions=(),
                        tool_executor=RecordingExecutor(),
                    )
                )
            ]

        with self.assertRaisesRegex(RuntimeError, "left raw tool messages"):
            asyncio.run(collect())
        self.assertEqual(model.requests, [])

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
