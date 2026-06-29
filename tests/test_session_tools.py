import asyncio
import unittest

from alpha_forge.chat import ChatStreamEvent
from alpha_forge.config import Config
from alpha_forge.conversation import ToolCall
from alpha_forge.session import (
    MAX_TOOL_ITERATIONS,
    ChatUiState,
    ChatReplController,
    IterationOutput,
    ToolExchange,
    WorkItem,
)


class ScriptedToolChat:
    def __init__(self, iterations):  # type: ignore[no-untyped-def]
        self.iterations = list(iterations)
        self.requests = []

    async def stream_response(self, messages, *, tools):  # type: ignore[no-untyped-def]
        self.requests.append((list(messages), tools))
        for event in self.iterations.pop(0):
            yield event

    def list_models(self) -> list[str]:
        return []


class SessionToolLoopTests(unittest.TestCase):
    def _controller(self, chat):  # type: ignore[no-untyped-def]
        return ChatReplController(Config(api_key="sk-test"), chat=chat)

    def test_tool_call_iterates_until_final_response_and_updates_turn_block(self) -> None:
        chat = ScriptedToolChat(
            [
                [
                    ChatStreamEvent(
                        type="text_delta",
                        text="I'll calculate. ",
                    ),
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=0,
                        call_id="call-1",
                        name="calculator",
                        arguments='{"expression":',
                    ),
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=0,
                        arguments='"2 + 3 * 4"}',
                    ),
                ],
                [
                    ChatStreamEvent(type="text_delta", text="The answer "),
                    ChatStreamEvent(type="text_delta", text="is 14."),
                ],
            ]
        )
        controller = self._controller(chat)
        item = WorkItem("What is 2 + 3 * 4?")
        redraws: list[str] = []
        controller.request_redraw = lambda: redraws.append(
            controller.state.history_text()
        )

        asyncio.run(controller._stream_response(item))

        self.assertEqual(len(chat.requests), 2)
        second_messages = chat.requests[1][0]
        self.assertEqual(second_messages[-2].tool_calls[0].id, "call-1")
        self.assertEqual(second_messages[-1].role, "tool")
        self.assertEqual(second_messages[-1].tool_call_id, "call-1")
        self.assertEqual(second_messages[-1].content, "14")
        self.assertTrue(chat.requests[0][1])
        history = controller.state.history_text()
        self.assertIn(
            '  Tool call [calculator]: {"expression":"2 + 3 * 4"}',
            history,
        )
        self.assertIn("  Tool result [calculator]: 14", history)
        self.assertIn("  Assistant note: I'll calculate. ", history)
        self.assertIn("Assistant: The answer is 14.", history)
        self.assertLess(history.index("Tool call"), history.index("Tool result"))
        self.assertLess(
            history.index("Tool result"),
            history.index("I'll calculate."),
        )
        call_only = next(
            snapshot
            for snapshot in redraws
            if "Tool call [calculator]" in snapshot
            and "Tool result [calculator]" not in snapshot
        )
        result_snapshot = next(
            snapshot
            for snapshot in redraws
            if "Tool result [calculator]" in snapshot
        )
        self.assertNotIn("Tool result", call_only)
        self.assertLess(
            redraws.index(call_only),
            redraws.index(result_snapshot),
        )
        self.assertTrue(
            any("Assistant: I'll calculate. " in snapshot for snapshot in redraws)
        )
        self.assertTrue(
            any(
                "  Assistant note: I'll calculate. " in snapshot
                and "Tool call [calculator]" not in snapshot
                for snapshot in redraws
            )
        )

    def test_multiple_calls_and_tool_errors_are_returned_to_model(self) -> None:
        chat = ScriptedToolChat(
            [
                [
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=0,
                        call_id="call-1",
                        name="calculator",
                        arguments='{"expression":"6 / 2"}',
                    ),
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=1,
                        call_id="call-2",
                        name="missing",
                        arguments="{}",
                    ),
                ],
                [ChatStreamEvent(type="text_delta", text="Done.")],
            ]
        )
        controller = self._controller(chat)
        redraws: list[str] = []
        controller.request_redraw = lambda: redraws.append(
            controller.state.history_text()
        )

        asyncio.run(
            controller._stream_response(WorkItem("run both"))
        )

        tool_messages = [
            message for message in chat.requests[1][0] if message.role == "tool"
        ]
        self.assertEqual([message.content for message in tool_messages[:1]], ["3.0"])
        self.assertIn("error: unknown tool: missing", tool_messages[1].content or "")
        history = controller.state.history_text()
        self.assertIn("  Tool result [calculator]: 3.0", history)
        self.assertIn("  Tool error [missing]: error: unknown tool: missing", history)
        call_one = next(
            index
            for index, snapshot in enumerate(redraws)
            if "Tool call [calculator]" in snapshot
            and "Tool result [calculator]" not in snapshot
            and "Tool call [missing]" not in snapshot
        )
        result_one = next(
            index
            for index, snapshot in enumerate(redraws)
            if "Tool result [calculator]" in snapshot
            and "Tool call [missing]" not in snapshot
        )
        call_two = next(
            index
            for index, snapshot in enumerate(redraws)
            if "Tool call [missing]" in snapshot
            and "Tool error [missing]" not in snapshot
        )
        result_two = next(
            index
            for index, snapshot in enumerate(redraws)
            if "Tool error [missing]" in snapshot
        )
        self.assertLess(call_one, result_one)
        self.assertLess(result_one, call_two)
        self.assertLess(call_two, result_two)

    def test_malformed_arguments_become_failed_tool_result(self) -> None:
        controller = self._controller(ScriptedToolChat([]))

        result, failed = controller._execute_tool_call(
            ToolCall("call-1", "calculator", "{not json")
        )

        self.assertTrue(failed)
        self.assertTrue(result.startswith("error:"))

    def test_queued_turn_uses_completed_history_from_previous_turn(self) -> None:
        chat = ScriptedToolChat(
            [
                [ChatStreamEvent(type="text_delta", text="first reply")],
                [ChatStreamEvent(type="text_delta", text="second reply")],
            ]
        )

        async def _scenario():  # type: ignore[no-untyped-def]
            controller = self._controller(chat)
            consumer = asyncio.create_task(controller.consume())
            controller.submit("first")
            controller.submit("second")
            controller.request_exit()
            await consumer
            return controller

        asyncio.run(_scenario())

        second_request = chat.requests[1][0]
        self.assertEqual(
            [(message.role, message.content) for message in second_request],
            [
                ("system", "You are Alpha Forge, a concise and helpful assistant."),
                ("user", "first"),
                ("assistant", "first reply"),
                ("user", "second"),
            ],
        )

    def test_next_turn_preserves_completed_tool_protocol_messages(self) -> None:
        chat = ScriptedToolChat(
            [
                [
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=0,
                        call_id="call-1",
                        name="calculator",
                        arguments='{"expression":"2*9"}',
                    )
                ],
                [ChatStreamEvent(type="text_delta", text="The answer is 18.")],
                [ChatStreamEvent(type="text_delta", text="The answer is 0.6.")],
            ]
        )

        async def _scenario():  # type: ignore[no-untyped-def]
            controller = self._controller(chat)
            consumer = asyncio.create_task(controller.consume())
            controller.submit("calculate 2 * 9")
            await controller.queue.join()
            controller.submit("3 / 5")
            await controller.queue.join()
            controller.request_exit()
            await consumer

        asyncio.run(_scenario())

        continuation_roles = [message.role for message in chat.requests[1][0]]
        self.assertEqual(
            continuation_roles,
            ["system", "user", "assistant", "tool"],
        )
        next_turn = chat.requests[2][0]
        self.assertEqual(
            [(message.role, message.content) for message in next_turn],
            [
                ("system", "You are Alpha Forge, a concise and helpful assistant."),
                ("user", "calculate 2 * 9"),
                ("assistant", None),
                ("tool", "18"),
                ("assistant", "The answer is 18."),
                ("user", "3 / 5"),
            ],
        )
        self.assertEqual(next_turn[2].tool_calls[0].id, "call-1")
        self.assertEqual(next_turn[3].tool_call_id, "call-1")

    def test_stops_after_tool_iteration_limit(self) -> None:
        iterations = [
            [
                ChatStreamEvent(
                    type="tool_call_delta",
                    index=0,
                    call_id=f"call-{index}",
                    name="calculator",
                    arguments='{"expression":"1+1"}',
                )
            ]
            for index in range(MAX_TOOL_ITERATIONS)
        ]
        chat = ScriptedToolChat(iterations)
        controller = self._controller(chat)

        asyncio.run(
            controller._stream_response(WorkItem("loop forever"))
        )

        self.assertEqual(len(chat.requests), MAX_TOOL_ITERATIONS)
        self.assertIn(
            f"request failed: tool iteration limit reached ({MAX_TOOL_ITERATIONS})",
            controller.state.history_text(),
        )


class BlockHistoryTests(unittest.TestCase):
    def test_renders_iterations_as_one_block_and_standalone_notice_as_next(self) -> None:
        state = ChatUiState()
        turn = state.start_turn("first line\nsecond line")
        state.set_iteration(
            turn,
            0,
            IterationOutput(
                assistant_text="checking\ncarefully",
                tools=(
                    ToolExchange(
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        result="4",
                    ),
                ),
                tool_requesting=True,
            ),
        )
        state.set_iteration(
            turn,
            1,
            IterationOutput(assistant_text="The answer is 4."),
        )
        state.finish_turn(turn)
        state.add_notice("model changed")

        lines = state.history_lines()
        text = state.history_text()

        self.assertEqual(
            [line.role for line in lines],
            [
                "user",
                "user",
                "tool_call",
                "tool_result",
                "assistant_note",
                "assistant_note",
                "assistant",
                "spacer",
                "notice",
            ],
        )
        self.assertEqual(
            text,
            "\n".join(
                [
                    "You: first line",
                    "     second line",
                    '  Tool call [calculator]: {"expression":"2+2"}',
                    "  Tool result [calculator]: 4",
                    "  Assistant note: checking",
                    "                  carefully",
                    "Assistant: The answer is 4.",
                    "",
                    "Notice: model changed",
                ]
            ),
        )

    def test_streaming_iteration_snapshot_is_reclassified_as_assistant_note(self) -> None:
        state = ChatUiState()
        turn = state.start_turn("calculate")
        state.set_iteration(
            turn,
            0,
            IterationOutput(assistant_text="Let me check"),
        )
        self.assertIn("Assistant: Let me check", state.history_text())

        state.set_iteration(
            turn,
            0,
            IterationOutput(
                assistant_text="Let me check",
                tool_requesting=True,
            ),
        )

        self.assertNotIn("\nAssistant: Let me check", state.history_text())
        self.assertIn("  Assistant note: Let me check", state.history_text())
