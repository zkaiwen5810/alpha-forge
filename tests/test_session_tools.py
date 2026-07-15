import asyncio
import tempfile
import unittest
from pathlib import Path

import alpha_forge.session as session_compat
from alpha_forge.chat import ChatStreamEvent
from alpha_forge.config import Config
from alpha_forge.conversation import AssistantMessage, ToolCall, ToolMessage
from alpha_forge.repl_controller import (
    MAX_TOOL_ITERATIONS,
    ChatReplController,
    WorkItem,
)
from alpha_forge.tool_results import ToolResultManager
from alpha_forge.tools import Tool, ToolRegistry
from alpha_forge.ui_state import (
    ChatUiState,
    IterationOutput,
    TokenUsage,
    ToolExchange,
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


class SessionCompatibilityTests(unittest.TestCase):
    def test_session_facade_reexports_split_modules(self) -> None:
        self.assertIs(session_compat.ChatReplController, ChatReplController)
        self.assertIs(session_compat.ChatUiState, ChatUiState)
        self.assertIs(session_compat.IterationOutput, IterationOutput)


class SessionToolLoopTests(unittest.TestCase):
    def _controller(self, chat, **kwargs):  # type: ignore[no-untyped-def]
        return ChatReplController(
            Config(api_key="sk-test"),
            chat=chat,
            **kwargs,
        )

    def test_controller_names_ui_state_explicitly(self) -> None:
        controller = self._controller(ScriptedToolChat([]))

        self.assertIsInstance(controller.ui_state, ChatUiState)
        self.assertFalse(hasattr(controller, "state"))

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
                    ChatStreamEvent(
                        type="usage",
                        prompt_tokens=100,
                        cached_tokens=60,
                        total_tokens=125,
                    ),
                ],
                [
                    ChatStreamEvent(type="text_delta", text="The answer "),
                    ChatStreamEvent(type="text_delta", text="is 14."),
                    ChatStreamEvent(
                        type="usage",
                        prompt_tokens=200,
                        cached_tokens=150,
                        total_tokens=250,
                    ),
                ],
            ]
        )
        controller = self._controller(chat)
        item = WorkItem("What is 2 + 3 * 4?")
        redraws: list[str] = []
        controller.request_redraw = lambda: redraws.append(
            controller.ui_state.history_text()
        )

        asyncio.run(controller._stream_response(item))

        self.assertEqual(len(chat.requests), 2)
        second_messages = chat.requests[1][0]
        self.assertEqual(second_messages[-2].tool_calls[0].id, "call-1")
        self.assertEqual(second_messages[-1].role, "tool")
        self.assertEqual(second_messages[-1].tool_call_id, "call-1")
        self.assertEqual(second_messages[-1].content, "14")
        self.assertTrue(chat.requests[0][1])
        history = controller.ui_state.history_text()
        self.assertIn(
            '  Tool call [calculator]: {"expression":"2 + 3 * 4"}',
            history,
        )
        self.assertIn("  Tool result [calculator]: 14", history)
        self.assertIn("  Assistant note: I'll calculate. ", history)
        self.assertIn("Assistant: The answer is 14.", history)
        self.assertNotIn("60% reused", history)
        self.assertIn(
            "Total tokens: 250 | Prompt cache: 75% reused",
            history,
        )
        self.assertTrue(
            any(
                snapshot.endswith(
                    "Total tokens: 125 | Prompt cache: 60% reused"
                )
                for snapshot in redraws
            )
        )
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
            controller.ui_state.history_text()
        )

        asyncio.run(
            controller._stream_response(WorkItem("run both"))
        )

        tool_messages = [
            message for message in chat.requests[1][0] if message.role == "tool"
        ]
        self.assertEqual([message.content for message in tool_messages[:1]], ["3.0"])
        self.assertIn("error: unknown tool: missing", tool_messages[1].content or "")
        history = controller.ui_state.history_text()
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
            and "Tool error [missing]" not in snapshot
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
        self.assertLess(call_one, call_two)
        self.assertLess(call_two, result_one)
        self.assertLess(call_two, result_two)
        self.assertLess(result_one, result_two)

    def test_malformed_arguments_become_failed_tool_result(self) -> None:
        controller = self._controller(ScriptedToolChat([]))

        result, failed = controller._execute_tool_call(
            ToolCall("call-1", "calculator", "{not json")
        )

        self.assertTrue(failed)
        self.assertTrue(result.startswith("error:"))

    def test_preview_is_sent_to_model_rendered_in_ui_and_persisted(self) -> None:
        full_result = (
            "HEAD"
            + ("a" * 500)
            + "SECRET_MIDDLE"
            + ("b" * 500)
            + "TAIL"
        )
        tool = Tool(
            name="large_result",
            function=lambda _arguments: full_result,
            description="Return a large result.",
            prompt="Return a large result.",
            input_schema={"type": "object"},
        )
        chat = ScriptedToolChat(
            [
                [
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=0,
                        call_id="call-large",
                        name="large_result",
                        arguments="{}",
                    )
                ],
                [ChatStreamEvent(type="text_delta", text="Done.")],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )
            controller = self._controller(
                chat,
                tool_registry=ToolRegistry([tool]),
                tool_result_manager=manager,
            )

            asyncio.run(controller._stream_response(WorkItem("run it")))

            tool_message = chat.requests[1][0][-1]
            self.assertIsInstance(tool_message, ToolMessage)
            self.assertIn("[alpha-forge tool-result-preview]", tool_message.content)
            self.assertNotIn("SECRET_MIDDLE", tool_message.content)
            assert tool_message.preview is not None
            self.assertEqual(
                tool_message.preview.persisted_path.read_text(),
                full_result,
            )
            self.assertIn(
                str(tool_message.preview.persisted_path),
                tool_message.content,
            )
            history = controller.ui_state.history_text()
            self.assertIn("Tool result preview [large_result]", history)
            self.assertIn("[alpha-forge tool-result-preview]", history)
            self.assertNotIn("SECRET_MIDDLE", history)

    def test_persistence_failure_avoids_incomplete_assistant_message(self) -> None:
        full_result = "x" * 1_000
        tool = Tool(
            name="large_result",
            function=lambda _arguments: full_result,
            description="Return a large result.",
            prompt="Return a large result.",
            input_schema={"type": "object"},
        )
        chat = ScriptedToolChat(
            [
                [
                    ChatStreamEvent(
                        type="tool_call_delta",
                        index=0,
                        call_id="call-large",
                        name="large_result",
                        arguments="{}",
                    )
                ],
                [ChatStreamEvent(type="text_delta", text="unreachable")],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / "blocked"
            blocked.write_text("not a directory")
            manager = ToolResultManager(
                persist_directory=blocked,
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )
            controller = self._controller(
                chat,
                tool_registry=ToolRegistry([tool]),
                tool_result_manager=manager,
            )

            asyncio.run(controller._stream_response(WorkItem("run it")))

        self.assertEqual(len(chat.requests), 1)
        self.assertFalse(
            any(
                isinstance(message, AssistantMessage)
                for message in controller.conversation.messages
            )
        )
        self.assertIn("request failed: cannot persist", controller.ui_state.history_text())

    def test_clear_rotates_tool_result_session(self) -> None:
        manager = ToolResultManager(session_id="before")
        controller = self._controller(
            ScriptedToolChat([]),
            tool_result_manager=manager,
        )

        controller.submit("/clear")

        self.assertNotEqual(manager.session_id, "before")
        self.assertIn("conversation cleared", controller.ui_state.history_text())

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
            controller.ui_state.history_text(),
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
                        result=ToolMessage("4", tool_call_id="call-1"),
                    ),
                ),
                tool_requesting=True,
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    cached_tokens=72,
                    total_tokens=120,
                ),
            ),
        )
        state.set_iteration(
            turn,
            1,
            IterationOutput(
                assistant_text="The answer is 4.",
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    cached_tokens=80,
                    total_tokens=130,
                ),
            ),
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
                "token_usage",
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
                    "Total tokens: 130 | Prompt cache: 80% reused",
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

    def test_token_usage_formats_zero_reuse_without_raw_cache_count(self) -> None:
        state = ChatUiState()
        turn = state.start_turn("hello")
        state.set_iteration(
            turn,
            0,
            IterationOutput(
                assistant_text="Hi.",
                token_usage=TokenUsage(
                    prompt_tokens=48,
                    cached_tokens=0,
                    total_tokens=60,
                ),
            ),
        )

        history = state.history_text()
        self.assertIn(
            "Total tokens: 60 | Prompt cache: no reuse yet",
            history,
        )
        self.assertNotIn("48", history)

    def test_only_latest_turns_latest_iteration_shows_token_usage(self) -> None:
        state = ChatUiState()
        first_turn = state.start_turn("first")
        state.set_iteration(
            first_turn,
            0,
            IterationOutput(
                token_usage=TokenUsage(100, 25, 120),
            ),
        )
        second_turn = state.start_turn("second")
        state.set_iteration(
            second_turn,
            0,
            IterationOutput(
                token_usage=TokenUsage(100, 50, 130),
            ),
        )
        state.set_iteration(
            second_turn,
            1,
            IterationOutput(
                token_usage=TokenUsage(100, 75, 140),
            ),
        )

        history = state.history_text()
        self.assertNotIn("25% reused", history)
        self.assertNotIn("50% reused", history)
        self.assertEqual(history.count("Total tokens:"), 1)
        self.assertIn(
            "Total tokens: 140 | Prompt cache: 75% reused",
            history,
        )

    def test_token_usage_shows_total_when_cache_details_are_missing(self) -> None:
        state = ChatUiState()
        turn = state.start_turn("hello")
        state.set_iteration(
            turn,
            0,
            IterationOutput(
                token_usage=TokenUsage(
                    prompt_tokens=40,
                    total_tokens=55,
                ),
            ),
        )

        self.assertTrue(state.history_text().endswith("Total tokens: 55"))
