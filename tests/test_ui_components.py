"""Characterize terminal behavior across the component refactor."""

import asyncio
import base64
import unittest
from unittest.mock import Mock

from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.application.events import (
    InputQueued,
    InputStarted,
    PersistenceFailed,
    RequestFailed,
    ResponseStreamStarted,
    ResponseStreamUpdated,
    SessionView,
    SessionViewChanged,
    ToolPermissionRequested,
    ToolPermissionResolved,
)
from alpha_forge.config import Config
from alpha_forge.hooks import PreToolExecution
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.projectors.ui_history import UiModelOutput, UiPrompt, UiToolResult
from alpha_forge.providers import TextDelta, TokenUsage, ToolCall
from alpha_forge.sessions import Session
from alpha_forge.transcript import InputAccepted
from alpha_forge.ui.terminal import TerminalChatUi


class TerminalBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.session = Session.create(in_memory=True)
        self.coordinator = ApplicationCoordinator(
            Config("key"),
            provider=Mock(),
            session=self.session,
        )
        self.pipe = create_pipe_input()
        self.input = self.pipe.__enter__()
        self.output = DummyOutput()
        self.ui = TerminalChatUi(self.coordinator, input=self.input, output=self.output)
        self.addCleanup(self.pipe.__exit__, None, None, None)
        self.addCleanup(self.session.close)
        self.addCleanup(self.ui._event_subscription.unsubscribe)

    def publish_history(self, text, session_id="session"):
        self.coordinator.event_router.publish(
            SessionViewChanged(
                SessionView(session_id, 2, (UiPrompt(1, "prompt", text),)),
                True,
            )
        )

    def test_history_wraps_and_preserves_role_styles(self):
        self.publish_history("abcdefghij")
        content = self.ui.history_area.control.create_content(6, 3)
        self.assertEqual(
            [content.get_line(i) for i in range(content.line_count)],
            [
                [("class:user-message", "You: a")],
                [("class:user-message", "bcdefg")],
                [("class:user-message", "hij")],
            ],
        )

    def test_history_scroll_follow_tail_and_resize(self):
        self.publish_history("\n".join(str(i) for i in range(12)))
        self.ui.history_area.control.create_content(80, 4)
        self.assertEqual(
            self.ui.history_area.control.vertical_scroll(
                self.ui.history_area.control.window
            ),
            8,
        )
        self.ui.history_area.control.scroll_page(-1)
        self.assertEqual(
            self.ui.history_area.control.vertical_scroll(
                self.ui.history_area.control.window
            ),
            5,
        )
        self.publish_history("\n".join(str(i) for i in range(15)))
        self.ui.history_area.control.create_content(80, 4)
        self.assertEqual(
            self.ui.history_area.control.vertical_scroll(
                self.ui.history_area.control.window
            ),
            5,
        )
        self.ui.history_area.control.scroll_lines(100)
        self.assertEqual(
            self.ui.history_area.control.vertical_scroll(
                self.ui.history_area.control.window
            ),
            11,
        )
        self.ui.history_area.control.create_content(80, 6)
        self.assertEqual(
            self.ui.history_area.control.vertical_scroll(
                self.ui.history_area.control.window
            ),
            9,
        )

    def test_equal_revision_session_switch_invalidates_history(self):
        self.publish_history("first", "one")
        self.assertEqual(self.ui.history_area.control.transcript_text(), "You: first")
        self.publish_history("second", "two")
        self.assertEqual(self.ui.history_area.control.transcript_text(), "You: second")

    def test_clipboard_contains_only_committed_text(self):
        self.publish_history("hello 世界")
        self.coordinator.event_router.publish(
            ResponseStreamStarted("prompt", "request")
        )
        self.coordinator.event_router.publish(
            ResponseStreamUpdated("request", TextDelta("draft"))
        )
        self.output.write_raw = Mock()
        self.output.flush = Mock()
        self.ui._copy_history_to_terminal_clipboard()
        encoded = base64.b64encode("You: hello 世界".encode()).decode()
        self.output.write_raw.assert_called_once_with(f"\x1b]52;c;{encoded}\a")
        self.output.flush.assert_called_once_with()
        self.assertEqual(self.ui.bottom_area.status_message, "transcript copied")

    def test_enter_completes_then_submits_and_unknown_command_is_submitted(self):
        self.ui.bottom_area.input_panel.editor.text = "/he"
        self.assertTrue(self.ui.bottom_area.input_panel.accept_input(None))
        self.assertEqual(self.ui.bottom_area.input_panel.editor.text, "/help")
        self.assertFalse(self.ui.bottom_area.input_panel.accept_input(None))
        self.assertEqual(
            self.ui.bottom_area.queued_inputs.pending_area.text, "1. /help"
        )
        self.ui.bottom_area.input_panel.editor.text = "/missing"
        self.assertEqual(
            self.ui.bottom_area.input_panel.suggestions_area.text,
            "No matching commands.",
        )
        self.assertFalse(self.ui.bottom_area.input_panel.accept_input(None))
        self.assertEqual(
            self.ui.bottom_area.queued_inputs.pending_area.text, "1. /help\n2. /missing"
        )

    def test_persistence_failure_keeps_stream_preview_and_halts_status(self):
        self.coordinator.event_router.publish(
            ResponseStreamStarted("prompt", "request")
        )
        self.coordinator.event_router.publish(
            ResponseStreamUpdated("request", TextDelta("draft"))
        )
        self.coordinator.event_router.publish(PersistenceFailed("query", "disk full"))
        self.assertEqual(
            self.ui.history_area.control.state.active_text(),
            "Assistant: draft\nError: not persisted: disk full",
        )
        self.assertIn(
            "Cannot persist query: disk full", self.ui.bottom_area.status_message
        )
        self.coordinator.request_exit()
        self.assertIn(
            "Persistence failed; input processing stopped",
            self.ui.bottom_area.status_message,
        )

    def test_mouse_support_and_shutdown_shortcuts(self):
        self.assertTrue(self.ui.app.mouse_support())
        self.assertNotIn("F2", self.ui.bottom_area.status_message)
        bindings = self.ui.app.key_bindings
        ctrl_d = bindings.get_bindings_for_keys(("c-d",))[0]
        self.ui.bottom_area.input_panel.editor.text = "draft"
        ctrl_d.handler(None)
        self.assertTrue(self.coordinator.accepting)
        self.ui.bottom_area.input_panel.editor.text = ""
        ctrl_d.handler(None)
        self.assertFalse(self.coordinator.accepting)
        asyncio.run(self.coordinator.consume())

    def test_queue_updates_are_reactive_without_becoming_history(self) -> None:
        self.coordinator.event_router.publish(InputQueued("one", "hello"))
        self.assertEqual(self.ui.bottom_area.queued_inputs.pending_inputs, ["hello"])
        self.assertEqual(self.ui.bottom_area.status_message, "1 input queued")
        self.assertEqual(
            self.ui.history_area.control.state.transcript_text(), "No messages yet."
        )

        self.coordinator.event_router.publish(InputStarted("one"))
        self.assertEqual(self.ui.bottom_area.queued_inputs.pending_inputs, [])
        self.assertEqual(self.ui.bottom_area.status_message, "Ready")

    def test_permission_request_and_resolution_are_ephemeral(self) -> None:
        request = ToolPermissionRequested(
            request_id="request",
            call_id="call",
            tool_name="bash",
            tool_input=FrozenJsonObject({"cmd": "pwd"}),
        )

        self.coordinator.event_router.publish(request)

        self.assertEqual(
            self.ui.bottom_area.permission_panel.pending_request.request_id, "request"
        )
        self.assertEqual(self.ui.bottom_area.status_message, "Approval required: bash")
        self.assertEqual(
            self.ui.history_area.control.state.transcript_text(), "No messages yet."
        )

        self.coordinator.event_router.publish(ToolPermissionResolved("request", False))
        self.assertIsNone(self.ui.bottom_area.permission_panel.pending_request)
        self.assertEqual(self.ui.bottom_area.status_message, "Denying tool")

    def test_request_failure_clears_ephemeral_draft(self) -> None:
        self.coordinator.event_router.publish(
            ResponseStreamStarted("prompt", "request")
        )
        self.coordinator.event_router.publish(
            ResponseStreamUpdated("request", TextDelta("partial"))
        )
        self.coordinator.event_router.publish(RequestFailed("boom"))
        self.assertEqual(self.ui.history_area.control.state.active_text(), "")
        self.assertEqual(self.ui.bottom_area.status_message, "Request failed: boom")

    def test_tool_preview_preserves_tail_and_usage_is_right_aligned(self):
        call = ToolCall("call", "echo", "{}")
        self.coordinator.event_router.publish(
            SessionViewChanged(
                SessionView(
                    "session",
                    5,
                    (
                        UiPrompt(1, "prompt", "question"),
                        UiModelOutput(
                            2, "tools", "prompt", None, None, None, (call,), None
                        ),
                        UiToolResult(
                            3,
                            "result",
                            "tools",
                            "call",
                            "\n".join(f"line-{i}" for i in range(25)),
                            "success",
                            True,
                        ),
                        UiModelOutput(
                            4,
                            "answer",
                            "prompt",
                            "done",
                            None,
                            None,
                            (),
                            TokenUsage(total_tokens=12),
                        ),
                    ),
                )
            )
        )
        lines = self.ui.history_area.control.state.transcript_lines()
        results = [line for line in lines if line.role == "tool_result"]
        self.assertEqual(len(results), 20)
        self.assertEqual(
            results[0].text,
            "  Tool result preview (excluded from model context) [echo]: line-5",
        )
        self.assertTrue(results[-1].text.endswith("line-24"))
        content = self.ui.history_area.control.create_content(100, 30)
        self.assertEqual(
            content.get_line(content.line_count - 1),
            [("class:token-usage-message", "Total tokens: 12".rjust(100))],
        )

    def test_positioned_mouse_events_stay_in_their_components(self):
        async def run():
            editor = self.ui.bottom_area.input_panel.editor
            suggestions = self.ui.bottom_area.input_panel.suggestions_area
            history = self.ui.history_area.control
            self.output.get_size = lambda: Size(rows=16, columns=40)
            self.publish_history("\n".join(str(i) for i in range(100)))
            editor.text = "/"

            def render():
                self.ui.app.layout.update_parents_relations()
                self.ui.app.renderer.render(self.ui.app, self.ui.app.layout)

            async def mouse(window, button, suffix="M", x_offset=0):
                render()
                position = self.ui.app.renderer._last_screen.visible_windows_to_write_positions[
                    window
                ]
                data = (
                    f"\x1b[<{button};{position.xpos + x_offset + 1};"
                    f"{position.ypos + 1}{suffix}"
                )
                self.ui.app.key_processor.feed(KeyPress(Keys.Vt100MouseEvent, data))
                self.ui.app.key_processor.process_keys()
                await asyncio.sleep(0)
                render()

            try:
                with set_app(self.ui.app):
                    render()
                    scroll_before = history.vertical_scroll(history.window)
                    cursor_before = editor.buffer.cursor_position
                    for button in (64, 65):
                        await mouse(editor.window, button)
                        self.assertEqual(editor.text, "/")
                        self.assertEqual(editor.buffer.cursor_position, cursor_before)
                        self.assertEqual(
                            history.vertical_scroll(history.window), scroll_before
                        )

                    # Wheel routing follows the pointer while the editor keeps focus.
                    await mouse(history.window, 64)
                    self.assertEqual(
                        history.vertical_scroll(history.window), scroll_before - 3
                    )
                    await mouse(history.window, 65)
                    self.assertEqual(
                        history.vertical_scroll(history.window), scroll_before
                    )
                    self.assertIs(self.ui.app.layout.current_control, editor.control)

                    self.assertEqual(suggestions.window.vertical_scroll, 0)
                    await mouse(suggestions.window, 65)
                    self.assertGreater(suggestions.window.vertical_scroll, 0)
                    await mouse(suggestions.window, 64)
                    self.assertEqual(suggestions.window.vertical_scroll, 0)
                    self.assertEqual(
                        history.vertical_scroll(history.window), scroll_before
                    )
                    self.assertEqual(editor.text, "/")
                    self.assertIs(self.ui.app.layout.current_control, editor.control)

                    # Non-wheel events still reach the editor's original handler.
                    await mouse(editor.window, 0, x_offset=len("alpha> "))
                    await mouse(editor.window, 0, suffix="m", x_offset=len("alpha> "))
                    self.assertEqual(editor.buffer.cursor_position, 0)
            finally:
                await self.ui.app.cancel_and_wait_for_background_tasks()

        asyncio.run(run())

    def test_positionless_wheel_does_not_recall_inputs_but_arrow_keys_do(self):
        async def run():
            editor = self.ui.bottom_area.input_panel.editor
            history = self.ui.history_area.control
            editor.buffer.history.append_string("previous input")
            editor.text = "draft"
            self.publish_history("\n".join(str(i) for i in range(100)))

            async def press(key):
                self.ui.app.key_processor.feed(KeyPress(key))
                self.ui.app.key_processor.process_keys()
                await asyncio.sleep(0)

            try:
                with set_app(self.ui.app):
                    self.ui.app.renderer.render(self.ui.app, self.ui.app.layout)
                    await asyncio.sleep(0)  # Allow Buffer to load input history.
                    scroll_before = history.vertical_scroll(history.window)
                    for key in (Keys.ScrollUp, Keys.ScrollDown):
                        await press(key)
                        self.assertEqual(editor.text, "draft")
                        self.assertEqual(
                            history.vertical_scroll(history.window), scroll_before
                        )
                    await press(Keys.Up)
                    self.assertEqual(editor.text, "previous input")
                    await press(Keys.Down)
                    self.assertEqual(editor.text, "draft")
            finally:
                await self.ui.app.cancel_and_wait_for_background_tasks()

        asyncio.run(run())

    def test_escape_denies_permission_and_restores_draft_focus(self):
        self.ui.bottom_area.input_panel.editor.text = "/he"
        lifecycle = PreToolExecution(
            call_id="call",
            tool_name="bash",
            tool_input=FrozenJsonObject({"cmd": "pwd"}),
        )

        async def run():
            pending = asyncio.create_task(
                self.coordinator.request_tool_permission(lifecycle)
            )
            await asyncio.sleep(0)
            self.assertFalse(self.ui.bottom_area._input_container.filter())
            escape = self.ui.app.key_bindings.get_bindings_for_keys(("escape",))[0]
            self.assertTrue(escape.filter())
            escape.handler(None)
            return await pending

        self.assertFalse(asyncio.run(run()))
        self.assertEqual(self.ui.bottom_area.input_panel.editor.text, "/he")
        self.assertTrue(self.ui.bottom_area.input_panel.show_suggestions())
        self.assertIs(
            self.ui.app.layout.current_control,
            self.ui.bottom_area.input_panel.editor.control,
        )

    def test_real_input_loop_completes_with_tab_and_exits(self):
        async def run():
            consumer = asyncio.create_task(self.coordinator.consume())
            self.ui.app.pre_run_callables.append(
                lambda: self.input.send_text("/he\t\r/exit\r")
            )
            try:
                return await asyncio.wait_for(self.ui.run_async(), timeout=3)
            finally:
                self.coordinator.request_exit()
                await asyncio.wait_for(consumer, timeout=3)

        self.assertEqual(asyncio.run(run()), 0)
        self.assertEqual(
            [
                e.text
                for e in self.session.transcript.events
                if isinstance(e, InputAccepted)
            ],
            ["/help", "/exit"],
        )

    def test_key_dispatch_preserves_shortcuts_across_focus_and_modal_dialog(self):
        async def press(key):
            with set_app(self.ui.app):
                self.ui.app.layout.update_parents_relations()
                self.ui.app.renderer.render(self.ui.app, self.ui.app.layout)
                self.ui.app.key_processor.feed(KeyPress(key))
                self.ui.app.key_processor.process_keys()
            await asyncio.sleep(0)

        async def run():
            try:
                for target in (
                    self.ui.history_area.control,
                    self.ui.bottom_area.input_panel.focus_target,
                    self.ui.bottom_area.input_panel.suggestions_area.control,
                ):
                    self.ui.bottom_area.input_panel.editor.text = "/he"
                    self.ui.app.layout.focus(target)
                    await press(Keys.ControlI)
                    self.assertEqual(
                        self.ui.bottom_area.input_panel.editor.text, "/help"
                    )
                    self.assertEqual(
                        self.ui.bottom_area.queued_inputs.pending_inputs, []
                    )

                request = ToolPermissionRequested(
                    request_id="request",
                    call_id="call",
                    tool_name="bash",
                    tool_input=FrozenJsonObject({"cmd": "pwd"}),
                )
                self.coordinator.event_router.publish(request)
                self.ui.bottom_area.input_panel.editor.text = "/he"
                self.publish_history("\n".join(str(i) for i in range(100)))
                self.ui.history_area.control.create_content(80, 4)
                await press(Keys.ControlI)
                self.assertEqual(self.ui.bottom_area.input_panel.editor.text, "/he")
                self.assertIs(
                    self.ui.app.layout.current_control,
                    self.ui.bottom_area.permission_panel.allow_button.control,
                )
                history = self.ui.history_area.control
                scroll_before = history.vertical_scroll(history.window)
                page_height = max(1, history.window.render_info.window_height - 1)
                self.assertGreater(scroll_before, 0)
                await press(Keys.PageUp)
                self.assertEqual(
                    history.vertical_scroll(history.window),
                    max(0, scroll_before - page_height),
                )
                for key in (Keys.ScrollUp, Keys.ScrollDown):
                    scroll_before = history.vertical_scroll(history.window)
                    await press(key)
                    self.assertEqual(
                        history.vertical_scroll(history.window), scroll_before
                    )
                    self.assertEqual(self.ui.bottom_area.input_panel.editor.text, "/he")
                await press(Keys.F2)
                self.assertTrue(self.ui.app.mouse_support())
                self.assertNotIn("copy-select", self.ui.bottom_area.status_message)
                await press(Keys.ControlC)
                self.assertFalse(self.coordinator.accepting)
            finally:
                await self.ui.app.cancel_and_wait_for_background_tasks()

        asyncio.run(run())
