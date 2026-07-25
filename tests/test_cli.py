import base64
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from alpha_forge.cli import build_parser, main
from alpha_forge.config import Config
from alpha_forge.repl_controller import ChatReplController
from alpha_forge.slash_commands import SlashCommandCompleter
from alpha_forge.streaming import (
    ModelResponse,
    StreamCompleted,
    TextDelta,
    TokenUsage,
)
from alpha_forge.system_events import (
    ModelResponseStarted,
    TranscriptUpdated,
)
from alpha_forge.terminal_ui import TerminalChatUi
from alpha_forge.transcript import CommandMessage


@contextlib.contextmanager
def chdir(path: str | os.PathLike[str]):
    """Temporarily change the working directory."""
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


def _no_dotenv():
    """Disable load_dotenv so tests don't accidentally pick up a stray
    .env file from the test runner's environment."""
    return patch("alpha_forge.config.load_dotenv", lambda: None)


def _capture_run_repl():
    """Patch run_repl so tests can inspect the Config main() built."""
    captured: dict = {}

    def fake_run_repl(config: Config, **kwargs) -> int:
        captured["config"] = config
        return 0

    return patch("alpha_forge.cli.run_repl", fake_run_repl), captured


class CliTests(unittest.TestCase):
    def _controller(self, chat) -> ChatReplController:  # type: ignore[no-untyped-def]
        return ChatReplController(Config(api_key="sk-test"), chat=chat)

    @staticmethod
    def _record_notice(controller: ChatReplController, text: str) -> None:
        command = controller.session.add_command(
            raw="/test-notice",
            name="/test-notice",
            arguments="",
        )
        controller.session.add_command_result(
            command.command_id,
            status="success",
            messages=(CommandMessage(text),),
        )
        controller.ui_state.handle(TranscriptUpdated(controller.session.head_turn_id))

    def test_prompt_submits_on_enter(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui.input_area.text = "hello"

            self.assertFalse(ui._has_pending_prompts())
            keep_text = ui.input_area.buffer.accept_handler(ui.input_area.buffer)

        self.assertFalse(keep_text)
        self.assertFalse(ui.input_area.buffer.multiline())
        completions = list(
            ui.input_area.buffer.completer.get_completions(
                Document("/m"),
                CompleteEvent(completion_requested=True),
            )
        )
        self.assertEqual(completions, [])
        self.assertTrue(ui._has_pending_prompts())
        self.assertEqual(controller.ui_state.pending_prompts, ["hello"])

    def test_slash_suggestions_render_in_prompt_subpanel(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )

            self.assertFalse(ui._should_show_slash_suggestions())
            ui.input_area.text = "/m"

            self.assertTrue(ui._should_show_slash_suggestions())
            self.assertIn("/model", ui.slash_suggestions_area.text)
            self.assertIn("List available models", ui.slash_suggestions_area.text)

    def test_mouse_support_defaults_to_history_scroll_mode(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )

            self.assertTrue(ui._mouse_enabled)
            self.assertIn("mouse-scroll", ui._status_text())
            self.assertIn("F2 toggle", ui._status_text())

    def test_mouse_mode_can_be_toggled_for_terminal_selection(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui._toggle_mouse_mode()

            self.assertFalse(ui._mouse_enabled)
            self.assertIn("copy-select", ui._status_text())
            self.assertIn("terminal selection enabled", ui._status_text())

    def test_copy_history_writes_osc52_clipboard_sequence(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        self._record_notice(controller, "copy me")
        output = DummyOutput()
        writes: list[str] = []
        flushes: list[bool] = []

        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=output,
            )
            output.write_raw = writes.append  # type: ignore[method-assign]
            output.flush = lambda: flushes.append(True)  # type: ignore[method-assign]
            ui._copy_history_to_terminal_clipboard()

        expected_payload = base64.b64encode(b"Notice: copy me").decode()
        self.assertEqual(writes, [f"\x1b]52;c;{expected_payload}\a"])
        self.assertEqual(flushes, [True])
        self.assertIn("transcript copied", ui._status_text())

    def test_ephemeral_response_follows_durable_content_in_history_pane(
        self,
    ) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        turn_id = controller.session.submit_user("hello")
        controller.ui_state.handle(ModelResponseStarted(turn_id))
        controller.ui_state.handle(TextDelta("partial response"))
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )

            copied = ui._history_plain_text()
            rendered = ui.history_control.create_content(80, 3)

        self.assertIn("You: hello", copied)
        self.assertNotIn("partial response", copied)
        self.assertEqual(
            rendered.get_line(rendered.line_count - 1),
            [("class:assistant-message", "Assistant: partial response")],
        )
        durable_index = next(
            index
            for index in range(rendered.line_count)
            if rendered.get_line(index) == [("class:user-message", "You: hello")]
        )
        self.assertLess(durable_index, rendered.line_count - 1)

    def test_history_control_exposes_complete_scrollable_content(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        self._record_notice(
            controller,
            "\n".join(f"line {index}" for index in range(30)),
        )

        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            content = ui.history_control.create_content(80, 5)

        self.assertEqual(content.line_count, 30)
        self.assertEqual(ui._history_total_lines, 30)
        self.assertEqual(ui._history_view_height, 5)
        self.assertEqual(ui._history_scroll_offset, 25)
        self.assertEqual(content.cursor_position.y, 25)
        self.assertEqual(
            content.get_line(0),
            [("class:notice-message", "Notice: line 0")],
        )
        self.assertEqual(
            content.get_line(29),
            [("class:notice-message", "        line 29")],
        )

    def test_mouse_wheel_scroll_routes_to_history_viewport(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui._history_total_lines = 100
            ui._history_view_height = 10
            ui._history_scroll_offset = 10
            ui._history_follow_tail = False

            ui._scroll_history_lines(-3)
            self.assertEqual(ui._history_scroll_offset, 7)

            ui._scroll_history_lines(3)
            self.assertEqual(ui._history_scroll_offset, 10)

    def test_mouse_wheel_over_prompt_scrolls_history(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui._history_total_lines = 100
            ui._history_view_height = 10
            ui._history_scroll_offset = 10
            ui._history_follow_tail = False
            event = MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.SCROLL_UP,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            )

            ui.input_area.control.mouse_handler(event)

        self.assertEqual(ui._history_scroll_offset, 7)

    def test_mouse_wheel_over_history_scrolls_history(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui._history_total_lines = 100
            ui._history_view_height = 10
            ui._history_scroll_offset = 10
            ui._history_follow_tail = False
            event = MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.SCROLL_DOWN,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            )

            ui.history_control.mouse_handler(event)

        self.assertEqual(ui._history_scroll_offset, 13)

    def test_history_scroll_normalizes_tail_sentinel_before_wheel_delta(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui._history_total_lines = 110
            ui._history_view_height = 10
            ui._history_follow_tail = True

            ui._scroll_history_lines(-3)

        self.assertEqual(ui._history_scroll_offset, 97)

    def test_slash_suggestions_are_below_prompt_input(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            root = ui._root_container()

            self.assertIs(root.children[-2], ui.input_area.window)

    def test_enter_autocompletes_partial_slash_command(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui.input_area.text = "/m"

            keep_text = ui.input_area.buffer.accept_handler(ui.input_area.buffer)

        self.assertTrue(keep_text)
        self.assertEqual(ui.input_area.text, "/model")
        self.assertEqual(controller.ui_state.pending_prompts, [])

    def test_tab_uses_slash_autocomplete(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui.input_area.text = "/m"

            completed = ui._complete_slash_command()

        self.assertTrue(completed)
        self.assertEqual(ui.input_area.text, "/model")

    def test_history_refresh_preserves_scrolled_position(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            self._record_notice(
                controller,
                "\n".join(f"line {index}" for index in range(20)),
            )
            ui.refresh()
            ui._history_follow_tail = False
            ui._history_scroll_offset = 1

            ui.refresh()
            self.assertEqual(ui._history_scroll_offset, 1)

            self._record_notice(controller, "new tail")
            ui.refresh()

        self.assertEqual(ui._history_scroll_offset, 1)

    def test_pageup_scrolls_history_without_focusing_history(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            self._record_notice(
                controller,
                "\n".join(f"line {index}" for index in range(20)),
            )
            ui.refresh()
            ui._history_total_lines = 200
            ui._history_view_height = 20
            ui._history_scroll_offset = 100
            ui._history_follow_tail = False
            starting_scroll = ui._history_scroll_offset

            ui._scroll_history_page(-1)

        self.assertLess(ui._history_scroll_offset, starting_scroll)

    def test_slash_suggestions_hide_after_command_argument(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            ui.input_area.text = "/model extra"

            self.assertFalse(ui._should_show_slash_suggestions())

    def test_user_can_type_while_consumer_streams(self) -> None:
        """A second user message can be accepted while the first response
        is still streaming."""

        import asyncio

        class FakeChatClient:
            def __init__(self) -> None:
                self.responses: dict[str, str] = {}
                self.started = asyncio.Event()

            async def stream_response(self, messages, *, tools):  # type: ignore[no-untyped-def]
                last_user = next(
                    message.content
                    for message in reversed(messages)
                    if message.role == "user"
                )
                self.responses[last_user] = f"reply-to:{last_user}"
                self.started.set()
                for token in list(self.responses[last_user]):
                    await asyncio.sleep(0)
                    yield TextDelta(token)
                yield StreamCompleted(
                    ModelResponse(self.responses[last_user], finish_reason="stop")
                )

            def list_models(self) -> list[str]:
                return []

        async def scenario() -> ChatReplController:
            chat = FakeChatClient()
            controller = self._controller(chat)
            exit_codes: list[int] = []
            controller.request_app_exit = exit_codes.append

            consumer_task = asyncio.create_task(controller.consume())
            controller.submit("hello")
            await chat.started.wait()
            controller.submit("second")
            controller.request_exit()
            await consumer_task

            self.assertEqual(exit_codes, [0])
            return controller

        controller = asyncio.run(scenario())

        self.assertEqual(controller.ui_state.pending_prompts, [])
        self.assertIn("reply-to:hello", controller.ui_state.history_text())
        self.assertIn("reply-to:second", controller.ui_state.history_text())
        self.assertIn("You: hello", controller.ui_state.history_text())
        self.assertNotIn("Alpha:", controller.ui_state.history_text())

    def test_user_messages_use_distinct_history_style(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        turn_id = controller.session.submit_user("hello")
        controller.session.add_assistant_message(
            turn_id=turn_id,
            response=ModelResponse("reply"),
        )

        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            fragments = ui._history_fragments()

        self.assertEqual(fragments[0], ("class:user-message", "You: hello"))
        self.assertEqual(
            fragments[2],
            ("class:assistant-message", "Assistant: reply"),
        )
        assistant_attrs = TerminalChatUi._style().get_attrs_for_style_str(
            "class:assistant-message"
        )
        self.assertEqual(assistant_attrs.color, "")

    def test_assistant_notes_use_distinct_italic_cyan_style(self) -> None:
        self.assertEqual(
            TerminalChatUi._history_line_style("assistant_note"),
            "class:assistant-note-message",
        )
        style = TerminalChatUi._style()
        attrs = style.get_attrs_for_style_str("class:assistant-note-message")
        self.assertTrue(attrs.italic)
        self.assertEqual(attrs.color, "5fd7ff")

    def test_token_usage_uses_subdued_italic_style(self) -> None:
        self.assertEqual(
            TerminalChatUi._history_line_style("token_usage"),
            "class:token-usage-message",
        )
        attrs = TerminalChatUi._style().get_attrs_for_style_str(
            "class:token-usage-message"
        )
        self.assertTrue(attrs.italic)
        self.assertEqual(attrs.color, "87af87")

    def test_token_usage_is_right_aligned_to_history_width(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        controller = self._controller(FakeChatClient())
        turn_id = controller.session.submit_user("hello")
        controller.session.add_assistant_message(
            turn_id=turn_id,
            response=ModelResponse(
                "reply",
                usage=TokenUsage(100, 75, 125),
            ),
        )

        with create_pipe_input() as pipe_input:
            ui = TerminalChatUi(
                controller,
                input=pipe_input,
                output=DummyOutput(),
            )
            cache_line = ui._history_line_fragments(60, None)[-1][0][1]

        self.assertEqual(len(cache_line), 60)
        self.assertTrue(
            cache_line.endswith("Total tokens: 125 | Prompt cache: 75% reused")
        )

    def test_failed_response_is_rendered_in_history(self) -> None:
        import asyncio

        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return []

        async def scenario() -> ChatReplController:
            controller = self._controller(FakeChatClient())
            consumer_task = asyncio.create_task(controller.consume())
            controller.submit("hello")
            controller.request_exit()
            await consumer_task
            return controller

        controller = asyncio.run(scenario())

        self.assertEqual(controller.ui_state.pending_prompts, [])
        self.assertIn("request failed: boom", controller.ui_state.history_text())

    def test_parser_does_not_expose_system_option(self) -> None:
        help_text = build_parser().format_help()

        self.assertNotIn("--system", help_text)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--system", "custom"])

    def test_parser_exposes_init_config(self) -> None:
        # New flag for bootstrapping the user config file.
        self.assertIn("--init-config", build_parser().format_help())

    def test_slash_completer_matches_after_slash_character(self) -> None:
        completer = SlashCommandCompleter(["/clear", "/model"])

        completions = list(completer.get_completions(Document("/m"), None))

        self.assertEqual([completion.text for completion in completions], ["/model"])

    def test_slash_completer_ignores_normal_text(self) -> None:
        completer = SlashCommandCompleter(["/clear", "/model"])

        completions = list(completer.get_completions(Document("hello"), None))

        self.assertEqual(completions, [])

    def test_model_command_lists_models_and_marks_current_model(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return ["gpt-other", "gpt-test"]

        controller = ChatReplController(
            Config(api_key="sk-test", model="gpt-test"),
            chat=FakeChatClient(),
        )
        controller.submit("/model")
        output = controller.ui_state.history_text()

        self.assertIn("  gpt-other", output)
        self.assertIn("* gpt-test", output)

    def test_model_command_marks_provider_prefixed_current_model(self) -> None:
        class FakeChatClient:
            async def stream_response(self, _messages, *, tools):  # type: ignore[no-untyped-def]
                if False:
                    yield TextDelta("")

            def list_models(self) -> list[str]:
                return ["openai/gpt-4o"]

        controller = ChatReplController(
            Config(api_key="sk-test", model="gpt-4o"),
            chat=FakeChatClient(),
        )
        controller.submit("/model")
        output = controller.ui_state.history_text()

        self.assertIn("* openai/gpt-4o", output)

    # --- Layered-merge end-to-end tests (drive main() and inspect Config) ---

    def _write_user_config(self, xdg_root: Path, body: str) -> Path:
        cfg_dir = xdg_root / "alpha-forge"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / "config.toml"
        path.write_text(body)
        return path

    def test_cli_flag_overrides_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            self._write_user_config(
                xdg,
                '[openai]\napi_key = "file-key"\nmodel = "file-model"\n',
            )
            env = {"XDG_CONFIG_HOME": str(xdg), "OPENAI_API_KEY": "env-key"}
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main(["--model", "cli-model"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].api_key, "file-key")
            self.assertEqual(captured["config"].model, "cli-model")

    def test_user_config_overrides_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            self._write_user_config(
                xdg,
                '[openai]\napi_key = "file-key"\nmodel = "file-model"\n',
            )
            env = {
                "XDG_CONFIG_HOME": str(xdg),
                "OPENAI_API_KEY": "env-key",
                "OPENAI_MODEL": "env-model",
            }
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main([])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].api_key, "file-key")
            self.assertEqual(captured["config"].model, "file-model")

    def test_env_overrides_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            env = {
                "XDG_CONFIG_HOME": str(xdg),
                "OPENAI_API_KEY": "env-key",
                "OPENAI_MODEL": "env-model",
            }
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main([])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].api_key, "env-key")
            self.assertEqual(captured["config"].model, "env-model")

    def test_base_url_cli_flag_overrides_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            self._write_user_config(
                xdg,
                "[openai]\n"
                'api_key = "file-key"\n'
                'base_url = "https://file.example/v1"\n',
            )
            env = {"XDG_CONFIG_HOME": str(xdg)}
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main(["--base-url", "https://cli.example/v1"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].base_url, "https://cli.example/v1")

    def test_dotenv_still_loaded(self) -> None:
        # Regression guard: load_dotenv() in cli.main must keep working
        # so existing .env-based workflows are preserved.
        # We mock load_dotenv to inject the env var the way a real
        # .env file would, then verify it flows into load_env_config.
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()

            def fake_load_dotenv() -> None:
                os.environ["OPENAI_API_KEY"] = "from-dotenv"

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with patch("alpha_forge.config.load_dotenv", fake_load_dotenv):
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["config"].api_key, "from-dotenv")

    def test_init_config_writes_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main(["--init-config"])

            target = xdg / "alpha-forge" / "config.toml"
            self.assertEqual(exit_code, 0)
            self.assertTrue(target.exists())
            contents = target.read_text()
            self.assertIn("[openai]", contents)
            self.assertIn("api_key", contents)
            self.assertIn("# model", contents)
            self.assertIn("# base_url", contents)
            self.assertIn("# timeout", contents)

    def test_init_config_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            target = xdg / "alpha-forge" / "config.toml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("preexisting = 'do-not-overwrite'\n")

            stderr = io.StringIO()
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with contextlib.redirect_stderr(stderr):
                    exit_code = main(["--init-config"])

            self.assertEqual(exit_code, 1)
            self.assertEqual(target.read_text(), "preexisting = 'do-not-overwrite'\n")
            self.assertIn("already exists", stderr.getvalue())

    def test_missing_api_key_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            stderr = io.StringIO()
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with _no_dotenv():
                    with contextlib.redirect_stderr(stderr):
                        exit_code = main([])

            self.assertEqual(exit_code, 2)
            message = stderr.getvalue()
            self.assertIn("api_key", message)
            # The error must point at the XDG config file path so the
            # user can find it without reading the source.
            self.assertIn("config.toml", message)
