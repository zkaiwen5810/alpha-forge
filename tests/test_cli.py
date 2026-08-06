import asyncio
import unittest

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.application.events import (
    InputQueued,
    InputStarted,
    ModelOutputRecorded,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    RequestFailed,
    SessionView,
    SessionViewChanged,
    ToolPermissionRequested,
    ToolPermissionResolved,
)
from alpha_forge.cli import build_parser
from alpha_forge.config import Config
from alpha_forge.hooks import PreToolExecution
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.providers import (
    OutputMessage,
    OutputText,
    ProviderOutput,
    TextDelta,
    TokenUsage,
    ToolCall,
)
from alpha_forge.projectors.ui_history import (
    UiModelOutput,
    UiPrompt,
    UiQueryFailure,
    UiToolResult,
)
from alpha_forge.sessions import Session
from alpha_forge.terminal_ui import MAX_PERMISSION_PREVIEW_CHARS, TerminalChatUi
from alpha_forge.tools import Tool, ToolExecutor, ToolRegistry
from alpha_forge.ui_state import ChatUiState


class CliSurfaceTests(unittest.TestCase):
    def test_api_keys_are_not_accepted_as_cli_arguments(self) -> None:
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--api-key", option_strings)
        self.assertIn("--base-url", option_strings)

    def test_cli_permissions_match_only_bash_and_file_writer(self) -> None:
        class Provider:
            def list_models(self):
                return ["gpt-test"]

        invoked: list[str] = []
        registry = ToolRegistry(
            [
                Tool(
                    name=name,
                    description=name,
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                    },
                    handler=lambda _arguments, name=name: invoked.append(name) or name,
                )
                for name in ("calculator", "file_writer", "bash")
            ]
        )
        controller = ApplicationCoordinator(
            Config("key"),
            provider=Provider(),
            session=Session.create(in_memory=True),
        )
        requests: list[ToolPermissionRequested] = []

        def deny(event: ToolPermissionRequested) -> None:
            requests.append(event)
            controller.resolve_tool_permission(event.request_id, False)

        controller.events.subscribe(ToolPermissionRequested, deny)
        executor = ToolExecutor(registry, controller.hooks)

        safe = asyncio.run(
            executor.execute(ToolCall("safe", "calculator", "{}"))
        )
        denied = asyncio.run(
            executor.execute(ToolCall("denied", "bash", "{}"))
        )

        self.assertEqual(safe.status, "success")
        self.assertEqual(denied.status, "error")
        self.assertEqual([event.event.tool_name for event in requests], ["bash"])
        self.assertEqual(invoked, ["calculator"])
        controller.session.close()


class UiStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ChatUiState(SessionView("session", 1, ()))

    def test_queue_updates_are_reactive_without_becoming_history(self) -> None:
        self.state.handle(InputQueued("one", "hello"))
        self.assertEqual(self.state.pending_inputs, ["hello"])
        self.assertEqual(self.state.status, "1 input queued")
        self.assertEqual(self.state.transcript_text(), "No messages yet.")

        self.state.handle(InputStarted("one"))
        self.assertEqual(self.state.pending_inputs, [])
        self.assertEqual(self.state.status, "Ready")

    def test_permission_request_and_resolution_are_ephemeral(self) -> None:
        lifecycle = PreToolExecution(
            call_id="call",
            tool_name="bash",
            tool_input=FrozenJsonObject({"cmd": "pwd"}),
        )

        self.state.handle(ToolPermissionRequested("request", lifecycle))

        self.assertEqual(self.state.pending_permission.request_id, "request")
        self.assertEqual(self.state.status, "Approval required: bash")
        self.assertEqual(self.state.transcript_text(), "No messages yet.")

        self.state.handle(ToolPermissionResolved("request", False))
        self.assertIsNone(self.state.pending_permission)
        self.assertEqual(self.state.status, "Denying tool")

    def test_provider_draft_is_ephemeral_until_committed_view(self) -> None:
        self.state.handle(ProviderRequestStarted("prompt", "request"))
        self.state.handle(ProviderDeltaReceived("request", TextDelta("hel")))
        self.assertIn("Assistant: hel", self.state.active_text())
        self.assertEqual(self.state.transcript_text(), "No messages yet.")

        output = ProviderOutput((OutputMessage((OutputText("hello"),)),))
        self.state.handle(ProviderResponseCompleted("request", output))
        self.assertIn("Assistant: hello", self.state.active_text())
        self.state.handle(
            SessionViewChanged(
                SessionView(
                    "session",
                    3,
                    (
                        UiPrompt(1, "prompt", "question"),
                        UiModelOutput(
                            2,
                            "output",
                            "prompt",
                            "hello",
                            None,
                            None,
                            (),
                            None,
                        ),
                    ),
                ),
                reset_active=True,
            )
        )
        self.state.handle(ModelOutputRecorded("output"))

        self.assertEqual(self.state.active_text(), "")
        self.assertIn("You: question", self.state.transcript_text())
        self.assertIn("Assistant: hello", self.state.transcript_text())

    def test_request_failure_clears_ephemeral_draft(self) -> None:
        self.state.handle(ProviderRequestStarted("prompt", "request"))
        self.state.handle(ProviderDeltaReceived("request", TextDelta("partial")))
        self.state.handle(RequestFailed("boom"))
        self.assertEqual(self.state.active_text(), "")
        self.assertEqual(self.state.status, "Request failed: boom")

    def test_tool_results_render_beside_calls_with_only_latest_usage(self) -> None:
        first_call = ToolCall("call-one", "calculator", '{"expression":"1+1"}')
        second_call = ToolCall("call-two", "calculator", '{"expression":"2+2"}')
        incomplete_call = ToolCall(
            "call-three",
            "calculator",
            '{"expression":"3+3"}',
        )
        state = ChatUiState(
            SessionView(
                "session",
                9,
                (
                    UiPrompt(1, "prompt-one", "calculate"),
                    UiModelOutput(
                        2,
                        "output-tools",
                        "prompt-one",
                        "I will check.",
                        None,
                        None,
                        (first_call, second_call),
                        TokenUsage(total_tokens=2_407),
                    ),
                    UiToolResult(
                        3,
                        "result-one",
                        "output-tools",
                        "call-one",
                        "2",
                        "success",
                        False,
                    ),
                    UiToolResult(
                        4,
                        "result-two",
                        "output-tools",
                        "call-two",
                        "4",
                        "success",
                        False,
                    ),
                    UiModelOutput(
                        5,
                        "output-final",
                        "prompt-one",
                        "Both checks passed.",
                        None,
                        None,
                        (),
                        TokenUsage(total_tokens=2_652, cached_tokens=1_792),
                    ),
                    UiPrompt(6, "prompt-two", "one more"),
                    UiModelOutput(
                        7,
                        "output-incomplete",
                        "prompt-two",
                        None,
                        None,
                        None,
                        (incomplete_call,),
                        TokenUsage(total_tokens=3_000),
                    ),
                ),
            )
        )

        lines = state.transcript_lines()
        texts = [line.text for line in lines]
        self.assertLess(
            texts.index('  Tool call [calculator]: {"expression":"1+1"}'),
            texts.index("  Tool result [calculator]: 2"),
        )
        self.assertLess(
            texts.index("  Tool result [calculator]: 2"),
            texts.index('  Tool call [calculator]: {"expression":"2+2"}'),
        )
        self.assertLess(
            texts.index('  Tool call [calculator]: {"expression":"2+2"}'),
            texts.index("  Tool result [calculator]: 4"),
        )
        usage_lines = [line.text for line in lines if line.role == "token_usage"]
        self.assertEqual(
            usage_lines,
            ["Total tokens: 2,652 | Cached tokens: 1,792"],
        )

    def test_latest_failed_prompt_suppresses_stale_usage(self) -> None:
        state = ChatUiState(
            SessionView(
                "session",
                4,
                (
                    UiPrompt(1, "prompt-one", "first"),
                    UiModelOutput(
                        2,
                        "output-one",
                        "prompt-one",
                        "done",
                        None,
                        None,
                        (),
                        TokenUsage(total_tokens=10),
                    ),
                    UiPrompt(3, "prompt-two", "second"),
                    UiQueryFailure(4, "prompt-two", "provider unavailable"),
                ),
            )
        )

        self.assertFalse(
            any(
                line.role == "token_usage"
                for line in state.transcript_lines()
            )
        )


class TerminalPermissionUiTests(unittest.TestCase):
    def test_permission_mode_preserves_draft_and_bounds_argument_preview(self) -> None:
        class Provider:
            def list_models(self):
                return ["gpt-test"]

        controller = ApplicationCoordinator(
            Config("key"),
            provider=Provider(),
            session=Session.create(in_memory=True),
        )
        lifecycle = PreToolExecution(
            call_id="call",
            tool_name="file_writer",
            tool_input=FrozenJsonObject({"content": "x" * 3_000}),
        )

        with create_pipe_input() as input:
            ui = TerminalChatUi(controller, input=input, output=DummyOutput())
            ui.input_area.text = "draft prompt"

            async def approve() -> bool:
                pending = asyncio.create_task(
                    controller.request_tool_permission(lifecycle)
                )
                await asyncio.sleep(0)

                self.assertEqual(ui.input_area.text, "draft prompt")
                self.assertIs(
                    ui.app.layout.current_control,
                    ui.permission_deny_button.control,
                )
                self.assertTrue(ui.permission_container.filter())
                self.assertFalse(ui.input_container.filter())
                preview = ui._render_permission_request()
                self.assertIn("file_writer", preview)
                self.assertIn("characters omitted", preview)
                self.assertLess(
                    len(preview),
                    MAX_PERMISSION_PREVIEW_CHARS + 200,
                )

                ui.permission_allow_button.handler()
                return await pending

            self.assertTrue(asyncio.run(approve()))
            self.assertEqual(ui.input_area.text, "draft prompt")
            self.assertIs(
                ui.app.layout.current_control,
                ui.input_area.control,
            )
            self.assertFalse(ui.permission_container.filter())
            self.assertTrue(ui.input_container.filter())
            ui._event_subscription.unsubscribe()
        controller.session.close()


if __name__ == "__main__":
    unittest.main()
