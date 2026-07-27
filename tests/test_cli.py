import unittest

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
)
from alpha_forge.cli import build_parser
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


if __name__ == "__main__":
    unittest.main()
