import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.config import Config
from alpha_forge.models import ToolCall
from alpha_forge.repl_controller import ChatReplController, WorkItem
from alpha_forge.session import Session
from alpha_forge.streaming import (
    ModelResponse,
    StreamCompleted,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    UsageUpdate,
)
from alpha_forge.tools import Tool, ToolRegistry
from alpha_forge.transcript import (
    Command,
    CommandResult,
    ModelOutput,
    SessionTransition,
    ToolResult,
    ToolResultLimit,
    Transcript,
    TurnFailure,
)


class ScriptedChat:
    def __init__(self, responses):  # type: ignore[no-untyped-def]
        self.responses = list(responses)
        self.requests = []

    async def stream_response(self, messages, *, tools):  # type: ignore[no-untyped-def]
        self.requests.append([message.to_openai() for message in messages])
        for event in self.responses.pop(0):
            yield event

    def list_models(self) -> list[str]:
        return ["gpt-test"]


def _completed(
    content: str | None,
    finish_reason: str,
    *,
    tool_calls: tuple[ToolCall, ...] = (),
    usage: TokenUsage | None = None,
) -> StreamCompleted:
    return StreamCompleted(
        ModelResponse(
            content,
            tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )
    )


def _controller(chat: ScriptedChat, *, session: Session | None = None):
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            function=lambda arguments: f"echo:{arguments['text']}",
            description="echo",
            prompt="echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    )
    return ChatReplController(
        Config(api_key="test", model="gpt-test"),
        chat=chat,  # type: ignore[arg-type]
        tool_registry=registry,
        session=session or Session(transcript=Transcript.in_memory()),
    )


class ControllerToolLoopTests(unittest.TestCase):
    def test_tool_call_is_persisted_then_executed_then_model_continues(self) -> None:
        chat = ScriptedChat(
            [
                [
                    ToolCallDelta(0, "call", "echo", '{"text":"hi"}'),
                    _completed(
                        None,
                        "tool_calls",
                        tool_calls=(ToolCall("call", "echo", '{"text":"hi"}'),),
                    ),
                ],
                [
                    TextDelta("done"),
                    UsageUpdate(TokenUsage(20, 10, 25)),
                    _completed(
                        "done",
                        "stop",
                        usage=TokenUsage(20, 10, 25),
                    ),
                ],
            ]
        )
        controller = _controller(chat)
        turn = controller.session.submit_user("run")

        asyncio.run(controller._stream_response(WorkItem("run", turn)))

        events = controller.session.transcript.events
        output_positions = [
            index
            for index, event in enumerate(events)
            if isinstance(event, ModelOutput)
        ]
        result_position = next(
            index for index, event in enumerate(events) if isinstance(event, ToolResult)
        )
        limit_position = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, ToolResultLimit)
        )
        self.assertLess(output_positions[0], result_position)
        self.assertLess(result_position, limit_position)
        self.assertLess(limit_position, output_positions[1])
        self.assertEqual(chat.requests[1][-2]["role"], "assistant")
        self.assertEqual(chat.requests[1][-1]["role"], "tool")
        self.assertIn("Assistant: done", controller.ui_state.transcript_text())

    def test_incomplete_tool_call_is_rendered_but_never_executed(self) -> None:
        chat = ScriptedChat([[ToolCallDelta(0, "call", "echo", '{"text":')]])
        controller = _controller(chat)
        turn = controller.session.submit_user("run")

        asyncio.run(controller._stream_response(WorkItem("run", turn)))

        self.assertFalse(
            any(
                isinstance(event, ToolResult)
                for event in controller.session.transcript.events
            )
        )
        self.assertTrue(
            any(
                isinstance(event, TurnFailure)
                for event in controller.session.transcript.events
            )
        )

    def test_completed_response_remains_visible_when_session_add_fails(
        self,
    ) -> None:
        chat = ScriptedChat([[TextDelta("complete"), _completed("complete", "stop")]])
        controller = _controller(chat)
        turn = controller.session.submit_user("run")

        with patch.object(
            controller.session,
            "add_assistant_message",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(controller._stream_response(WorkItem("run", turn)))

        self.assertTrue(controller.ui_state.has_unsaved_active)
        self.assertIn("complete", controller.ui_state.active_text())
        self.assertIn("disk full", controller.ui_state.active_text())
        self.assertFalse(
            any(
                isinstance(event, ModelOutput)
                for event in controller.session.transcript.events
            )
        )

    def test_malformed_tool_arguments_become_failed_result(self) -> None:
        chat = ScriptedChat(
            [
                [
                    ToolCallDelta(0, "call", "echo", "not json"),
                    _completed(
                        None,
                        "tool_calls",
                        tool_calls=(ToolCall("call", "echo", "not json"),),
                    ),
                ],
                [TextDelta("handled"), _completed("handled", "stop")],
            ]
        )
        controller = _controller(chat)
        turn = controller.session.submit_user("run")

        asyncio.run(controller._stream_response(WorkItem("run", turn)))

        result = next(
            event
            for event in controller.session.transcript.events
            if isinstance(event, ToolResult)
        )
        self.assertTrue(result.failed)
        self.assertIn("error:", result.content)

    def test_tool_limit_persistence_failure_keeps_ephemeral_results(self) -> None:
        chat = ScriptedChat(
            [
                [
                    ToolCallDelta(0, "call", "echo", '{"text":"hi"}'),
                    _completed(
                        None,
                        "tool_calls",
                        tool_calls=(ToolCall("call", "echo", '{"text":"hi"}'),),
                    ),
                ]
            ]
        )
        controller = _controller(chat)
        turn = controller.session.submit_user("run")

        with patch.object(
            controller.session,
            "finalize_tool_results",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(controller._stream_response(WorkItem("run", turn)))

        self.assertTrue(controller.ui_state.has_unsaved_active)
        self.assertIn("echo:hi", controller.ui_state.active_text())
        self.assertIn(
            "tool results not finalized: disk full",
            controller.ui_state.active_text(),
        )

    def test_queued_turn_receives_completed_parent_history(self) -> None:
        chat = ScriptedChat(
            [
                [
                    TextDelta("first answer"),
                    _completed("first answer", "stop"),
                ],
                [
                    TextDelta("second answer"),
                    _completed("second answer", "stop"),
                ],
            ]
        )
        controller = _controller(chat)
        first = controller.session.submit_user("first")
        second = controller.session.submit_user("second")

        asyncio.run(controller._stream_response(WorkItem("first", first)))
        asyncio.run(controller._stream_response(WorkItem("second", second)))

        self.assertEqual(
            [message["content"] for message in chat.requests[1]],
            ["first", "first answer", "second"],
        )


class CommandActivityTests(unittest.TestCase):
    def test_help_logs_command_and_result(self) -> None:
        controller = _controller(ScriptedChat([]))

        controller.submit("  /help  ")

        command = next(
            event
            for event in controller.session.transcript.events
            if isinstance(event, Command)
        )
        self.assertEqual(command.raw, "  /help  ")
        self.assertTrue(
            any(
                isinstance(event, CommandResult)
                for event in controller.session.transcript.events
            )
        )
        self.assertIn("/help /model", controller.ui_state.transcript_text())

    def test_unknown_command_is_audited_and_rendered_as_error(self) -> None:
        controller = _controller(ScriptedChat([]))

        controller.submit("/missing")

        result = next(
            event
            for event in controller.session.transcript.events
            if isinstance(event, CommandResult)
        )
        self.assertEqual(result.status, "error")
        self.assertIn("unknown command", controller.ui_state.transcript_text())

    def test_clear_writes_source_audit_and_destination_transition(self) -> None:
        controller = _controller(ScriptedChat([]))
        source = controller.session

        controller.submit("/clear")

        self.assertIsNot(controller.session, source)
        self.assertTrue(
            any(isinstance(event, CommandResult) for event in source.transcript.events)
        )
        transition = next(
            event
            for event in controller.session.transcript.events
            if isinstance(event, SessionTransition)
        )
        self.assertEqual(transition.kind, "clear")
        self.assertEqual(transition.source_session_id, source.session_id)

    def test_resume_requeues_unanswered_durable_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved = Session(transcript_path=Path(tmp) / "saved.jsonl")
            turn = saved.submit_user("unfinished")
            controller = _controller(ScriptedChat([]))

            controller.submit(f"/resume {saved.transcript_path}")

            self.assertEqual(controller.session.head_turn_id, turn)
            queued = controller.queue.get_nowait()
            self.assertEqual(queued, WorkItem("unfinished", turn))
            self.assertTrue(
                any(
                    isinstance(event, SessionTransition)
                    for event in controller.session.transcript.events
                )
            )


if __name__ == "__main__":
    unittest.main()
