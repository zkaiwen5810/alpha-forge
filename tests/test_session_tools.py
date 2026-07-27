import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.config import Config
from alpha_forge.events import Event
from alpha_forge.models import ToolCall
from alpha_forge.repl_controller import ChatReplController
from alpha_forge.session import Session
from alpha_forge.streaming import (
    ModelResponse,
    StreamCompleted,
    TextDelta,
    ToolCallDelta,
)
from alpha_forge.tools import Tool, ToolRegistry
from alpha_forge.transcript import (
    Command,
    CommandResult,
    ModelOutput,
    SessionTransition,
    ToolResult,
    ToolResultEdit,
    Transcript,
    TurnFailure,
    UserMessage,
)
from alpha_forge.ui_state import ChatUiState


class ScriptedChat:
    def __init__(self, responses):  # type: ignore[no-untyped-def]
        self.responses = list(responses)
        self.requests: list[list[dict[str, object]]] = []

    async def stream_response(self, messages, *, tools):  # type: ignore[no-untyped-def]
        self.requests.append([message.to_openai() for message in messages])
        for event in self.responses.pop(0):
            yield event

    def list_models(self) -> list[str]:
        return ["gpt-test"]


def _completed(
    content: str | None,
    *,
    calls: tuple[ToolCall, ...] = (),
) -> StreamCompleted:
    return StreamCompleted(ModelResponse(content, calls))


def _controller(
    chat: ScriptedChat,
    *,
    session: Session | None = None,
) -> tuple[ChatReplController, ChatUiState]:
    registry = ToolRegistry(
        [
            Tool(
                name="echo",
                function=lambda arguments: f"echo:{arguments['text']}",
                description="echo",
                prompt="echo",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            )
        ]
    )
    controller = ChatReplController(
        Config(api_key="test", model="gpt-test"),
        chat=chat,  # type: ignore[arg-type]
        tool_registry=registry,
        session=session or Session(transcript=Transcript.in_memory()),
    )
    ui = ChatUiState(controller.initial_view)
    controller.events.subscribe(Event, ui.handle)
    return controller, ui


async def _run(controller: ChatReplController, *inputs: str) -> None:
    for value in inputs:
        controller.submit(value)
    controller.request_exit()
    await controller.consume()


class ControllerToolLoopTests(unittest.TestCase):
    def test_tool_call_is_committed_executed_edited_then_continued(self) -> None:
        call = ToolCall("call", "echo", '{"text":"hi"}')
        chat = ScriptedChat(
            [
                [
                    ToolCallDelta(0, "call", "echo", '{"text":"hi"}'),
                    _completed(None, calls=(call,)),
                ],
                [TextDelta("done"), _completed("done")],
            ]
        )
        controller, ui = _controller(chat)

        asyncio.run(_run(controller, "run"))

        events = controller.session.transcript.events
        output_positions = [
            index for index, event in enumerate(events) if isinstance(event, ModelOutput)
        ]
        result_position = next(
            index for index, event in enumerate(events) if isinstance(event, ToolResult)
        )
        edit_position = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, ToolResultEdit)
        )
        self.assertLess(output_positions[0], result_position)
        self.assertLess(result_position, edit_position)
        self.assertLess(edit_position, output_positions[1])
        self.assertEqual(chat.requests[1][-2]["role"], "assistant")
        self.assertEqual(chat.requests[1][-1]["role"], "tool")
        self.assertIn("Assistant: done", ui.transcript_text())

    def test_incomplete_stream_is_failed_without_executing_tool(self) -> None:
        controller, _ui = _controller(
            ScriptedChat([[ToolCallDelta(0, "call", "echo", '{"text":')]])
        )

        asyncio.run(_run(controller, "run"))

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

    def test_completed_response_remains_visible_on_commit_failure(self) -> None:
        controller, ui = _controller(
            ScriptedChat([[TextDelta("complete"), _completed("complete")]])
        )

        with patch.object(
            controller.session,
            "add_assistant_message",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(_run(controller, "run"))

        self.assertTrue(ui.has_unsaved_active)
        self.assertIn("complete", ui.active_text())
        self.assertIn("disk full", ui.active_text())

    def test_model_commit_failure_prevents_requested_tool_execution(self) -> None:
        call = ToolCall("call", "echo", '{"text":"never"}')
        chat = ScriptedChat([[_completed(None, calls=(call,))]])
        controller, ui = _controller(chat)

        with patch.object(
            controller.session,
            "add_assistant_message",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(_run(controller, "run"))

        self.assertEqual(len(chat.requests), 1)
        self.assertFalse(
            any(
                isinstance(event, ToolResult)
                for event in controller.session.transcript.events
            )
        )
        self.assertTrue(ui.has_unsaved_active)

    def test_persistence_failure_skips_later_queued_inputs(self) -> None:
        chat = ScriptedChat([[_completed("unsaved")]])
        controller, ui = _controller(chat)

        with patch.object(
            controller.session,
            "add_assistant_message",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(_run(controller, "first", "must not run"))

        self.assertEqual(len(chat.requests), 1)
        self.assertFalse(controller.accepting)
        self.assertTrue(ui.has_unsaved_active)
        self.assertEqual(
            ui.status,
            "Persistence failed; input processing stopped",
        )

    def test_raw_result_commit_failure_prevents_prompt_edit_and_next_round(
        self,
    ) -> None:
        call = ToolCall("call", "echo", '{"text":"hi"}')
        chat = ScriptedChat([[_completed(None, calls=(call,))]])
        controller, ui = _controller(chat)

        with patch.object(
            controller.session,
            "record_tool_result",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(_run(controller, "run"))

        self.assertEqual(len(chat.requests), 1)
        self.assertFalse(
            any(
                isinstance(event, ToolResultEdit)
                for event in controller.session.transcript.events
            )
        )
        self.assertTrue(ui.has_unsaved_active)

    def test_prompt_edit_commit_failure_retains_preview_and_stops_query(
        self,
    ) -> None:
        call = ToolCall("call", "echo", '{"text":"hi"}')
        chat = ScriptedChat([[_completed(None, calls=(call,))]])
        controller, ui = _controller(chat)

        with patch.object(
            controller.session,
            "add_prompt_edit",
            side_effect=RuntimeError("disk full"),
        ):
            asyncio.run(_run(controller, "run"))

        self.assertEqual(len(chat.requests), 1)
        self.assertTrue(
            any(
                isinstance(event, ToolResult)
                for event in controller.session.transcript.events
            )
        )
        self.assertTrue(ui.has_unsaved_active)
        self.assertIn("echo:hi", ui.active_text())

    def test_malformed_arguments_become_failed_raw_result(self) -> None:
        call = ToolCall("call", "echo", "not json")
        chat = ScriptedChat(
            [
                [_completed(None, calls=(call,))],
                [_completed("handled")],
            ]
        )
        controller, _ui = _controller(chat)

        asyncio.run(_run(controller, "run"))

        result = next(
            event
            for event in controller.session.transcript.events
            if isinstance(event, ToolResult)
        )
        self.assertTrue(result.failed)
        self.assertIn("error:", result.content)

    def test_queued_turn_sees_completed_parent_history(self) -> None:
        chat = ScriptedChat(
            [
                [_completed("first answer")],
                [_completed("second answer")],
            ]
        )
        controller, _ui = _controller(chat)

        asyncio.run(_run(controller, "first", "second"))

        self.assertEqual(
            [message["content"] for message in chat.requests[1]],
            ["first", "first answer", "second"],
        )


class UnifiedInputTests(unittest.TestCase):
    def test_waiting_input_is_not_persisted_until_dequeued(self) -> None:
        controller, _ui = _controller(
            ScriptedChat([[_completed("done")]])
        )

        controller.submit("waiting")

        self.assertFalse(
            any(
                isinstance(event, UserMessage)
                for event in controller.session.transcript.events
            )
        )
        controller.request_exit()
        asyncio.run(controller.consume())

    def test_help_is_queued_audited_and_rendered(self) -> None:
        controller, ui = _controller(ScriptedChat([]))

        asyncio.run(_run(controller, "/help"))

        self.assertTrue(
            any(
                isinstance(event, Command)
                for event in controller.session.transcript.events
            )
        )
        self.assertTrue(
            any(
                isinstance(event, CommandResult)
                for event in controller.session.transcript.events
            )
        )
        self.assertIn("/help /model", ui.transcript_text())

    def test_prompt_clear_prompt_uses_fifo_session_selection(self) -> None:
        chat = ScriptedChat(
            [
                [_completed("old answer")],
                [_completed("new answer")],
            ]
        )
        controller, ui = _controller(chat)
        source = controller.session

        asyncio.run(_run(controller, "old prompt", "/clear", "new prompt"))

        self.assertIsNot(controller.session, source)
        self.assertTrue(
            any(
                isinstance(event, CommandResult)
                for event in source.transcript.events
            )
        )
        self.assertTrue(
            any(
                isinstance(event, SessionTransition)
                for event in controller.session.transcript.events
            )
        )
        self.assertNotIn("old prompt", ui.transcript_text())
        self.assertIn("new prompt", ui.transcript_text())
        self.assertEqual(chat.requests[1][0]["content"], "new prompt")

    def test_resume_repairs_and_continues_unfinished_turn_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved = Session(transcript_path=Path(tmp) / "saved.jsonl")
            turn = saved.submit_user("unfinished")
            controller, _ui = _controller(
                ScriptedChat([[_completed("recovered")]])
            )

            asyncio.run(_run(controller, f"/resume {saved.transcript_path}"))

            self.assertEqual(controller.session.head_turn_id, turn)
            self.assertTrue(
                any(
                    isinstance(event, ModelOutput)
                    for event in controller.session.transcript.events
                )
            )

    def test_resume_edits_repaired_tool_batch_before_model_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved = Session(transcript_path=Path(tmp) / "saved.jsonl")
            turn = saved.submit_user("unfinished")
            call = ToolCall("call", "echo", '{"text":"must not rerun"}')
            prior_output = saved.add_assistant_message(
                turn_id=turn,
                response=ModelResponse(None, (call,)),
            )
            chat = ScriptedChat([[_completed("recovered")]])
            controller, _ui = _controller(chat)

            asyncio.run(_run(controller, f"/resume {saved.transcript_path}"))

            events = controller.session.transcript.events
            edit_position = next(
                index
                for index, event in enumerate(events)
                if isinstance(event, ToolResultEdit)
                and event.output_id == prior_output.output_id
            )
            next_output_position = next(
                index
                for index, event in enumerate(events)
                if isinstance(event, ModelOutput)
                and event.output_id != prior_output.output_id
            )
            self.assertLess(edit_position, next_output_position)
            recovered_results = [
                event
                for event in events
                if isinstance(event, ToolResult)
                and event.output_id == prior_output.output_id
            ]
            self.assertEqual(len(recovered_results), 1)
            self.assertTrue(recovered_results[0].failed)
            recovered_result_position = events.index(recovered_results[0])
            self.assertLess(recovered_result_position, edit_position)
            self.assertEqual(len(chat.requests), 1)
            self.assertEqual(
                [message["role"] for message in chat.requests[0]],
                ["system", "user", "assistant", "tool"],
            )
            self.assertIn(
                "interrupted before its result was durably recorded",
                chat.requests[0][-1]["content"],
            )

    def test_exit_stops_accepting_and_runs_after_earlier_input(self) -> None:
        controller, ui = _controller(
            ScriptedChat([[_completed("before exit")]])
        )
        controller.submit("first")
        controller.submit("/exit")
        controller.submit("ignored")

        asyncio.run(controller.consume())

        self.assertFalse(controller.accepting)
        self.assertIn("before exit", ui.transcript_text())
        self.assertNotIn("ignored", ui.transcript_text())


if __name__ == "__main__":
    unittest.main()
