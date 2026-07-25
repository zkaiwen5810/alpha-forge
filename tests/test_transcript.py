import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.model_messages import ToolCall
from alpha_forge.model_history import ModelHistoryProjector
from alpha_forge.session import Session
from alpha_forge.streaming import (
    ModelResponse,
    ModelResponseAccumulator,
    ReasoningDelta,
    StreamCompleted,
    TextDelta,
    ToolCallDelta,
)
from alpha_forge.system_events import (
    AssistantMessageAddFailed,
    ModelResponseStarted,
)
from alpha_forge.transcript import (
    Command,
    CommandMessage,
    CommandResult,
    ModelOutput,
    SessionTransition,
    ToolResult,
    ToolResultLimit,
    Transcript,
    TranscriptCorruptError,
    TranscriptPersistenceError,
    TurnFailure,
    UserMessage,
)
from alpha_forge.ui_history import (
    UiCommandMessage,
    UiHistoryProjector,
    UiModelOutput,
    UiToolResult,
)
from alpha_forge.ui_state import ChatUiState


class TranscriptStorageTests(unittest.TestCase):
    def test_append_reload_permissions_and_flat_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            transcript = Transcript.create(
                system_prompt="system",
                session_id="session",
                path=path,
            )
            transcript.append(UserMessage("turn", None, "hello ☃"))
            transcript.append(ModelOutput("output", "turn", "hi", finish_reason="stop"))

            resumed = Transcript.resume(path)

            self.assertEqual(resumed.events, transcript.events)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(all(row["schema_version"] == 3 for row in rows))
            self.assertEqual(
                [row["type"] for row in rows],
                ["session.start", "user.message", "model.output"],
            )

    def test_resume_repairs_only_an_unterminated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            transcript = Transcript.create(system_prompt=None, path=path)
            transcript.append(UserMessage("turn", None, "hello"))
            with path.open("ab") as stream:
                stream.write(b'{"partial":')

            resumed = Transcript.resume(path)
            resumed.append(TurnFailure("turn", "interrupted"))

            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertIsInstance(resumed.events[-1], TurnFailure)

    def test_model_output_keeps_ordered_tool_calls_in_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            session = Session(transcript_path=path)
            turn = session.submit_user("run")
            session.add_assistant_message(
                turn_id=turn,
                response=ModelResponse(
                    "working",
                    (
                        ToolCall("one", "first", "{}"),
                        ToolCall("two", "second", '{"value":2}'),
                    ),
                ),
            )

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            output_rows = [row for row in rows if row["type"] == "model.output"]

            self.assertEqual(len(output_rows), 1)
            self.assertEqual(
                [call["id"] for call in output_rows[0]["payload"]["tool_calls"]],
                ["one", "two"],
            )
            self.assertFalse(any(row["type"] == "tool.calls" for row in rows))

    def test_resume_rejects_v2_and_corrupt_completed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sequence": 0,
                        "event_id": "event",
                        "recorded_at": "now",
                        "type": "session.started",
                        "payload": {},
                    }
                )
                + "\n"
            )
            with self.assertRaises(TranscriptCorruptError):
                Transcript.resume(path)

    def test_invalid_reference_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            transcript = Transcript.create(system_prompt=None, path=path)
            size = path.stat().st_size

            with self.assertRaises(TranscriptCorruptError):
                transcript.append(ToolResult("result", "missing", "call", "raw"))

            self.assertEqual(path.stat().st_size, size)
            self.assertEqual(transcript.revision, 1)

    def test_failed_append_does_not_publish_and_poisons_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            transcript = Transcript.create(system_prompt=None, path=path)
            real_open = os.open

            def fail_append(raw_path, flags, mode=0o777):  # type: ignore[no-untyped-def]
                if flags & os.O_APPEND:
                    raise OSError("disk full")
                return real_open(raw_path, flags, mode)

            with patch("alpha_forge.transcript.os.open", side_effect=fail_append):
                with self.assertRaises(TranscriptPersistenceError):
                    transcript.append(UserMessage("turn", None, "hello"))

            self.assertEqual(transcript.revision, 1)
            with self.assertRaises(TranscriptPersistenceError):
                transcript.append(UserMessage("turn", None, "hello"))


class ProjectionTests(unittest.TestCase):
    def test_raw_result_and_limit_replay_exact_model_preview(self) -> None:
        session = Session(transcript=Transcript.in_memory(system_prompt="system"))
        turn = session.submit_user("run")
        output = session.add_assistant_message(
            turn_id=turn,
            response=ModelResponse(
                None,
                (ToolCall("call", "tool", "{}"),),
                finish_reason="tool_calls",
            ),
        )
        raw = session.add_tool_result(
            output_id=output.output_id,
            call_id="call",
            content="HEAD" + "x" * 20_000 + "TAIL",
            failed=False,
        )
        limit = session.finalize_tool_results(
            output_id=output.output_id,
            results=(raw,),
        )

        messages = ModelHistoryProjector(session.transcript).messages(head_turn_id=turn)
        tool_message = messages[-1]
        expected = session._limiter.render(raw, limit.decisions[0])
        self.assertEqual(tool_message.content, expected)  # type: ignore[attr-defined]
        self.assertIn("transcript_ref", expected)
        self.assertTrue(
            any(isinstance(e, ToolResult) for e in session.transcript.events)
        )
        self.assertTrue(
            any(isinstance(e, ToolResultLimit) for e in session.transcript.events)
        )

    def test_model_projection_selects_only_root_to_head_ancestry(self) -> None:
        session = Session(transcript=Transcript.in_memory(system_prompt=None))
        first = session.submit_user("first")
        session.add_assistant_message(
            turn_id=first,
            response=ModelResponse("one"),
        )
        abandoned = session.submit_user("abandoned")
        session.add_assistant_message(
            turn_id=abandoned,
            response=ModelResponse("old"),
        )
        session.select_head(first)
        branch = session.submit_user("branch")
        session.add_assistant_message(
            turn_id=branch,
            response=ModelResponse("new"),
        )

        self.assertEqual(
            [message.content for message in session.messages],  # type: ignore[attr-defined]
            ["first", "one", "branch", "new"],
        )

    def test_explicit_none_parent_starts_an_independent_root(self) -> None:
        session = Session(transcript=Transcript.in_memory(system_prompt=None))
        first = session.submit_user("first")
        session.add_assistant_message(
            turn_id=first,
            response=ModelResponse("one"),
        )
        second_root = session.submit_user("new root", parent_turn_id=None)
        session.add_assistant_message(
            turn_id=second_root,
            response=ModelResponse("two"),
        )

        self.assertEqual(
            [message.content for message in session.messages],  # type: ignore[attr-defined]
            ["new root", "two"],
        )

    def test_ui_projector_is_flat_and_filters_raw_unlimited_results(self) -> None:
        session = Session(transcript=Transcript.in_memory())
        turn = session.submit_user("run")
        output = session.add_assistant_message(
            turn_id=turn,
            response=ModelResponse(
                None,
                (ToolCall("call", "tool", "{}"),),
            ),
        )
        raw = session.add_tool_result(
            output_id=output.output_id,
            call_id="call",
            content="raw",
            failed=False,
        )
        before = UiHistoryProjector(session.transcript).items(head_turn_id=turn)
        self.assertTrue(any(isinstance(item, UiModelOutput) for item in before))
        self.assertFalse(any(isinstance(item, UiToolResult) for item in before))

        session.finalize_tool_results(output_id=output.output_id, results=(raw,))
        after = UiHistoryProjector(session.transcript).items(head_turn_id=turn)
        self.assertTrue(any(isinstance(item, UiToolResult) for item in after))

    def test_command_input_is_audited_but_ui_only_projects_result(self) -> None:
        session = Session(transcript=Transcript.in_memory())
        command = session.add_command(
            raw="/help",
            name="/help",
            arguments="",
        )
        session.add_command_result(
            command.command_id,
            status="success",
            messages=(CommandMessage("help"),),
        )

        items = UiHistoryProjector(session.transcript).items()

        self.assertTrue(any(isinstance(e, Command) for e in session.transcript.events))
        self.assertTrue(
            any(isinstance(e, CommandResult) for e in session.transcript.events)
        )
        self.assertEqual(
            [
                item.message.content
                for item in items
                if isinstance(item, UiCommandMessage)
            ],
            ["help"],
        )


class SessionProtocolTests(unittest.TestCase):
    def test_session_enforces_tool_protocol_not_transcript(self) -> None:
        session = Session(transcript=Transcript.in_memory())
        turn = session.submit_user("run")
        output = session.add_assistant_message(
            turn_id=turn,
            response=ModelResponse(
                None,
                (ToolCall("call", "tool", "{}"),),
            ),
        )

        with self.assertRaises(RuntimeError):
            session.add_assistant_message(
                turn_id=turn,
                response=ModelResponse("too early"),
            )

        raw = session.add_tool_result(
            output_id=output.output_id,
            call_id="call",
            content="done",
            failed=False,
        )
        session.finalize_tool_results(output_id=output.output_id, results=(raw,))
        session.add_assistant_message(
            turn_id=turn,
            response=ModelResponse("finished"),
        )

    def test_recovery_synthesizes_missing_results_and_requeues(self) -> None:
        session = Session(transcript=Transcript.in_memory())
        turn = session.submit_user("run")
        session.add_assistant_message(
            turn_id=turn,
            response=ModelResponse(
                None,
                (
                    ToolCall("one", "tool", "{}"),
                    ToolCall("two", "tool", "{}"),
                ),
            ),
        )

        pending = session.recover_unfinished_turns()

        self.assertEqual(
            [(item.turn_id, item.content) for item in pending], [(turn, "run")]
        )
        results = [
            event
            for event in session.transcript.events
            if isinstance(event, ToolResult)
        ]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.failed for result in results))
        self.assertIsInstance(session.transcript.events[-1], ToolResultLimit)

    def test_session_transition_is_destination_activity(self) -> None:
        source = Session(transcript=Transcript.in_memory(session_id="source"))
        command = source.add_command(
            raw="/clear",
            name="/clear",
            arguments="",
        )
        destination = Session(transcript=Transcript.in_memory(session_id="destination"))
        destination.add_session_transition(
            kind="clear",
            source_session_id=source.session_id,
            source_command_id=command.command_id,
        )
        self.assertIsInstance(destination.transcript.events[-1], SessionTransition)


class EphemeralStateTests(unittest.TestCase):
    def test_partial_output_is_ui_only_until_session_persists_response(self) -> None:
        session = Session(transcript=Transcript.in_memory())
        turn = session.submit_user("hello")
        ui = ChatUiState(session.transcript, head_turn_id=turn)
        ui.handle(ModelResponseStarted(turn))
        ui.handle(ReasoningDelta("thinking"))
        ui.handle(TextDelta("partial"))
        ui.handle(ToolCallDelta(0, "call", "tool", "{"))

        self.assertIn("partial", ui.active_text())
        self.assertIn("streaming", ui.active_text())
        self.assertNotIn("partial", ui.transcript_text())
        self.assertFalse(
            any(isinstance(event, ModelOutput) for event in session.transcript.events)
        )

    def test_completed_response_stays_active_on_persistence_failure(self) -> None:
        session = Session(transcript=Transcript.in_memory())
        turn = session.submit_user("hello")
        ui = ChatUiState(session.transcript, head_turn_id=turn)
        response = ModelResponse("complete", finish_reason="stop")
        ui.handle(ModelResponseStarted(turn))
        ui.handle(TextDelta("complete"))
        ui.handle(StreamCompleted(response))
        ui.handle(AssistantMessageAddFailed("disk full"))

        self.assertEqual(response.content, "complete")
        self.assertTrue(ui.has_unsaved_active)
        self.assertIn("not persisted: disk full", ui.active_text())

    def test_mutable_draft_finishes_into_immutable_response(self) -> None:
        draft = ModelResponseAccumulator()
        draft.apply(TextDelta("hello"))
        draft.apply(ToolCallDelta(0, "call", "tool", '{"x":'))
        draft.apply(ToolCallDelta(0, "", "", "1}"))
        response = draft.build("tool_calls")

        self.assertEqual(response.content, "hello")
        self.assertEqual(response.tool_calls[0].arguments, '{"x":1}')
        self.assertEqual(response.finish_reason, "tool_calls")


if __name__ == "__main__":
    unittest.main()
