import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.context import ContextPipeline, ToolResultBudgetPolicy
from alpha_forge.providers import (
    OutputMessage,
    OutputText,
    ProviderOutput,
    ToolCall,
)
from alpha_forge.sessions import Session
from alpha_forge.transcript import (
    ContextEdited,
    InputAccepted,
    PolicyInvocation,
    QueryFailed,
    SCHEMA_VERSION,
    SetToolExchangeVisibility,
    ToolResult,
    TranscriptCorruptError,
    TranscriptPersistenceError,
    TranscriptStore,
)
from alpha_forge.projectors import (
    ModelContextProjector,
    UiHistoryProjector,
)
from alpha_forge.projectors.ui_history import (
    UiModelOutput,
    UiQueryFailure,
    UiToolResult,
)


def _answer(text: str) -> ProviderOutput:
    return ProviderOutput((OutputMessage((OutputText(text),)),), "stop")


class TranscriptSchemaTests(unittest.TestCase):
    def test_new_record_schema_is_version_one_and_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            session = Session.create(
                transcript_path=path,
                session_id="session-one",
            )
            self.assertFalse(path.exists())
            prompt = session.accept_prompt("hello")
            self.assertTrue(path.exists())
            session.record_model_output(prompt.event_id, _answer("hi"))
            session.close()

            records = [
                json.loads(line) for line in path.read_text().splitlines()
            ]

        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual([record["schema_version"] for record in records], [1, 1, 1])
        self.assertEqual([record["sequence"] for record in records], [0, 1, 2])
        self.assertEqual(
            [record["type"] for record in records],
            ["session.opened", "input.accepted", "model.output"],
        )
        serialized = json.dumps(records)
        self.assertNotIn("turn_id", serialized)
        self.assertNotIn("parent_event_id", serialized)

    def test_new_session_persists_when_first_command_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            session = Session.create(transcript_path=path)

            self.assertFalse(path.exists())
            session.accept_command(
                text="/help",
                name="/help",
                arguments="",
            )
            self.assertTrue(path.exists())
            session.close()

            records = [
                json.loads(line) for line in path.read_text().splitlines()
            ]

        self.assertEqual(
            [record["type"] for record in records],
            ["session.opened", "input.accepted"],
        )

    def test_resume_replays_and_validates_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            session = Session.create(transcript_path=path)
            prompt = session.accept_prompt("hello")
            session.record_model_output(prompt.event_id, _answer("hi"))
            expected = session.ui_history()
            session.close()

            resumed = Session.resume(path)
            self.assertEqual(resumed.ui_history(), expected)
            self.assertIsNone(resumed.transcript.state.active_prompt_event_id)
            resumed.close()

    def test_context_edit_round_trips_with_reproducible_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            session = Session.create(transcript_path=path)
            prompt = session.accept_prompt("tool")
            output = session.record_model_output(
                prompt.event_id,
                ProviderOutput((ToolCall("call", "tool", "{}"),)),
            )
            session.record_tool_result(
                model_output_event_id=output.event_id,
                call_id="call",
                status="success",
                content="x" * 1000,
            )
            expected = session.prepare_context(
                ContextPipeline(
                    (
                        ToolResultBudgetPolicy(
                            individual_limit=300,
                            aggregate_limit=300,
                        ),
                    )
                )
            )
            session.close()

            resumed = Session.resume(path)
            actual = ModelContextProjector(resumed.transcript).project()
            resumed.close()

        self.assertEqual(actual, expected)

    def test_other_schema_versions_are_rejected_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "sequence": 0,
                        "event_id": "old",
                        "recorded_at": "2026-01-01T00:00:00Z",
                        "type": "session.opened",
                        "payload": {
                            "session_id": "old",
                            "instructions": None,
                        },
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(
                TranscriptCorruptError,
                "unsupported transcript schema version",
            ):
                TranscriptStore.resume(path)

    def test_exclusive_writer_and_expected_revision_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            store = TranscriptStore.create(instructions=None, path=path)
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(
                TranscriptPersistenceError,
                "stale transcript revision",
            ):
                store.append(
                    InputAccepted("prompt", "hello"),
                    expected_revision=0,
                )
            store.append(
                InputAccepted("prompt", "hello"),
                expected_revision=1,
            )
            with self.assertRaises(TranscriptPersistenceError):
                TranscriptStore.resume(path)
            store.close()

    def test_replay_indexes_are_exposed_as_read_snapshots(self) -> None:
        store = TranscriptStore.in_memory(instructions=None)
        visible = store.state
        visible.event_ids.clear()
        visible.session = None

        self.assertEqual(store.revision, 1)
        self.assertIsNotNone(store.state.session)
        store.close()

    def test_incomplete_final_fragment_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            store = TranscriptStore.create(instructions=None, path=path)
            store.append(
                InputAccepted("prompt", "hello"),
                expected_revision=1,
            )
            store.close()
            with path.open("ab") as stream:
                stream.write(b'{"schema_version":1')

            resumed = TranscriptStore.resume(path)
            self.assertEqual(resumed.revision, 2)
            resumed.close()
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_failed_wal_append_never_updates_visible_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            store = TranscriptStore.create(instructions=None, path=path)
            revision = store.revision
            with patch(
                "alpha_forge.transcript.store.os.write",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(
                    TranscriptPersistenceError,
                    "disk full",
                ):
                    store.append(
                        InputAccepted("prompt", "hello"),
                        expected_revision=revision,
                    )
            self.assertEqual(store.revision, revision)
            self.assertIsNone(store.state.active_prompt_event_id)
            with self.assertRaisesRegex(
                TranscriptPersistenceError,
                "earlier write failure",
            ):
                store.append(
                    InputAccepted("prompt", "retry"),
                    expected_revision=revision,
                )
            store.close()


class SessionInterruptionTests(unittest.TestCase):
    def test_activation_preserves_history_and_closes_each_unfinished_boundary(self):
        for recorded_count in (None, 0, 1, 2):
            with self.subTest(recorded_count=recorded_count):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "session.jsonl"
                    session = Session.create(transcript_path=path)
                    prompt = session.accept_prompt("unfinished")
                    if recorded_count is not None:
                        output = session.record_model_output(
                            prompt.event_id,
                            ProviderOutput((
                                ToolCall("one", "tool", "{}"),
                                ToolCall("two", "tool", "{}"),
                            )),
                        )
                        for call_id in ("one", "two")[:recorded_count]:
                            session.record_tool_result(
                                model_output_event_id=output.event_id,
                                call_id=call_id,
                                status="success",
                                content=f"saved {call_id}",
                            )
                    original = session.transcript.records
                    session.close()
                    original_bytes = path.read_bytes()

                    session = Session.resume(path)
                    self.assertEqual(path.read_bytes(), original_bytes)
                    self.assertEqual(session.transcript.records, original)
                    session.interrupt_open_query()
                    self.assertEqual(session.transcript.records[:len(original)], original)
                    results = [
                        event for event in session.transcript.events
                        if isinstance(event, ToolResult)
                    ]
                    if recorded_count is None:
                        self.assertEqual(results, [])
                    else:
                        self.assertEqual([r.call_id for r in results], ["one", "two"])
                        self.assertEqual(
                            [r.status for r in results],
                            ["success"] * recorded_count
                            + ["interrupted"] * (2 - recorded_count),
                        )
                        for result in results[recorded_count:]:
                            self.assertIn("outcome is unknown", result.content)
                    failure = session.transcript.events[-1]
                    self.assertIsInstance(failure, QueryFailed)
                    self.assertEqual(failure.stage, "interrupted")
                    self.assertEqual(failure.prompt_event_id, prompt.event_id)
                    self.assertIsInstance(session.ui_history()[-1], UiQueryFailure)
                    expected_context = ModelContextProjector(session.transcript).project()
                    revision = session.revision
                    session.interrupt_open_query()
                    self.assertEqual(session.revision, revision)
                    session.close()

                    session = Session.resume(path)
                    session.interrupt_open_query()
                    self.assertEqual(session.revision, revision)
                    self.assertEqual(
                        ModelContextProjector(session.transcript).project(),
                        expected_context,
                    )
                    session.accept_prompt("continue")
                    session.close()

    def test_finalization_restarts_after_each_failed_append(self):
        for committed_count in (0, 1, 2):
            with self.subTest(committed_count=committed_count):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "session.jsonl"
                    session = Session.create(transcript_path=path)
                    prompt = session.accept_prompt("unfinished")
                    session.record_model_output(
                        prompt.event_id,
                        ProviderOutput((
                            ToolCall("one", "tool", "{}"),
                            ToolCall("two", "tool", "{}"),
                        )),
                    )
                    revision = session.revision
                    commit = session._commit

                    def fail_after_commits(event):
                        if session.revision == revision + committed_count:
                            raise TranscriptPersistenceError("disk full")
                        return commit(event)

                    with patch.object(session, "_commit", side_effect=fail_after_commits):
                        with self.assertRaises(TranscriptPersistenceError):
                            session.interrupt_open_query()
                    self.assertEqual(session.revision, revision + committed_count)
                    session.close()

                    session = Session.resume(path)
                    session.interrupt_open_query()
                    self.assertEqual(session.revision, revision + 3)
                    self.assertIsNone(session.transcript.state.active_prompt_event_id)
                    ModelContextProjector(session.transcript).project()
                    session.close()

    def test_empty_completed_and_failed_queries_need_no_finalization(self):
        for ending in ("empty", "completed", "failed"):
            with self.subTest(ending=ending):
                session = Session.create(in_memory=True)
                self.addCleanup(session.close)
                if ending != "empty":
                    prompt = session.accept_prompt("hello")
                    if ending == "completed":
                        session.record_model_output(prompt.event_id, _answer("done"))
                    else:
                        session.fail_query(prompt.event_id, stage="provider", message="offline")
                revision = session.revision
                session.interrupt_open_query()
                self.assertEqual(session.revision, revision)


class TranscriptProtocolTests(unittest.TestCase):
    def test_model_output_is_atomic_but_tool_results_are_flat(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("use both")
        calls = (
            ToolCall("one", "a", "{}"),
            ToolCall("two", "b", "{}"),
        )
        output = session.record_model_output(
            prompt.event_id,
            ProviderOutput(calls, "tool_calls"),
        )
        first = session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="one",
            status="success",
            content="first",
        )
        second = session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="two",
            status="error",
            content="second",
        )

        self.assertEqual(session.transcript.state.outputs[output.event_id].items, calls)
        self.assertEqual(
            session.transcript.state.results_by_output[output.event_id],
            {"one": first.event_id, "two": second.event_id},
        )

    def test_duplicate_or_unknown_tool_results_are_rejected(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("tool")
        output = session.record_model_output(
            prompt.event_id,
            ProviderOutput((ToolCall("one", "a", "{}"),)),
        )
        with self.assertRaises(TranscriptCorruptError):
            session.record_tool_result(
                model_output_event_id=output.event_id,
                call_id="unknown",
                status="success",
                content="x",
            )
        session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="one",
            status="success",
            content="x",
        )
        with self.assertRaises(TranscriptCorruptError):
            session.record_tool_result(
                model_output_event_id=output.event_id,
                call_id="one",
                status="success",
                content="again",
            )

    def test_tool_results_must_follow_provider_call_order(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("tools")
        output = session.record_model_output(
            prompt.event_id,
            ProviderOutput(
                (
                    ToolCall("first", "tool", "{}"),
                    ToolCall("second", "tool", "{}"),
                )
            ),
        )
        with self.assertRaisesRegex(
            TranscriptCorruptError,
            "call order",
        ):
            session.record_tool_result(
                model_output_event_id=output.event_id,
                call_id="second",
                status="success",
                content="out of order",
            )

    def test_visibility_targets_old_exchange_by_event_id_not_time(self) -> None:
        session = Session.create(in_memory=True)
        first_prompt = session.accept_prompt("old query")
        tool_output = session.record_model_output(
            first_prompt.event_id,
            ProviderOutput((ToolCall("call", "tool", "{}"),)),
        )
        result = session.record_tool_result(
            model_output_event_id=tool_output.event_id,
            call_id="call",
            status="success",
            content="raw",
        )
        session.record_model_output(first_prompt.event_id, _answer("old done"))
        second_prompt = session.accept_prompt("new query")
        session.record_model_output(second_prompt.event_id, _answer("new done"))

        session.transcript.append(
            ContextEdited(
                PolicyInvocation(
                    "future_context_occupation_policy",
                    1,
                    {"occupation_ratio": 0.92},
                ),
                (SetToolExchangeVisibility(tool_output.event_id, False),),
            ),
            expected_revision=session.revision,
        )

        model = ModelContextProjector(session.transcript).project()
        ui = UiHistoryProjector(session.transcript).items()
        model_ids = {
            item.output_event_id
            for item in model.items
            if hasattr(item, "output_event_id")
        }
        self.assertNotIn(tool_output.event_id, model_ids)
        self.assertFalse(
            any(
                getattr(item, "result_event_id", None) == result.event_id
                for item in model.items
            )
        )
        self.assertTrue(
            any(
                isinstance(item, UiModelOutput)
                and item.output_event_id == tool_output.event_id
                for item in ui
            )
        )
        projected_result = next(
            item
            for item in ui
            if isinstance(item, UiToolResult)
            and item.result_event_id == result.event_id
        )
        self.assertTrue(projected_result.excluded_from_model)

        session.transcript.append(
            ContextEdited(
                PolicyInvocation("manual_restore", 1, {}),
                (SetToolExchangeVisibility(tool_output.event_id, True),),
            ),
            expected_revision=session.revision,
        )
        restored_ids = {
            item.output_event_id
            for item in ModelContextProjector(session.transcript).project().items
            if hasattr(item, "output_event_id")
        }
        self.assertIn(tool_output.event_id, restored_ids)

    def test_visibility_cannot_hide_current_exchange_tail_or_noop(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("query")
        output = session.record_model_output(
            prompt.event_id,
            ProviderOutput((ToolCall("call", "tool", "{}"),)),
        )
        session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="call",
            status="success",
            content="raw",
        )
        with self.assertRaises(TranscriptCorruptError):
            session.transcript.append(
                ContextEdited(
                    PolicyInvocation("policy", 1, {}),
                    (SetToolExchangeVisibility(output.event_id, False),),
                ),
                expected_revision=session.revision,
            )
        with self.assertRaises(TranscriptCorruptError):
            session.transcript.append(
                ContextEdited(PolicyInvocation("policy", 1, {}), ()),
                expected_revision=session.revision,
            )


if __name__ == "__main__":
    unittest.main()
