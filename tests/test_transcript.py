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
    SCHEMA_VERSION,
    SetToolExchangeVisibility,
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
            prompt = session.accept_prompt("hello")
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
            self.assertIsNone(resumed.open_query())
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
            with self.assertRaises(TranscriptPersistenceError):
                TranscriptStore.resume(path)
            with self.assertRaisesRegex(
                TranscriptPersistenceError,
                "stale transcript revision",
            ):
                store.append(
                    InputAccepted("prompt", "hello"),
                    expected_revision=0,
                )
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
            store.close()
            with path.open("ab") as stream:
                stream.write(b'{"schema_version":1')

            resumed = TranscriptStore.resume(path)
            self.assertEqual(resumed.revision, 1)
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
