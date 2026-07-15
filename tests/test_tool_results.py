import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.tool_results import (
    RawToolResult,
    ToolResultBudgetError,
    ToolResultManager,
    ToolResultPersistenceError,
    default_persist_directory,
)


class ToolResultManagerTests(unittest.TestCase):
    def test_default_limits_match_builtin_policy(self) -> None:
        manager = ToolResultManager(session_id="session")

        self.assertEqual(manager.individual_limit, 16_000)
        self.assertEqual(manager.aggregate_limit, 32_000)

    def test_results_within_budgets_are_unchanged_and_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ToolResultManager(
                persist_directory=root,
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            messages = manager.process(
                (
                    RawToolResult("call-1", "first"),
                    RawToolResult("call-2", "second", failed=True),
                )
            )

            self.assertEqual(
                [message.content for message in messages],
                ["first", "second"],
            )
            self.assertEqual([message.failed for message in messages], [False, True])
            self.assertTrue(all(message.preview is None for message in messages))
            self.assertEqual(list(root.rglob("*")), [])

    def test_result_at_individual_limit_is_not_previewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            message = manager.process(
                (RawToolResult("call-1", "x" * 500),)
            )[0]

            self.assertIsNone(message.preview)
            self.assertEqual(message.content, "x" * 500)

    def test_individual_overflow_creates_head_tail_preview(self) -> None:
        content = "HEAD" + ("x" * 1_000) + "TAIL"
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            message = manager.process((RawToolResult("call-1", content),))[0]

            self.assertEqual(len(message.content), 500)
            self.assertIn("[alpha-forge tool-result-preview]", message.content)
            self.assertIn("truncated: true", message.content)
            self.assertIn("reason: individual_limit", message.content)
            self.assertIn("HEAD", message.content)
            self.assertIn("TAIL", message.content)
            self.assertIsNotNone(message.preview)
            assert message.preview is not None
            self.assertEqual(message.preview.original_chars, len(content))
            self.assertEqual(message.preview.persisted_path.read_text(), content)
            self.assertEqual(
                message.preview.persisted_path,
                Path(tmp).resolve()
                / "session"
                / "tool-results"
                / "call-1.txt",
            )
            self.assertIn(str(message.preview.persisted_path), message.content)

    def test_failed_result_retains_failure_state_when_previewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            message = manager.process(
                (RawToolResult("call-error", "error: " + ("x" * 1_000), True),)
            )[0]

            self.assertTrue(message.failed)
            self.assertIsNotNone(message.preview)

    def test_parseable_json_uses_json_extension_and_preserves_exact_text(self) -> None:
        content = json.dumps({"items": ["x" * 1_000]}, indent=2)
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            message = manager.process((RawToolResult("call-json", content),))[0]

            assert message.preview is not None
            self.assertEqual(message.preview.persisted_path.suffix, ".json")
            self.assertEqual(message.preview.persisted_path.read_text(), content)

    def test_aggregate_budget_water_fills_and_preserves_short_result(self) -> None:
        short = "s" * 100
        long_one = "a" * 900
        long_two = "b" * 900
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=1_000,
                aggregate_limit=1_100,
                session_id="session",
            )

            messages = manager.process(
                (
                    RawToolResult("call-short", short),
                    RawToolResult("call-a", long_one),
                    RawToolResult("call-b", long_two),
                )
            )

            self.assertEqual(messages[0].content, short)
            self.assertIsNone(messages[0].preview)
            self.assertEqual(
                [len(message.content) for message in messages],
                [100, 500, 500],
            )
            self.assertEqual(sum(len(message.content) for message in messages), 1_100)
            assert messages[1].preview is not None
            assert messages[2].preview is not None
            self.assertEqual(messages[1].preview.reason, "aggregate_limit")
            self.assertEqual(messages[2].preview.reason, "aggregate_limit")
            self.assertFalse(
                (Path(tmp) / "session" / "tool-results" / "call-short.txt").exists()
            )

    def test_individual_and_aggregate_reason_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=700,
                aggregate_limit=1_000,
                session_id="session",
            )

            messages = manager.process(
                (
                    RawToolResult("call-a", "a" * 900),
                    RawToolResult("call-b", "b" * 900),
                )
            )

            self.assertEqual([len(message.content) for message in messages], [500, 500])
            self.assertTrue(
                all(
                    message.preview is not None
                    and message.preview.reason == "individual_and_aggregate_limits"
                    for message in messages
                )
            )

    def test_unsafe_call_id_cannot_escape_tool_results_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            message = manager.process(
                (RawToolResult("../../escape", "x" * 1_000),)
            )[0]

            assert message.preview is not None
            self.assertEqual(message.tool_call_id, "../../escape")
            self.assertEqual(
                message.preview.persisted_path.parent,
                Path(tmp).resolve() / "session" / "tool-results",
            )
            self.assertNotIn("/", message.preview.persisted_path.name)
            self.assertEqual(message.preview.persisted_path.read_text(), "x" * 1_000)

    def test_too_small_preview_budget_fails_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ToolResultManager(
                persist_directory=root,
                individual_limit=10,
                aggregate_limit=10,
                session_id="session",
            )

            with self.assertRaises(ToolResultBudgetError):
                manager.process((RawToolResult("call-1", "x" * 20),))

            self.assertEqual(list(root.rglob("*")), [])

    def test_persistence_failure_does_not_return_a_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / "blocked"
            blocked.write_text("not a directory")
            manager = ToolResultManager(
                persist_directory=blocked,
                individual_limit=500,
                aggregate_limit=800,
                session_id="session",
            )

            with self.assertRaises(ToolResultPersistenceError):
                manager.process((RawToolResult("call-1", "x" * 1_000),))

    def test_rotate_session_changes_future_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ToolResultManager(
                persist_directory=Path(tmp),
                individual_limit=500,
                aggregate_limit=800,
                session_id="first",
            )
            message = manager.process(
                (RawToolResult("call-1", "x" * 1_000),)
            )[0]
            assert message.preview is not None
            old_path = message.preview.persisted_path

            new_session = manager.rotate_session()

            self.assertNotEqual(new_session, "first")
            self.assertEqual(manager.session_id, new_session)
            self.assertEqual(len(new_session), 32)
            self.assertTrue(old_path.exists())

    def test_default_persist_directory_honors_xdg_data_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"XDG_DATA_HOME": tmp}):
                path = default_persist_directory()

        self.assertEqual(path, Path(tmp).resolve() / "alpha-forge")
