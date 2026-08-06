import json
import tempfile
import unittest
from pathlib import Path

from alpha_forge.context import MAX_TOOL_RESULT_CHARS
from alpha_forge.tools import (
    MAX_FILE_READ_CHARS,
    ToolExecutionError,
    load_builtin_tools,
    read_file,
    write_file,
)


def _read_content(result: str) -> str:
    return result.split("--- content ---\n", maxsplit=1)[1]


class FileReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_builtin_tools()

    def test_builtin_loader_exposes_reader_and_writer_aliases(self) -> None:
        self.assertEqual(self.registry.get("file_reader").name, "file_reader")
        self.assertEqual(self.registry.get("read_file").name, "file_reader")
        self.assertEqual(self.registry.get("file_writer").name, "file_writer")
        self.assertEqual(self.registry.get("write_file").name, "file_writer")
        self.assertEqual(
            [spec.name for spec in self.registry.specs()],
            ["calculator", "file_reader", "file_writer", "bash"],
        )

    def test_reads_unicode_character_ranges_with_continuation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unicode.txt"
            path.write_text("aβc\r\ndef", encoding="utf-8", newline="")

            first = read_file({"path": str(path), "offset": 1, "limit": 4})
            second = read_file({"path": str(path), "offset": 5, "limit": 20})

        self.assertIn("offset: 1", first)
        self.assertIn("returned_chars: 4", first)
        self.assertIn("next_offset: 5", first)
        self.assertIn("eof: false", first)
        self.assertEqual(_read_content(first), "βc\r\n")
        self.assertIn("next_offset: null", second)
        self.assertIn("eof: true", second)
        self.assertEqual(_read_content(second), "def")

    def test_read_result_cannot_exceed_individual_result_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.txt"
            path.write_text("x" * (MAX_FILE_READ_CHARS + 1), encoding="utf-8")

            result = read_file({"path": str(path), "limit": MAX_FILE_READ_CHARS})

        self.assertLessEqual(len(result), MAX_TOOL_RESULT_CHARS)
        self.assertEqual(len(_read_content(result)), MAX_FILE_READ_CHARS)
        self.assertIn(f"next_offset: {MAX_FILE_READ_CHARS}", result)
        self.assertIn("eof: false", result)

    def test_rejects_invalid_ranges_missing_files_and_non_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "short.txt"
            path.write_text("short", encoding="utf-8")
            binary = root / "binary.bin"
            binary.write_bytes(b"\xff")

            cases = [
                {"path": str(path), "offset": -1},
                {"path": str(path), "limit": 0},
                {"path": str(path), "limit": MAX_FILE_READ_CHARS + 1},
                {"path": str(path), "offset": 6},
                {"path": str(root / "missing")},
                {"path": str(root)},
                {"path": str(binary)},
            ]
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ToolExecutionError):
                        read_file(arguments)


class FileWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_builtin_tools()

    def test_create_append_replace_and_write_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "document.txt"

            created = json.loads(
                write_file(
                    {
                        "path": str(path),
                        "operation": "create",
                        "content": "first\r\nsecond\r\n",
                    },
                )
            )
            write_file(
                {
                    "path": str(path),
                    "operation": "append",
                    "content": "third\r\n",
                },
            )
            replaced = json.loads(
                write_file(
                    {
                        "path": str(path),
                        "operation": "replace",
                        "old_text": "second\r\n",
                        "content": "2nd\r\n",
                    },
                )
            )

            self.assertEqual(
                path.read_bytes(),
                b"first\r\n2nd\r\nthird\r\n",
            )
            self.assertTrue(created["created"])
            self.assertEqual(replaced["replacements"], 1)

            written = json.loads(
                write_file(
                    {
                        "path": str(path),
                        "operation": "write",
                        "content": "replacement",
                    },
                )
            )

            self.assertEqual(path.read_text(encoding="utf-8"), "replacement")
            self.assertFalse(written["created"])

    def test_create_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.txt"
            path.write_text("keep", encoding="utf-8")

            with self.assertRaises(ToolExecutionError):
                write_file(
                    {
                        "path": str(path),
                        "operation": "create",
                        "content": "replace",
                    },
                )

            self.assertEqual(path.read_text(encoding="utf-8"), "keep")

    def test_guarded_replace_mismatch_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repeated.txt"
            path.write_text("same same", encoding="utf-8")

            with self.assertRaisesRegex(ToolExecutionError, "found 2"):
                write_file(
                    {
                        "path": str(path),
                        "operation": "replace",
                        "old_text": "same",
                        "content": "new",
                    },
                )

            self.assertEqual(path.read_text(encoding="utf-8"), "same same")

            write_file(
                {
                    "path": str(path),
                    "operation": "replace",
                    "old_text": "same",
                    "content": "new",
                    "expected_replacements": 2,
                },
            )
            self.assertEqual(path.read_text(encoding="utf-8"), "new new")

    def test_replace_of_missing_file_does_not_create_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "missing" / "nested"
            path = parent / "file.txt"

            with self.assertRaisesRegex(ToolExecutionError, "does not exist"):
                write_file(
                    {
                        "path": str(path),
                        "operation": "replace",
                        "old_text": "old",
                        "content": "new",
                    },
                )

            self.assertFalse(parent.exists())

    def test_rejects_invalid_operation_and_replace_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.txt"
            path.write_text("content", encoding="utf-8")
            cases = [
                {"path": str(path), "operation": "delete", "content": ""},
                {"path": str(path), "operation": "replace", "content": ""},
                {
                    "path": str(path),
                    "operation": "replace",
                    "content": "new",
                    "old_text": "content",
                    "expected_replacements": 0,
                },
            ]
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ToolExecutionError):
                        write_file(arguments)


if __name__ == "__main__":
    unittest.main()
