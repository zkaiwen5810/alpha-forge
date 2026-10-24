import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from alpha_forge_langchain import AgentApplication
from alpha_forge_langchain.defaults import database_path


class CliTests(unittest.TestCase):
    def setUp(self):
        """Override TestCase.setUp to isolate the database and subprocess environment."""
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "cli.sqlite3"
        self.environment = os.environ.copy()
        self.environment.pop("OPENAI_API_KEY", None)
        self.environment["LANGSMITH_TRACING"] = "false"
        self.environment["LANGCHAIN_TRACING_V2"] = "false"

    def tearDown(self):
        """Override TestCase.tearDown to remove temporary files."""
        self.temporary.cleanup()

    def invoke(self, *arguments, offline=False):
        if offline:
            entry = [str(Path(__file__).with_name("cli_fixture.py"))]
        else:
            entry = ["-m", "alpha_forge_langchain"]
        return subprocess.run(
            [sys.executable, *entry, "--database", str(self.database), *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_commands_across_processes(self):
        created = self.invoke("new")
        self.assertEqual(created.returncode, 0, created.stderr)
        conversation = created.stdout.strip()
        UUID(conversation)
        self.assertEqual(created.stderr, "")
        history = self.invoke("history", conversation)
        self.assertEqual(history.returncode, 0, history.stderr)
        self.assertEqual(json.loads(history.stdout), [])
        for prompt in ("first", "second"):
            sent = self.invoke("send", conversation, prompt, offline=True)
            self.assertEqual(sent.returncode, 0, sent.stderr)
            self.assertEqual(sent.stdout, "offline answer\n")
        history = self.invoke("history", conversation)
        self.assertEqual(history.returncode, 0, history.stderr)
        self.assertEqual(
            [m["content"] for m in json.loads(history.stdout)],
            [
                "first",
                "offline answer",
                "second",
                "offline answer",
            ],
        )

    def test_errors_and_no_secret_flag(self):
        result = self.invoke("history", "unknown")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("unknown thread", result.stderr)
        conversation = self.invoke("new").stdout.strip()
        result = self.invoke("send", conversation, "hello")
        self.assertEqual(result.returncode, 1)
        self.assertIn("OPENAI_API_KEY", result.stderr)
        result = self.invoke("send", conversation, " ")
        self.assertEqual(result.returncode, 1)
        self.assertIn("non-empty", result.stderr)
        result = self.invoke("--api-key", "not-a-key", "new")
        self.assertEqual(result.returncode, 2)

    def test_interactive_commands_require_a_terminal(self):
        for arguments in [(), ("chat",), ("chat", "existing")]:
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertEqual(result.returncode, 1)
                self.assertIn("requires a terminal", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertFalse(self.database.exists())

    def test_abrupt_process_exit_restores_last_completed_turn(self):
        crashed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("crash_fixture.py")),
                str(self.database),
            ],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(crashed.returncode, 23, crashed.stderr)
        conversation = crashed.stdout.strip()
        restored = self.invoke("history", conversation)
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertIn("Unfinished work excluded", restored.stderr)
        self.assertEqual(
            [m["content"] for m in json.loads(restored.stdout)], ["first", "baseline"]
        )
        sent = self.invoke("send", conversation, "fresh", offline=True)
        self.assertEqual(sent.returncode, 0, sent.stderr)
        history = self.invoke("history", conversation)
        self.assertEqual(
            [m["content"] for m in json.loads(history.stdout)],
            [
                "first",
                "baseline",
                "fresh",
                "offline answer",
            ],
        )

    def test_busy_database_across_processes(self):
        import asyncio

        async def hold():
            async with AgentApplication(self.database):
                result = await asyncio.to_thread(self.invoke, "new")
                self.assertEqual(result.returncode, 1)
                self.assertIn("database is in use", result.stderr)

        asyncio.run(hold())
        self.assertEqual(self.invoke("new").returncode, 0)

    def test_unfinished_history_diagnostic(self):
        import asyncio

        async def prepare():
            async with AgentApplication(self.database) as runtime:
                conversation = await runtime.threads.acreate()
                assert runtime._persistence._threads is not None
                await runtime._persistence._threads.abegin(
                    await runtime.threads.aget(conversation)
                )
                return conversation

        conversation = asyncio.run(prepare())
        result = self.invoke("history", conversation)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])
        self.assertIn("Unfinished work excluded", result.stderr)

    def test_xdg_default_and_code_settings(self):
        from unittest.mock import patch

        from alpha_forge_langchain import defaults

        with patch.dict(os.environ, {"XDG_DATA_HOME": self.temporary.name}):
            self.assertEqual(
                database_path(),
                Path(self.temporary.name)
                / "alpha-forge-langchain/conversations.sqlite3",
            )
        self.assertEqual(defaults.BASE_URL, "http://litellm:4000/v1")


if __name__ == "__main__":
    unittest.main()
