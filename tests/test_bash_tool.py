import asyncio
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from alpha_forge.model_messages import ToolCall
from alpha_forge.models import RawToolResult
from alpha_forge.prompt_editor import ToolResultPromptEditor
from alpha_forge.session import Session
from alpha_forge.streaming import ModelResponse
from alpha_forge.tool_execution import ToolExecutor
from alpha_forge.tools import (
    DEFAULT_BASH_TIMEOUT_SECONDS,
    MAX_BASH_TIMEOUT_SECONDS,
    MIN_BASH_TIMEOUT_SECONDS,
    ToolExecutionError,
    load_builtin_tools,
    run_bash,
)


def _section(result: str, name: str, next_name: str | None = None) -> str:
    content = result.split(f"--- {name} ---\n", maxsplit=1)[1]
    if next_name is not None:
        content = content.split(f"--- {next_name} ---\n", maxsplit=1)[0]
    return content.removesuffix("\n")


class BashToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_builtin_tools()

    def test_builtin_loader_exposes_bash_alias_and_schema(self) -> None:
        tool = self.registry.get("bash")

        self.assertIs(self.registry.get("shell"), tool)
        self.assertEqual(tool.name, "bash")
        self.assertEqual(
            tool.input_schema,
            {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Non-empty Bash command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Optional absolute or process-relative working directory."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": MIN_BASH_TIMEOUT_SECONDS,
                        "maximum": MAX_BASH_TIMEOUT_SECONDS,
                        "default": DEFAULT_BASH_TIMEOUT_SECONDS,
                        "description": "Maximum execution time in seconds.",
                    },
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        )

    def test_runs_pwd_in_selected_directory_and_supports_bash_pipelines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            result = self.registry.execute(
                "bash",
                {
                    "cmd": (
                        "pwd; printf alpha | tr '[:lower:]' '[:upper:]'; "
                        "printf warning >&2"
                    ),
                    "cwd": str(cwd),
                },
            )

        self.assertIn(f"cwd: {json.dumps(str(cwd))}", result)
        self.assertIn("timeout_seconds: 30", result)
        self.assertIn("exit_code: 0", result)
        self.assertIn("timed_out: false", result)
        self.assertEqual(_section(result, "stdout", "stderr"), f"{cwd}\nALPHA")
        self.assertEqual(_section(result, "stderr"), "warning")

    def test_defaults_to_process_working_directory(self) -> None:
        result = self.registry.execute("shell", {"cmd": "pwd"})

        self.assertEqual(_section(result, "stdout", "stderr"), str(Path.cwd()))

    def test_sanitizes_secret_and_bash_startup_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            startup = Path(tmp) / "bash-env"
            startup.write_text("printf startup-file >&2\n", encoding="utf-8")
            environment = {
                "ALPHA_TEST_VISIBLE": "visible",
                "ALPHA_TEST_TOKEN": "hidden-token",
                "DATABASE_PASSWORD": "hidden-password",
                "OPENAI_API_KEY": "hidden-key",
                "BASH_ENV": str(startup),
            }
            with patch.dict(os.environ, environment):
                result = self.registry.execute(
                    "bash",
                    {
                        "cmd": (
                            "printf '%s|%s|%s|%s' "
                            '"${ALPHA_TEST_VISIBLE-unset}" '
                            '"${ALPHA_TEST_TOKEN-unset}" '
                            '"${DATABASE_PASSWORD-unset}" '
                            '"${OPENAI_API_KEY-unset}"'
                        )
                    },
                )

        self.assertEqual(
            _section(result, "stdout", "stderr"),
            "visible|unset|unset|unset",
        )
        self.assertEqual(_section(result, "stderr"), "")

    def test_nonzero_exit_becomes_failed_executor_result(self) -> None:
        executed = asyncio.run(
            ToolExecutor(self.registry).execute(
                ToolCall(
                    "call-bash",
                    "bash",
                    json.dumps(
                        {"cmd": "printf output; printf problem >&2; exit 7"}
                    ),
                )
            )
        )

        self.assertTrue(executed.failed)
        self.assertTrue(executed.content.startswith("error: [alpha-forge bash]"))
        self.assertIn("exit_code: 7", executed.content)
        self.assertIn("--- stdout ---\noutput", executed.content)
        self.assertIn("--- stderr ---\nproblem", executed.content)

    def test_timeout_returns_failure_without_waiting_for_command(self) -> None:
        started = time.monotonic()

        with self.assertRaisesRegex(ToolExecutionError, "timed_out: true"):
            self.registry.execute(
                "bash",
                {
                    "cmd": "sleep 5 &",
                    "timeout": MIN_BASH_TIMEOUT_SECONDS,
                },
            )

        self.assertLess(time.monotonic() - started, 2)

    def test_timeout_kills_the_process_group_and_collects_output(self) -> None:
        process = Mock()
        process.pid = 1234
        process.returncode = -signal.SIGKILL
        process.poll.return_value = None
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("bash", MIN_BASH_TIMEOUT_SECONDS),
            ("partial output", "partial error"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "alpha_forge.tools.bash.shutil.which",
                    return_value="/bin/bash",
                ),
                patch(
                    "alpha_forge.tools.bash.subprocess.Popen",
                    return_value=process,
                ) as popen,
                patch("alpha_forge.tools.bash.os.killpg") as killpg,
            ):
                with self.assertRaises(ToolExecutionError) as raised:
                    run_bash(
                        {
                            "cmd": "long command",
                            "cwd": tmp,
                            "timeout": MIN_BASH_TIMEOUT_SECONDS,
                        }
                    )

        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(1234, signal.SIGKILL)
        self.assertIn("partial output", str(raised.exception))
        self.assertIn("partial error", str(raised.exception))
        self.assertIn("timed_out: true", str(raised.exception))

    def test_oversized_output_is_self_contained_and_readable_by_range(self) -> None:
        result = self.registry.execute(
            "bash",
            {"cmd": "printf 'x%.0s' {1..20000}"},
        )
        stdout_offset = result.index("--- stdout ---\n") + len("--- stdout ---\n")
        session = Session()
        turn_id = session.submit_user("run")
        output = session.add_assistant_message(
            turn_id=turn_id,
            response=ModelResponse(
                None,
                (ToolCall("call-bash", "bash", "{}"),),
            ),
        )
        raw = session.add_tool_result(
            output_id=output.output_id,
            call_id="call-bash",
            content=result,
            failed=False,
        )
        editor = ToolResultPromptEditor(
            individual_limit=500,
            aggregate_limit=800,
        )
        session.add_prompt_edit(
            output_id=output.output_id,
            edit=editor.edit(
                (
                    RawToolResult(
                        raw.result_id,
                        raw.call_id,
                        raw.content,
                        raw.failed,
                    ),
                )
            ),
        )

        read_result = session.read_tool_result(
            raw.result_id,
            offset=stdout_offset,
            limit=25,
        )

        self.assertTrue(read_result.startswith("x" * 25))
        self.assertEqual(session.transcript.result(raw.result_id).content, result)

    def test_rejects_invalid_arguments_and_missing_bash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "file.txt"
            file_path.write_text("not a directory", encoding="utf-8")
            cases = [
                {},
                {"cmd": "  "},
                {"cmd": "pwd", "cwd": str(root / "missing")},
                {"cmd": "pwd", "cwd": str(file_path)},
                {"cmd": "pwd", "timeout": True},
                {"cmd": "pwd", "timeout": float("nan")},
                {"cmd": "pwd", "timeout": MIN_BASH_TIMEOUT_SECONDS / 2},
                {"cmd": "pwd", "timeout": MAX_BASH_TIMEOUT_SECONDS + 1},
            ]
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ToolExecutionError):
                        self.registry.execute("bash", arguments)

        with patch("alpha_forge.tools.bash.shutil.which", return_value=None):
            with self.assertRaisesRegex(ToolExecutionError, "not found"):
                run_bash({"cmd": "pwd"})


if __name__ == "__main__":
    unittest.main()
