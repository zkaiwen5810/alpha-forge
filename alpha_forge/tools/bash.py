"""Non-interactive Bash command execution."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alpha_forge.tools.base import Tool, ToolExecutionError

DEFAULT_BASH_TIMEOUT_SECONDS = 30.0
MAX_BASH_TIMEOUT_SECONDS = 300.0
MIN_BASH_TIMEOUT_SECONDS = 0.1

_SENSITIVE_ENV_PARTS = frozenset(
    {
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "PASSWD",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    }
)
_UNSAFE_BASH_ENV_NAMES = frozenset({"BASH_ENV", "ENV"})


def _command(arguments: Mapping[str, Any]) -> str:
    value = arguments.get("cmd")
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError("cmd must be a non-empty string")
    return value


def _working_directory(arguments: Mapping[str, Any]) -> Path:
    value = arguments.get("cwd")
    if value is None:
        try:
            return Path.cwd().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ToolExecutionError(
                f"cannot resolve current working directory: {exc}"
            ) from exc
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError("cwd must be a non-empty string when provided")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolExecutionError(
            f"cannot resolve working directory {value!r}: {exc}"
        ) from exc
    if not path.is_dir():
        raise ToolExecutionError(f"cwd is not a directory: {path}")
    return path


def _timeout(arguments: Mapping[str, Any]) -> float:
    value = arguments.get("timeout", DEFAULT_BASH_TIMEOUT_SECONDS)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolExecutionError("timeout must be a number")
    timeout = float(value)
    if not math.isfinite(timeout):
        raise ToolExecutionError("timeout must be finite")
    if timeout < MIN_BASH_TIMEOUT_SECONDS:
        raise ToolExecutionError(
            f"timeout must be at least {MIN_BASH_TIMEOUT_SECONDS:g} seconds"
        )
    if timeout > MAX_BASH_TIMEOUT_SECONDS:
        raise ToolExecutionError(
            f"timeout cannot exceed {MAX_BASH_TIMEOUT_SECONDS:g} seconds"
        )
    return timeout


def _is_sensitive_environment_name(name: str) -> bool:
    if name.upper() in _UNSAFE_BASH_ENV_NAMES:
        return True
    return bool(_SENSITIVE_ENV_PARTS.intersection(name.upper().split("_")))


def _sanitized_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not _is_sensitive_environment_name(name)
    }


def _section(name: str, content: str) -> str:
    section = f"--- {name} ---\n{content}"
    if content and not content.endswith("\n"):
        section += "\n"
    return section


def _format_result(
    *,
    cwd: Path,
    timeout: float,
    exit_code: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
) -> str:
    header = (
        "[alpha-forge bash]\n"
        f"cwd: {json.dumps(str(cwd), ensure_ascii=False)}\n"
        f"timeout_seconds: {timeout:g}\n"
        f"exit_code: {exit_code if exit_code is not None else 'null'}\n"
        f"timed_out: {'true' if timed_out else 'false'}\n"
    )
    return header + _section("stdout", stdout) + _section("stderr", stderr)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        if process.poll() is None:
            process.kill()


def run_bash(arguments: Mapping[str, Any]) -> str:
    """Run one bounded, non-interactive Bash invocation."""
    cmd = _command(arguments)
    cwd = _working_directory(arguments)
    timeout = _timeout(arguments)
    executable = shutil.which("bash")
    if executable is None:
        raise ToolExecutionError("bash executable was not found on PATH")

    try:
        process = subprocess.Popen(
            [executable, "--noprofile", "--norc", "-c", cmd],
            cwd=cwd,
            env=_sanitized_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolExecutionError(f"cannot start bash: {exc}") from exc

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        result = _format_result(
            cwd=cwd,
            timeout=timeout,
            exit_code=process.returncode,
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
        )
        raise ToolExecutionError(result)

    result = _format_result(
        cwd=cwd,
        timeout=timeout,
        exit_code=process.returncode,
        timed_out=False,
        stdout=stdout,
        stderr=stderr,
    )
    if process.returncode != 0:
        raise ToolExecutionError(result)
    return result


BASH_TOOL = Tool(
    name="bash",
    aliases=("shell",),
    display_description="Runs a non-interactive Bash command.",
    description=(
        "Run a command through non-interactive Bash with pipes, redirects, "
        "and command chaining available. Use cwd to select a working "
        "directory and timeout for bounded long-running commands. Each call "
        "uses a fresh shell, so directory and environment changes do not "
        "persist. Nonzero exits and timeouts are failed tool results that "
        "retain stdout and stderr."
    ),
    input_schema={
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
    handler=run_bash,
)


__all__ = [
    "BASH_TOOL",
    "DEFAULT_BASH_TIMEOUT_SECONDS",
    "MAX_BASH_TIMEOUT_SECONDS",
    "MIN_BASH_TIMEOUT_SECONDS",
    "run_bash",
]
