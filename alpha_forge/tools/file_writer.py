"""UTF-8 text-file writer with full and fine-grained operations."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alpha_forge.tools.base import Tool, ToolExecutionError

_OPERATIONS = ("write", "create", "append", "replace")


def _string_argument(
    arguments: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool,
) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        qualification = "" if allow_empty else " non-empty"
        raise ToolExecutionError(f"{name} must be a{qualification} string")
    return value


def _resolve_destination(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError("path must be a non-empty string")
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ToolExecutionError(f"cannot resolve file {value!r}: {exc}") from exc


def _prepare_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolExecutionError(
            f"cannot create parent directory for {path}: {exc}"
        ) from exc
    if path.exists() and not path.is_file():
        raise ToolExecutionError(f"path is not a regular file: {path}")


def _write_exclusive(path: Path, content: str) -> None:
    created = False
    fd: int | None = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            fd = None
            stream.write(content)
    except (OSError, UnicodeError) as exc:
        if created:
            path.unlink(missing_ok=True)
        raise ToolExecutionError(f"cannot create file {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _write_atomic(path: Path, content: str) -> None:
    temp_path: Path | None = None
    fd: int | None = None
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(temp_name)
        os.fchmod(fd, existing_mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            fd = None
            stream.write(content)
        os.replace(temp_path, path)
        temp_path = None
    except (OSError, UnicodeError) as exc:
        raise ToolExecutionError(f"cannot write file {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _append(path: Path, content: str) -> bool:
    created = not path.exists()
    fd: int | None = None
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        with os.fdopen(fd, "a", encoding="utf-8", newline="") as stream:
            fd = None
            stream.write(content)
    except (OSError, UnicodeError) as exc:
        if created:
            path.unlink(missing_ok=True)
        raise ToolExecutionError(f"cannot append to file {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
    return created


def _replace(arguments: Mapping[str, Any], path: Path, content: str) -> int:
    if not path.is_file():
        raise ToolExecutionError(f"file does not exist: {path}")
    old_text = _string_argument(arguments, "old_text", allow_empty=False)
    expected = arguments.get("expected_replacements", 1)
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ToolExecutionError("expected_replacements must be a positive integer")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            original = stream.read()
    except (OSError, UnicodeError) as exc:
        raise ToolExecutionError(f"cannot read file {path}: {exc}") from exc
    replacements = original.count(old_text)
    if replacements != expected:
        raise ToolExecutionError(
            f"expected {expected} replacement(s), found {replacements}; "
            "file was not changed"
        )
    _write_atomic(path, original.replace(old_text, content))
    return replacements


def write_file(arguments: Mapping[str, Any]) -> str:
    """Apply one explicit UTF-8 text-file write operation."""
    path = _resolve_destination(arguments.get("path"))
    operation = arguments.get("operation")
    if not isinstance(operation, str) or operation not in _OPERATIONS:
        choices = ", ".join(_OPERATIONS)
        raise ToolExecutionError(f"operation must be one of: {choices}")
    content = _string_argument(arguments, "content", allow_empty=True)
    if operation != "replace":
        _prepare_parent(path)

    created = not path.exists()
    replacements = 0
    if operation == "create":
        _write_exclusive(path, content)
    elif operation == "write":
        _write_atomic(path, content)
    elif operation == "append":
        created = _append(path, content)
    else:
        created = False
        replacements = _replace(arguments, path, content)

    result: dict[str, object] = {
        "path": str(path),
        "operation": operation,
        "written_chars": len(content),
        "created": created,
    }
    if operation == "replace":
        result["replacements"] = replacements
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


FILE_WRITER_TOOL = Tool(
    name="file_writer",
    aliases=("write_file",),
    is_mcp=False,
    description="Creates or updates a UTF-8 text file.",
    prompt=(
        "Write a UTF-8 text file. Choose one operation: write creates or "
        "overwrites the whole file; create fails if the file exists; append "
        "adds content; replace changes exact old_text only when the number of "
        "matches equals expected_replacements (default 1). Prefer replace for "
        "small, guarded edits instead of rewriting a large file. Parent "
        "directories are created when needed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or working-directory-relative file path.",
            },
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS),
                "description": "Write operation to apply.",
            },
            "content": {
                "type": "string",
                "description": (
                    "Full text for write/create, text to add for append, or "
                    "replacement text for replace."
                ),
            },
            "old_text": {
                "type": "string",
                "description": "Non-empty exact text to find for replace.",
            },
            "expected_replacements": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": (
                    "Required exact match count for replace; mismatch leaves "
                    "the file unchanged."
                ),
            },
        },
        "required": ["path", "operation", "content"],
        "additionalProperties": False,
    },
    function=write_file,
)


__all__ = ["FILE_WRITER_TOOL", "write_file"]
