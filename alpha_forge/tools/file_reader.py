"""Bounded UTF-8 text-file reader."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alpha_forge.prompt_editor import MAX_TOOL_RESULT_CHARS
from alpha_forge.tools.base import Tool, ToolExecutionError

# Keep enough room for range metadata and for several ordinary reads to share
# the aggregate tool-batch budget. These values track the central result policy.
DEFAULT_FILE_READ_CHARS = MAX_TOOL_RESULT_CHARS // 2
MAX_FILE_READ_CHARS = MAX_TOOL_RESULT_CHARS * 3 // 4
_IO_CHUNK_CHARS = 64 * 1024


def _integer_argument(
    arguments: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolExecutionError(f"{name} must be an integer")
    if value < minimum:
        raise ToolExecutionError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ToolExecutionError(f"{name} cannot exceed {maximum}")
    return value


def _resolve_file(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError("path must be a non-empty string")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolExecutionError(f"cannot resolve file {value!r}: {exc}") from exc
    if not path.is_file():
        raise ToolExecutionError(f"path is not a regular file: {path}")
    return path


def _read_segment(path: Path, *, offset: int, limit: int) -> tuple[str, bool]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            skipped = 0
            while skipped < offset:
                chunk = stream.read(min(_IO_CHUNK_CHARS, offset - skipped))
                if not chunk:
                    raise ToolExecutionError(
                        f"offset {offset} exceeds file length {skipped}"
                    )
                skipped += len(chunk)

            content = stream.read(limit)
            has_more = bool(stream.read(1))
    except ToolExecutionError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ToolExecutionError(f"cannot read file {path}: {exc}") from exc
    return content, has_more


def _format_result(
    path: Path,
    *,
    offset: int,
    content: str,
    has_more: bool,
) -> str:
    """Build a self-describing result that always fits one result budget."""
    while True:
        next_offset = offset + len(content) if has_more else None
        header = (
            "[alpha-forge file-read]\n"
            f"path: {json.dumps(str(path), ensure_ascii=False)}\n"
            f"offset: {offset}\n"
            f"returned_chars: {len(content)}\n"
            f"next_offset: {next_offset if next_offset is not None else 'null'}\n"
            f"eof: {'false' if has_more else 'true'}\n"
            "--- content ---\n"
        )
        result = header + content
        overflow = len(result) - MAX_TOOL_RESULT_CHARS
        if overflow <= 0:
            return result
        if overflow >= len(content):
            raise ToolExecutionError(
                "file path metadata is too large for the tool-result budget"
            )
        content = content[:-overflow]
        has_more = True


def read_file(arguments: Mapping[str, Any]) -> str:
    """Read one bounded character range from a UTF-8 text file."""
    path = _resolve_file(arguments.get("path"))
    offset = _integer_argument(
        arguments,
        "offset",
        default=0,
        minimum=0,
    )
    limit = _integer_argument(
        arguments,
        "limit",
        default=DEFAULT_FILE_READ_CHARS,
        minimum=1,
        maximum=MAX_FILE_READ_CHARS,
    )
    content, has_more = _read_segment(path, offset=offset, limit=limit)
    return _format_result(
        path,
        offset=offset,
        content=content,
        has_more=has_more,
    )


FILE_READER_TOOL = Tool(
    name="file_reader",
    aliases=("read_file",),
    is_mcp=False,
    description="Reads a bounded character range from a UTF-8 text file.",
    prompt=(
        "Read a UTF-8 text file in bounded character ranges. Use offset 0 for "
        "the first range, then use the returned next_offset until eof is true. "
        "Offsets count Unicode characters, not bytes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or working-directory-relative file path.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Zero-based Unicode character offset.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_FILE_READ_CHARS,
                "default": DEFAULT_FILE_READ_CHARS,
                "description": "Maximum Unicode characters to return.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    function=read_file,
)


__all__ = [
    "DEFAULT_FILE_READ_CHARS",
    "FILE_READER_TOOL",
    "MAX_FILE_READ_CHARS",
    "read_file",
]
