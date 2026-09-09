"""Read and expose bounded pages of raw tool results from one transcript."""

from collections.abc import Mapping

from alpha_forge.context.tool_result_budget import MAX_TOOL_RESULT_CHARS
from alpha_forge.tools.base import Tool
from alpha_forge.transcript.store import TranscriptStore


class ToolResultReader:
    def __init__(self, transcript: TranscriptStore) -> None:
        self._transcript = transcript

    def read(
        self,
        result_event_id: str,
        *,
        offset: int = 0,
        limit: int = MAX_TOOL_RESULT_CHARS,
    ) -> str:
        if not isinstance(result_event_id, str) or not result_event_id:
            raise ValueError("result_event_id must be a non-empty string")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be non-negative")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > MAX_TOOL_RESULT_CHARS
        ):
            raise ValueError(f"limit must be between 1 and {MAX_TOOL_RESULT_CHARS}")
        try:
            _, result = self._transcript.result(result_event_id)
        except KeyError as exc:
            raise ValueError(f"unknown transcript result: {result_event_id}") from exc
        chunk = result.content[offset : offset + limit]
        next_offset = offset + len(chunk)
        eof = next_offset >= len(result.content)
        return (
            f"{chunk}\n"
            "[alpha-forge transcript-result]\n"
            f"result_event_id: {result_event_id}\n"
            f"next_offset: {next_offset}\n"
            f"eof: {str(eof).lower()}"
        )

    def as_tool(self) -> Tool:
        def read(arguments: Mapping[str, object]) -> str:
            result_id = arguments.get("result_event_id")
            if not isinstance(result_id, str) or not result_id:
                raise ValueError("result_event_id must be a non-empty string")
            offset = _integer(arguments, "offset", 0)
            limit = _integer(arguments, "limit", MAX_TOOL_RESULT_CHARS)
            return self.read(result_id, offset=offset, limit=limit)

        return Tool(
            name="tool_result_reader",
            handler=read,
            display_description="Reads a complete result stored in this transcript.",
            description=(
                "Read a raw tool result referenced by result_event_id. Use "
                "offset and limit to page through large results."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "result_event_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_RESULT_CHARS,
                    },
                },
                "required": ["result_event_id"],
                "additionalProperties": False,
            },
        )


def _integer(
    arguments: Mapping[str, object],
    name: str,
    default: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
