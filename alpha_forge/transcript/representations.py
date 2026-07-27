"""Deterministic rendering for schema-v1 tool-result representations."""

from __future__ import annotations

import json

from alpha_forge.transcript.events import (
    HeadTailPreview,
    OriginalRepresentation,
    ToolPreviewReason,
    ToolResultRepresentation,
)


def preview_frame(
    result_event_id: str,
    original_chars: int,
    reason: ToolPreviewReason,
) -> tuple[str, str, str]:
    return (
        (
            "[alpha-forge tool-result-preview]\n"
            "truncated: true\n"
            f"reason: {reason}\n"
            f"original_chars: {original_chars}\n"
            f"result_event_id: {json.dumps(result_event_id)}\n"
            "--- preview head ---\n"
        ),
        "\n--- content omitted ---\n",
        "\n--- preview tail ---\n",
    )


def validate_tool_result_representation(
    *,
    result_event_id: str,
    content: str,
    representation: ToolResultRepresentation,
) -> None:
    if isinstance(representation, OriginalRepresentation):
        return
    if not isinstance(representation, HeadTailPreview):
        raise ValueError("unsupported tool-result representation")
    if representation.renderer != "tool_result_preview":
        raise ValueError("unsupported tool-result renderer")
    if representation.renderer_version != 1:
        raise ValueError("unsupported tool-result renderer version")
    if representation.reason not in (
        "individual_limit",
        "aggregate_limit",
        "individual_and_aggregate_limits",
    ):
        raise ValueError("invalid tool-result preview reason")
    values = (
        representation.original_chars,
        representation.rendered_chars,
        representation.head_chars,
        representation.tail_chars,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("head-tail sizes must be non-negative integers")
    if representation.original_chars != len(content):
        raise ValueError("head-tail original length does not match tool result")
    if representation.rendered_chars >= representation.original_chars:
        raise ValueError("head-tail preview must be shorter than its result")
    prefix, middle, suffix = preview_frame(
        result_event_id,
        representation.original_chars,
        representation.reason,
    )
    expected = (
        len(prefix)
        + len(middle)
        + len(suffix)
        + representation.head_chars
        + representation.tail_chars
    )
    if representation.rendered_chars != expected:
        raise ValueError(
            "head-tail excerpts do not reproduce the declared rendered size"
        )


def render_tool_result(
    result_event_id: str,
    content: str,
    representation: ToolResultRepresentation,
) -> str:
    validate_tool_result_representation(
        result_event_id=result_event_id,
        content=content,
        representation=representation,
    )
    if isinstance(representation, OriginalRepresentation):
        return content
    prefix, middle, suffix = preview_frame(
        result_event_id,
        representation.original_chars,
        representation.reason,
    )
    return (
        prefix
        + content[: representation.head_chars]
        + middle
        + suffix
        + (
            content[-representation.tail_chars :]
            if representation.tail_chars
            else ""
        )
    )


__all__ = [
    "preview_frame",
    "render_tool_result",
    "validate_tool_result_representation",
]
