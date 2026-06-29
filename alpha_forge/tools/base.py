"""Core tool definition and execution errors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

ToolFunction = Callable[[Mapping[str, Any]], str]
InputValidator = Callable[[Mapping[str, Any]], None]


class ToolError(Exception):
    """Base exception for tool lookup and execution failures."""


class ToolNotFoundError(ToolError):
    """Raised when a registry cannot resolve a tool name or alias."""


class ToolExecutionError(ToolError):
    """Raised when a tool cannot complete a call."""


@dataclass(frozen=True)
class Tool:
    """Application-facing definition for a callable model tool.

    ``description`` is intended for people inspecting the registry, while
    ``prompt`` is sent to the model as the OpenAI function description.
    ``is_mcp`` and ``validate_input`` are extension points retained for future
    MCP dispatch and custom input validation.
    """

    name: str
    function: ToolFunction
    description: str
    prompt: str
    input_schema: Mapping[str, Any]
    aliases: tuple[str, ...] = field(default_factory=tuple)
    is_mcp: bool = False
    validate_input: InputValidator | None = None

    def to_openai(self) -> dict[str, Any]:
        """Return this tool in Chat Completions function-tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.prompt,
                "parameters": dict(self.input_schema),
            },
        }
