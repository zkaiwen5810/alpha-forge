"""Tool registration, lookup, schema generation, and execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alpha_forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolNotFoundError,
)


class ToolRegistry:
    """Registry with canonical-name and alias lookup."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._lookup: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool, rejecting duplicate names and aliases."""
        identifiers = (tool.name, *tool.aliases)
        if not tool.name:
            raise ValueError("tool name cannot be empty")
        if any(not identifier for identifier in identifiers):
            raise ValueError(f"tool {tool.name!r} has an empty alias")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"tool {tool.name!r} repeats a name or alias")

        collisions = [
            identifier for identifier in identifiers if identifier in self._lookup
        ]
        if collisions:
            names = ", ".join(repr(identifier) for identifier in collisions)
            raise ValueError(f"tool name or alias already registered: {names}")

        self._tools[tool.name] = tool
        for identifier in identifiers:
            self._lookup[identifier] = tool

    def get(self, name_or_alias: str) -> Tool:
        """Resolve a canonical tool name or alias."""
        try:
            return self._lookup[name_or_alias]
        except KeyError as exc:
            raise ToolNotFoundError(f"unknown tool: {name_or_alias}") from exc

    def definitions(self) -> list[dict[str, Any]]:
        """Return model-facing OpenAI function definitions."""
        return [tool.to_openai() for tool in self._tools.values()]

    def execute(self, name_or_alias: str, arguments: Mapping[str, Any]) -> str:
        """Execute one tool call and normalize its result to text.

        Custom ``validate_input`` and MCP dispatch are deliberately not invoked
        yet; their metadata is part of the first-version definition so those
        behaviors can be added without changing the public tool contract.
        """
        tool = self.get(name_or_alias)
        try:
            result = tool.function(arguments)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(str(exc)) from exc
        if not isinstance(result, str):
            raise ToolExecutionError(
                f"tool {tool.name!r} returned {type(result).__name__}, expected str"
            )
        return result
