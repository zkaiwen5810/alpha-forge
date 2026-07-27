"""Tool registration and lookup without provider serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alpha_forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSpec,
)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._lookup: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
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
        try:
            return self._lookup[name_or_alias]
        except KeyError as exc:
            raise ToolNotFoundError(f"unknown tool: {name_or_alias}") from exc

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def copy(self) -> ToolRegistry:
        return ToolRegistry(list(self._tools.values()))

    def execute(self, name_or_alias: str, arguments: Mapping[str, Any]) -> str:
        tool = self.get(name_or_alias)
        try:
            result = tool.handler(arguments)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(str(exc)) from exc
        if not isinstance(result, str):
            raise ToolExecutionError(
                f"tool {tool.name!r} returned {type(result).__name__}, expected str"
            )
        return result


__all__ = ["ToolRegistry"]
