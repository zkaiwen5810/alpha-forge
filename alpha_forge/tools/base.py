"""Provider-neutral tool definitions and execution errors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.tools.validation import check_input_schema

ToolHandler = Callable[[Mapping[str, Any]], str]


class ToolError(Exception):
    """Base exception for tool lookup and execution failures."""


class ToolNotFoundError(ToolError):
    """Raised when a registry cannot resolve a tool name or alias."""


class ToolExecutionError(ToolError):
    """Raised when a tool cannot complete a call."""


@dataclass(frozen=True, slots=True, init=False)
class ToolSpec:
    """Provider-neutral metadata exposed to a model."""

    name: str
    description: str
    input_schema: FrozenJsonObject

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Mapping[str, Any],
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(
            self,
            "input_schema",
            FrozenJsonObject(input_schema),
        )
        check_input_schema(self.input_schema)


@dataclass(frozen=True, slots=True, init=False)
class Tool:
    """A provider-neutral specification paired with its application handler."""

    spec: ToolSpec
    handler: ToolHandler
    display_description: str
    aliases: tuple[str, ...]

    def __init__(
        self,
        *,
        name: str,
        handler: ToolHandler,
        description: str,
        input_schema: Mapping[str, Any],
        aliases: tuple[str, ...] = (),
        display_description: str | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "spec",
            ToolSpec(name, description, input_schema),
        )
        object.__setattr__(self, "handler", handler)
        object.__setattr__(
            self,
            "display_description",
            display_description or description,
        )
        object.__setattr__(self, "aliases", tuple(aliases))

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def input_schema(self) -> Mapping[str, Any]:
        return self.spec.input_schema


__all__ = [
    "Tool",
    "ToolError",
    "ToolExecutionError",
    "ToolHandler",
    "ToolNotFoundError",
    "ToolSpec",
]
