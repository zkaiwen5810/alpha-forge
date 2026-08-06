"""Session-agnostic execution of complete tool calls."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from alpha_forge.hooks import HookRegistry, PreToolExecution
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.providers.base import ToolCall
from alpha_forge.tools.base import Tool, ToolExecutionError
from alpha_forge.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ExecutedToolResult:
    call_id: str
    content: str
    status: Literal["success", "error"] = "success"


class ToolCallExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ExecutedToolResult:
        """Execute one complete call without persistence or context editing."""


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        hooks: HookRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.hooks = hooks or HookRegistry()

    async def execute(self, call: ToolCall) -> ExecutedToolResult:
        try:
            arguments = json.loads(
                call.arguments,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to a JSON object")
            tool = self.registry.get(call.name)
            self.registry._validator_for(tool).validate(arguments)
            await self.hooks.dispatch(
                PreToolExecution(
                    call_id=call.call_id,
                    tool_name=tool.name,
                    tool_input=FrozenJsonObject(arguments),
                )
            )
            content = await asyncio.to_thread(_invoke, tool, arguments)
            return ExecutedToolResult(call.call_id, content)
        except Exception as exc:
            return ExecutedToolResult(
                call.call_id,
                f"error: {str(exc) or type(exc).__name__}",
                "error",
            )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _invoke(tool: Tool, arguments: dict[str, Any]) -> str:
    try:
        result = tool.handler(arguments)
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(str(exc) or type(exc).__name__) from exc
    if not isinstance(result, str):
        raise ToolExecutionError(
            f"tool {tool.name!r} returned {type(result).__name__}, expected str"
        )
    return result


__all__ = ["ExecutedToolResult", "ToolCallExecutor", "ToolExecutor"]
