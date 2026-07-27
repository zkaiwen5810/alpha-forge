"""Session-agnostic execution of complete tool calls."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from alpha_forge.providers.base import ToolCall
from alpha_forge.tools.base import ToolError
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
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(self, call: ToolCall) -> ExecutedToolResult:
        return await asyncio.to_thread(self._execute_sync, call)

    def _execute_sync(self, call: ToolCall) -> ExecutedToolResult:
        try:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to a JSON object")
            content = self.registry.execute(call.name, arguments)
            return ExecutedToolResult(call.call_id, content)
        except (json.JSONDecodeError, ValueError, ToolError) as exc:
            return ExecutedToolResult(
                call.call_id,
                f"error: {exc}",
                "error",
            )


__all__ = ["ExecutedToolResult", "ToolCallExecutor", "ToolExecutor"]
