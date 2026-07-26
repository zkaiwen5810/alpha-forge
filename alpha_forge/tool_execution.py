"""Session-agnostic execution of complete model tool calls."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from alpha_forge.models import ToolCall
from alpha_forge.tools import ToolError, ToolRegistry


@dataclass(frozen=True, slots=True)
class ExecutedToolResult:
    """Raw result returned by the application-side execution boundary."""

    call_id: str
    content: str
    failed: bool = False


class ToolCallExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ExecutedToolResult:
        """Execute one complete call without editing or persistence."""


class ToolExecutor:
    """Run synchronous registry tools without blocking the event loop."""

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
            return ExecutedToolResult(call.id, content)
        except (json.JSONDecodeError, ValueError, ToolError) as exc:
            return ExecutedToolResult(call.id, f"error: {exc}", failed=True)


__all__ = ["ExecutedToolResult", "ToolCallExecutor", "ToolExecutor"]
