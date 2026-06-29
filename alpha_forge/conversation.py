"""Explicit OpenAI-compatible message history for multi-turn chat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by an assistant message."""

    id: str
    name: str
    arguments: str

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.to_openai() for tool_call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


class Conversation:
    def __init__(self, *, system_prompt: str | None = None) -> None:
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message("system", system_prompt))

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add_user(self, content: str) -> None:
        self._messages.append(Message("user", content))

    def add_assistant(
        self,
        content: str | None,
        *,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> None:
        self._messages.append(
            Message("assistant", content, tool_calls=tool_calls)
        )

    def add_tool(self, content: str, *, tool_call_id: str) -> None:
        self._messages.append(
            Message("tool", content, tool_call_id=tool_call_id)
        )

    def clear(self, *, keep_system: bool = True) -> None:
        if keep_system:
            self._messages = [message for message in self._messages if message.role == "system"]
        else:
            self._messages = []

    def to_openai_messages(self) -> list[dict[str, Any]]:
        return [message.to_openai() for message in self._messages]
