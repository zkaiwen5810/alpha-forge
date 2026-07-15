"""Explicit OpenAI-compatible message history for multi-turn chat."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

PreviewReason = Literal[
    "individual_limit",
    "aggregate_limit",
    "individual_and_aggregate_limits",
]


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
class ToolResultPreview:
    """Metadata for a tool result whose full content was persisted."""

    persisted_path: Path
    original_chars: int
    reason: PreviewReason


class Message(ABC):
    """Behavioral base for one Chat Completions message.

    Role-specific data intentionally lives only on concrete dataclasses. In
    particular, declaring ``content: str | None`` here would make subclasses
    with required string content perform an unsafe invariant field override.
    """

    @abstractmethod
    def to_openai(self) -> dict[str, Any]:
        """Serialize exactly one OpenAI-compatible message."""


@dataclass(frozen=True)
class SystemMessage(Message):
    content: str
    role: ClassVar[Literal["system"]] = "system"

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class UserMessage(Message):
    content: str
    role: ClassVar[Literal["user"]] = "user"

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class AssistantMessage(Message):
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    role: ClassVar[Literal["assistant"]] = "assistant"

    def to_openai(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.to_openai() for tool_call in self.tool_calls
            ]
        return message


@dataclass(frozen=True)
class ToolMessage(Message):
    content: str
    tool_call_id: str
    failed: bool = False
    preview: ToolResultPreview | None = None
    role: ClassVar[Literal["tool"]] = "tool"

    def to_openai(self) -> dict[str, Any]:
        # ``failed`` and ``preview`` are application metadata, not fields in
        # the Chat Completions tool-message schema. A preview's visible marker
        # and persisted path are already embedded in ``content`` so the model
        # receives the same bounded value the UI displays.
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }


class Conversation:
    def __init__(self, *, system_prompt: str | None = None) -> None:
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(SystemMessage(system_prompt))

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add_user(self, content: str) -> None:
        self._messages.append(UserMessage(content))

    def add_assistant(
        self,
        content: str | None,
        *,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> None:
        self._messages.append(
            AssistantMessage(content, tool_calls=tool_calls)
        )

    def add_tool(self, message: ToolMessage) -> None:
        self._messages.append(message)

    def clear(self, *, keep_system: bool = True) -> None:
        if keep_system:
            self._messages = [
                message
                for message in self._messages
                if isinstance(message, SystemMessage)
            ]
        else:
            self._messages = []

    def to_openai_messages(self) -> list[dict[str, Any]]:
        return [message.to_openai() for message in self._messages]
