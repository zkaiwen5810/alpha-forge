"""Completed OpenAI Chat Completions wire-message values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from alpha_forge.models import ToolCall


class Message(ABC):
    """Behavioral base for one Chat Completions message.

    Role-specific data intentionally lives only on concrete dataclasses. In
    particular, declaring ``content: str | None`` here would make subclasses
    with required string content perform an unsafe invariant field override.
    """

    @abstractmethod
    def to_openai(self) -> dict[str, Any]:
        """Serialize exactly one OpenAI-compatible message."""


@dataclass(frozen=True, slots=True)
class SystemMessage(Message):
    content: str
    role: ClassVar[Literal["system"]] = "system"

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class UserMessage(Message):
    content: str
    role: ClassVar[Literal["user"]] = "user"

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class AssistantMessage(Message):
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None
    refusal: str | None = None
    output_id: str | None = None
    role: ClassVar[Literal["assistant"]] = "assistant"

    def to_openai(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]
        if self.refusal is not None:
            message["refusal"] = self.refusal
        return message


@dataclass(frozen=True, slots=True)
class ToolMessage(Message):
    content: str
    tool_call_id: str
    failed: bool = False
    result_id: str | None = None
    raw: bool = False
    role: ClassVar[Literal["tool"]] = "tool"

    def to_openai(self) -> dict[str, Any]:
        # ``failed`` is application metadata, not a Chat Completions field.
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }


__all__ = [
    "AssistantMessage",
    "Message",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "UserMessage",
]
