"""Explicit message history for multi-turn chat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


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

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message("assistant", content))

    def clear(self, *, keep_system: bool = True) -> None:
        if keep_system:
            self._messages = [message for message in self._messages if message.role == "system"]
        else:
            self._messages = []

    def to_openai_messages(self) -> list[dict[str, str]]:
        return [message.to_openai() for message in self._messages]
