"""Slash-command contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config
from alpha_forge.conversation import Conversation


Print = Callable[[str], None]


@dataclass(frozen=True)
class CommandContext:
    config: Config
    conversation: Conversation
    chat: ChatClient
    print_text: Print


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    exit_requested: bool = False


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[[CommandContext], CommandResult]
