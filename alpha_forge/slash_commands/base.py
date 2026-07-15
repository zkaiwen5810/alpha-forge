"""Slash-command contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config


Print = Callable[[str], None]
StartNewSession = Callable[[], None]


@dataclass(frozen=True)
class CommandContext:
    config: Config
    chat: ChatClient
    print_text: Print
    start_new_session: StartNewSession


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    exit_requested: bool = False


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[[CommandContext], CommandResult]
