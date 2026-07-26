"""Side-effect-free slash-command contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from alpha_forge.transcript import CommandMessage

CommandAction = Literal["none", "exit", "clear", "resume"]


class ModelCatalog(Protocol):
    def list_models(self) -> list[str]:
        """Return model IDs visible to the configured OpenAI client."""


@dataclass(frozen=True, slots=True)
class CommandContext:
    current_model: str
    model_catalog: ModelCatalog
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    status: Literal["success", "error"] = "success"
    messages: tuple[CommandMessage, ...] = ()
    action: CommandAction = "none"


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[[CommandContext], CommandOutcome]


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    raw: str
    name: str
    arguments: str
    command: SlashCommand | None


class SlashCommandHandler:
    def __init__(self, commands: tuple[SlashCommand, ...]) -> None:
        self.commands = {command.name: command for command in commands}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.commands)

    def parse(self, text: str) -> ParsedCommand:
        parts = text.strip().split(maxsplit=1)
        name = parts[0]
        arguments = parts[1] if len(parts) == 2 else ""
        return ParsedCommand(text, name, arguments, self.commands.get(name))

    @staticmethod
    def execute(
        parsed: ParsedCommand,
        context: CommandContext,
    ) -> CommandOutcome:
        if parsed.command is None:
            return CommandOutcome(
                status="error",
                messages=(
                    CommandMessage(
                        f"unknown command: {parsed.name}",
                        "error",
                    ),
                ),
            )
        return parsed.command.handler(replace(context, arguments=parsed.arguments))


__all__ = [
    "CommandAction",
    "CommandContext",
    "ModelCatalog",
    "CommandOutcome",
    "ParsedCommand",
    "SlashCommand",
    "SlashCommandHandler",
]
