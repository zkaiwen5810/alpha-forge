"""Built-in slash commands."""

from alpha_forge.slash_commands.base import CommandOutcome, ParsedCommand
from alpha_forge.slash_commands.builtin import SLASH_COMMANDS, SlashCommandHandler
from alpha_forge.slash_commands.completion import SlashCommandCompleter

__all__ = [
    "SLASH_COMMANDS",
    "CommandOutcome",
    "ParsedCommand",
    "SlashCommandCompleter",
    "SlashCommandHandler",
]
