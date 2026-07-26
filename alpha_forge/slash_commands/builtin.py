"""Built-in slash-command decisions."""

from __future__ import annotations

from alpha_forge.slash_commands.base import (
    CommandContext,
    CommandOutcome,
    SlashCommand,
)
from alpha_forge.slash_commands.base import (
    SlashCommandHandler as BaseSlashCommandHandler,
)
from alpha_forge.transcript import CommandMessage


def _exit(_context: CommandContext) -> CommandOutcome:
    return CommandOutcome(action="exit")


def _clear(_context: CommandContext) -> CommandOutcome:
    return CommandOutcome(action="clear")


def _resume(context: CommandContext) -> CommandOutcome:
    if not context.arguments:
        return CommandOutcome(
            status="error",
            messages=(CommandMessage("usage: /resume PATH", "error"),),
        )
    return CommandOutcome(action="resume")


def _help(_context: CommandContext) -> CommandOutcome:
    return CommandOutcome(
        messages=(CommandMessage("/help /model /clear /resume PATH /exit"),)
    )


def _model(context: CommandContext) -> CommandOutcome:
    try:
        models = context.model_catalog.list_models()
    except Exception as exc:
        return CommandOutcome(
            status="error",
            messages=(CommandMessage(f"failed to list models: {exc}", "error"),),
        )

    return CommandOutcome(
        messages=tuple(
            CommandMessage(
                f"{'*' if _is_current_model(model, context.current_model) else ' '} "
                f"{model}"
            )
            for model in models
        )
    )


def _is_current_model(model: str, current_model: str) -> bool:
    return model == current_model or model.split("/", maxsplit=1)[-1] == current_model


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/clear", "Clear the current conversation", _clear),
    SlashCommand("/exit", "Exit the chat", _exit),
    SlashCommand("/help", "Show commands", _help),
    SlashCommand("/model", "List available models", _model),
    SlashCommand("/quit", "Exit the chat", _exit),
    SlashCommand("/resume", "Resume a transcript file", _resume),
)


class SlashCommandHandler(BaseSlashCommandHandler):
    def __init__(self, commands: tuple[SlashCommand, ...] = SLASH_COMMANDS) -> None:
        super().__init__(commands)
