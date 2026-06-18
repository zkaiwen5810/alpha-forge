"""Built-in slash-command implementations."""

from __future__ import annotations

from alpha_forge.slash_commands.base import CommandContext, CommandResult, SlashCommand


def _exit(_context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, exit_requested=True)


def _clear(context: CommandContext) -> CommandResult:
    context.conversation.clear()
    context.print_text("conversation cleared")
    return CommandResult(handled=True)


def _help(context: CommandContext) -> CommandResult:
    context.print_text("/help /model /clear /exit")
    return CommandResult(handled=True)


def _model(context: CommandContext) -> CommandResult:
    try:
        models = context.chat.list_models()
    except Exception as exc:
        context.print_text(f"failed to list models: {exc}")
        return CommandResult(handled=True)

    for model in models:
        marker = "*" if _is_current_model(model, context.config.model) else " "
        context.print_text(f"{marker} {model}")
    return CommandResult(handled=True)


def _is_current_model(model: str, current_model: str) -> bool:
    return model == current_model or model.split("/", maxsplit=1)[-1] == current_model


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/clear", "Clear the current conversation", _clear),
    SlashCommand("/exit", "Exit the chat", _exit),
    SlashCommand("/help", "Show commands", _help),
    SlashCommand("/model", "List available models", _model),
    SlashCommand("/quit", "Exit the chat", _exit),
)


class SlashCommandHandler:
    def __init__(self, commands: tuple[SlashCommand, ...] = SLASH_COMMANDS) -> None:
        self.commands = {command.name: command for command in commands}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.commands)

    def handle(self, text: str, context: CommandContext) -> CommandResult:
        command_name = text.strip().split(maxsplit=1)[0]
        command = self.commands.get(command_name)
        if command is None:
            return CommandResult(handled=False)
        return command.handler(context)
