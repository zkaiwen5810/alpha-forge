"""Prompt-toolkit completion for slash commands."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


class SlashCommandCompleter(Completer):
    def __init__(self, commands: Iterable[str]) -> None:
        self.commands = tuple(commands)

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterator[Completion]:
        del complete_event
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return

        for command in self.commands:
            if command.startswith(text):
                yield Completion(command, start_position=-len(text))
