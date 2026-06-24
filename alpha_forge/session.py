"""Chat session state and streaming orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config
from alpha_forge.conversation import Conversation, Message
from alpha_forge.slash_commands import SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext


DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."

Role = Literal["user", "assistant", "notice", "error"]
LineRole = Literal["user", "assistant", "notice", "error", "spacer"]
Redraw = Callable[[], None]
ExitRequest = Callable[[int], None]


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    messages: list[Message]


@dataclass
class TranscriptEntry:
    role: Role
    content: str


class ChatUiState:
    def __init__(self) -> None:
        self.transcript: list[TranscriptEntry] = []
        self.pending_prompts: list[str] = []
        self.history_line_roles: list[LineRole] = ["notice"]
        self.history_style_version = 0
        self.status = "Ready"

    def add_pending(self, prompt: str) -> None:
        self.pending_prompts.append(prompt)
        self.status = self._queue_status()

    def start_turn(self, prompt: str) -> TranscriptEntry:
        self._remove_pending(prompt)
        self.transcript.append(TranscriptEntry("user", prompt))
        assistant_entry = TranscriptEntry("assistant", "")
        self.transcript.append(assistant_entry)
        self.status = "Streaming response"
        self._touch_history()
        return assistant_entry

    def append_to_response(self, entry: TranscriptEntry, text: str) -> None:
        if entry not in self.transcript:
            self.transcript.append(entry)
        entry.content += text
        self._touch_history()

    def finish_response(self) -> None:
        self.status = self._queue_status()

    def fail_response(self, entry: TranscriptEntry, message: str) -> None:
        if entry in self.transcript and not entry.content:
            self.transcript.remove(entry)
        self.transcript.append(TranscriptEntry("error", f"request failed: {message}"))
        self.status = "Request failed"
        self._touch_history()

    def add_notice(self, text: str) -> None:
        self.transcript.append(TranscriptEntry("notice", text))
        self._touch_history()

    def add_error(self, text: str) -> None:
        self.transcript.append(TranscriptEntry("error", text))
        self.status = text
        self._touch_history()

    def clear_transcript(self) -> None:
        self.transcript.clear()
        self._touch_history()

    def request_exit(self) -> None:
        self.status = "Exiting after queued responses"

    def render_history(self) -> str:
        if not self.transcript:
            self.history_line_roles = ["notice"]
            return "No messages yet."

        lines: list[str] = []
        line_roles: list[LineRole] = []
        for entry_index, entry in enumerate(self.transcript):
            if entry_index:
                lines.append("")
                line_roles.append("spacer")

            entry_lines = entry.content.splitlines() or [""]
            for line in entry_lines:
                if entry.role == "user":
                    lines.append(f" {line} " if line else "  ")
                else:
                    lines.append(line)
                line_roles.append(entry.role)

        self.history_line_roles = line_roles
        return "\n".join(lines)

    def render_pending(self) -> str:
        if not self.pending_prompts:
            return "No pending prompts."
        return "\n".join(
            f"{index}. {prompt}"
            for index, prompt in enumerate(self.pending_prompts, start=1)
        )

    def _remove_pending(self, prompt: str) -> None:
        try:
            self.pending_prompts.remove(prompt)
        except ValueError:
            return

    def _queue_status(self) -> str:
        if not self.pending_prompts:
            return "Ready"
        if len(self.pending_prompts) == 1:
            return "1 prompt queued"
        return f"{len(self.pending_prompts)} prompts queued"

    def _touch_history(self) -> None:
        self.history_style_version += 1


class ChatReplController:
    def __init__(
        self,
        config: Config,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat: ChatClient | None = None,
        command_handler: SlashCommandHandler | None = None,
    ) -> None:
        self.config = config
        self.conversation = Conversation(system_prompt=system_prompt)
        self.chat = chat if chat is not None else ChatClient(config)
        self.command_handler = (
            command_handler if command_handler is not None else SlashCommandHandler()
        )
        self.state = ChatUiState()
        self.queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        self.exiting = False
        self.request_redraw: Redraw = lambda: None
        self.request_app_exit: ExitRequest = lambda _exit_code: None

    def submit(self, user_input: str) -> None:
        if self.exiting:
            return

        text = user_input.strip()
        if not text:
            return

        if text.startswith("/"):
            self._handle_command(text)
            return

        self.conversation.add_user(user_input)
        self.state.add_pending(user_input)
        self.queue.put_nowait(WorkItem(user_input, list(self.conversation.messages)))
        self.request_redraw()

    def request_exit(self) -> None:
        if self.exiting:
            return
        self.exiting = True
        self.state.request_exit()
        self.queue.put_nowait(None)
        self.request_redraw()

    async def consume(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    self.request_app_exit(0)
                    return
                await self._stream_response(item)
            finally:
                self.queue.task_done()

    async def _stream_response(self, item: WorkItem) -> None:
        response_entry = self.state.start_turn(item.prompt)
        self.request_redraw()

        chunks: list[str] = []
        try:
            async for piece in self.chat.stream(item.messages):
                chunks.append(piece)
                self.state.append_to_response(response_entry, piece)
                self.request_redraw()
        except Exception as exc:
            self.state.fail_response(response_entry, str(exc))
            self.request_redraw()
            return

        self.conversation.add_assistant("".join(chunks))
        self.state.finish_response()
        self.request_redraw()

    def _handle_command(self, text: str) -> None:
        command_name = text.split(maxsplit=1)[0]
        if command_name == "/clear":
            self.state.clear_transcript()

        command_result = self.command_handler.handle(
            text,
            CommandContext(
                config=self.config,
                conversation=self.conversation,
                chat=self.chat,
                print_text=self._add_notice,
            ),
        )
        if command_result.exit_requested:
            self.request_exit()
            return
        if command_result.handled:
            self.request_redraw()
            return

        self.state.add_error(f"unknown command: {command_name}")
        self.request_redraw()

    def _add_notice(self, text: str) -> None:
        self.state.add_notice(text)
        self.request_redraw()
