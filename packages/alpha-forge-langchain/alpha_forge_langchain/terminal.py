"""Full-screen prompt-toolkit UI for a streaming ChatSession."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .session import ChatSession


class ChatApplication:
    def __init__(
        self,
        session: ChatSession,
        *,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.session = session
        self.transcript = TextArea(read_only=True, scrollbar=True, wrap_lines=True)
        self.editor = TextArea(
            height=1,
            multiline=False,
            prompt="You> ",
            history=InMemoryHistory(),
            accept_handler=self._accept_input,
        )
        bindings = KeyBindings()

        @bindings.add("c-c")
        def cancel(_event: KeyPressEvent) -> None:
            self.session.request_exit(130)

        @bindings.add("c-d")
        def exit_chat(_event: KeyPressEvent) -> None:
            self.session.request_exit()

        @bindings.add("pageup")
        def scroll_up(_event: KeyPressEvent) -> None:
            self.transcript.buffer.cursor_up(
                count=max(1, self.app.output.get_size().rows // 2)
            )

        @bindings.add("pagedown")
        def scroll_down(_event: KeyPressEvent) -> None:
            self.transcript.buffer.cursor_down(
                count=max(1, self.app.output.get_size().rows // 2)
            )

        self.app: Application[int] = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(
                            FormattedTextControl(self._header),
                            height=1,
                            style="class:header",
                        ),
                        self.transcript,
                        Window(
                            FormattedTextControl(self._status),
                            height=1,
                            style="class:status",
                        ),
                        self.editor,
                    ]
                ),
                focused_element=self.editor,
            ),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=Style.from_dict({"header": "bold", "status": "italic #888888"}),
            input=input,
            output=output,
        )
        self.session.on_change = self._refresh
        self.session.on_exit = self._exit
        self._refresh()

    def _header(self) -> str:
        return f"Alpha Forge LangChain | Conversation: {self.session.conversation_id}"

    def _status(self) -> str:
        return (
            f"{self.session.status} | {self.session.queue.qsize()} queued"
            " | Enter: send | PgUp/PgDn: scroll | Ctrl+C/Ctrl+D: exit"
        )

    def _accept_input(self, buffer: Buffer) -> bool:
        self.session.submit(buffer.text)
        return False

    def _refresh(self) -> None:
        lines: list[str] = []
        for message in self.session.message_state.messages:
            if isinstance(message, HumanMessage):
                role = "You"
            elif isinstance(message, ToolMessage):
                role = f"Tool {message.name or message.tool_call_id} ({message.status})"
            else:
                role = "Assistant"
            status = (
                self.session.message_status.get(message.id, "completed")
                if message.id
                else "completed"
            )
            if status != "completed":
                role += f" [{status}]"
            lines.append(f"{role}: {message.text}")
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    lines.append(
                        f"  Tool call {call['name']}: {json.dumps(call['args'], ensure_ascii=False)}"
                    )
            lines.append("")
        lines.extend(self.session.notices)
        text = "\n".join(lines)
        old = self.transcript.buffer.document
        following = old.cursor_position == len(old.text)
        self.transcript.buffer.set_document(
            Document(
                text,
                cursor_position=len(text)
                if following
                else min(old.cursor_position, len(text)),
            ),
            bypass_readonly=True,
        )
        self.app.invalidate()

    def _exit(self, exit_code: int) -> None:
        if self.app.is_running:
            self.app.exit(result=exit_code)

    async def arun(self) -> int:
        worker = asyncio.create_task(self.session.arun())
        try:
            return await self.app.run_async()
        except EOFError:
            return 0
        finally:
            self.session.request_exit()
            # A second cancellation can interrupt the graph's async cleanup.
            # Only cancel here if the session worker has not started yet.
            if not worker.done() and not worker.cancelling():
                worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
