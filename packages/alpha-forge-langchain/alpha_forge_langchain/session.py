"""Async submission queue and native LangChain message state for terminal chat."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import aclosing
from copy import deepcopy
from uuid import uuid4

from langchain.agents import AgentState
from langchain.agents.middleware.types import OutputAgentState
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    message_chunk_to_message,
)
from langgraph.types import StreamPart

from .runtime import AgentRuntime, PersistenceError


class MessageState:
    """Reconcile message chunks and completed node updates by message ID."""

    def __init__(self, messages: Sequence[BaseMessage] = ()) -> None:
        self.messages: list[BaseMessage] = deepcopy(list(messages))
        self._indices = {m.id: i for i, m in enumerate(self.messages) if m.id}
        self._chunks: dict[str, BaseMessageChunk] = {}

    def add_message(self, message: BaseMessage) -> None:
        message = deepcopy(message)
        index = self._indices.get(message.id) if message.id else None
        if index is None:
            if message.id:
                self._indices[message.id] = len(self.messages)
            self.messages.append(message)
        else:
            self.messages[index] = message

    def update(
        self, part: StreamPart[AgentState[object], OutputAgentState[object]]
    ) -> None:
        if part["type"] == "messages":
            chunk, _metadata = part["data"]
            if isinstance(chunk, BaseMessageChunk):
                message_id = chunk.id
                if message_id:
                    previous = self._chunks.get(message_id)
                    chunk = (
                        previous + chunk if previous is not None else deepcopy(chunk)
                    )
                    self._chunks[message_id] = chunk
                self.add_message(message_chunk_to_message(chunk))
            else:
                self.add_message(chunk)
        elif part["type"] == "updates":
            for update in part["data"].values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", ()):
                    if isinstance(message, BaseMessage):
                        self.add_message(message)
                        if message.id:
                            self._chunks.pop(message.id, None)


class ChatSession:
    """One FIFO consumer; rendering subscribes through lightweight callbacks."""

    def __init__(
        self,
        runtime: AgentRuntime,
        conversation_id: str,
        messages: Sequence[BaseMessage] = (),
        *,
        on_change: Callable[[], None] = lambda: None,
        on_exit: Callable[[int], None] = lambda _code: None,
    ) -> None:
        self.runtime = runtime
        self.conversation_id = conversation_id
        self.message_state = MessageState(messages)
        self.on_change = on_change
        self.on_exit = on_exit
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.status = "Ready"
        self.notices: list[str] = []
        self.message_status: dict[str, str] = {}
        self.is_processing = False
        self.is_exiting = False
        self.exit_code = 0
        self._worker: asyncio.Task[None] | None = None

    def submit(self, input: str) -> None:
        if self.is_exiting or not input.strip():
            return
        if input.strip() == "/exit":
            self.request_exit()
            return
        self.queue.put_nowait(input)
        self.on_change()

    def request_exit(self, exit_code: int = 0) -> None:
        if self.is_exiting:
            return
        self.is_exiting = True
        self.exit_code = exit_code
        self.status = "Exiting"
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        if self._worker is not None:
            self._worker.cancel()
        self.on_change()
        self.on_exit(exit_code)

    async def arun(self) -> None:
        if self._worker is not None:
            raise RuntimeError("chat session is already running")
        self._worker = asyncio.current_task()
        try:
            while not self.is_exiting:
                input = await self.queue.get()
                start = len(self.message_state.messages)
                self.message_state.add_message(
                    HumanMessage(content=input, id=str(uuid4()))
                )
                self.is_processing = True
                self.status = "Responding"
                self.on_change()
                try:
                    completed = False
                    async with aclosing(
                        self.runtime.asend(self.conversation_id, input)
                    ) as stream:
                        async for part in stream:
                            if part["type"] == "values":
                                completed = True
                            else:
                                self.message_state.update(part)
                            self._mark_messages(
                                start, "completed" if completed else "pending"
                            )
                            self.on_change()
                    if not completed:
                        raise RuntimeError("stream ended without committed values")
                    self.status = "Ready"
                except asyncio.CancelledError:
                    self._mark_messages(start, "cancelled")
                    raise
                except PersistenceError as exc:
                    self._mark_messages(start, "failed")
                    self.notices.append(f"Persistence error: {exc}")
                    self.request_exit(1)
                    return
                except Exception as exc:  # noqa: BLE001
                    self._mark_messages(start, "failed")
                    self.notices.append(f"Error: {str(exc) or type(exc).__name__}")
                    self.status = "Last response failed; ready"
                finally:
                    self.is_processing = False
                    self.queue.task_done()
                    self.on_change()
        except asyncio.CancelledError:
            if not self.is_exiting:
                raise
        finally:
            self._worker = None

    def _mark_messages(self, start: int, status: str) -> None:
        for message in self.message_state.messages[start:]:
            if message.id:
                self.message_status[message.id] = status
