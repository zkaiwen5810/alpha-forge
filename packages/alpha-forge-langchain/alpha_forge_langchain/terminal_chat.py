"""Terminal chat input queue, presentation state, and query task lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langgraph.graph.state import CompiledStateGraph

from .errors import PersistenceError
from .message_state import MessageState
from .query_engine import QueryEngine


class TerminalChatController:
    """Drive one terminal thread; notify the UI as presentation state changes."""

    def __init__(
        self,
        query: QueryEngine,
        thread_id: str,
        messages: Sequence[BaseMessage] = (),
        *,
        agent_factory: Callable[[], Awaitable[CompiledStateGraph[Any, Any, Any, Any]]],
        on_change: Callable[[], None] = lambda: None,
        on_exit: Callable[[int], None] = lambda _code: None,
    ) -> None:
        self.query = query
        self.thread_id = thread_id
        self._agent_factory = agent_factory
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
            raise RuntimeError("terminal chat controller is already running")
        self._worker = asyncio.current_task()
        try:
            while not self.is_exiting:
                input = await self.queue.get()
                start = len(self.message_state.messages)
                human = HumanMessage(content=input, id=str(uuid4()))
                self.message_state.add_message(human)
                self.is_processing = True
                self.status = "Responding"
                self.on_change()
                try:
                    completed = False
                    agent = await self._agent_factory()
                    async with aclosing(
                        self.query.asubmit(
                            self.thread_id, agent=agent, input={"messages": [human]}
                        )
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
