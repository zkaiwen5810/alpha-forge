"""Composition and resource ownership; conversation tasks belong to callers."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import Self, cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from .agents import AgentFactory
from .persistence import Persistence
from .query_engine import QueryEngine
from .store import ThreadInfo


class ThreadRepository:
    """Thread administration and recovery information, separate from execution."""

    def __init__(self, persistence: Persistence) -> None:
        self._persistence = persistence

    async def acreate(self) -> str:
        async with self._persistence.failure_boundary():
            thread_id = str(uuid4())
            await self._persistence.ensure_thread(thread_id)
            return thread_id

    async def aget(self, thread_id: str) -> ThreadInfo:
        async with self._persistence.failure_boundary():
            assert self._persistence._threads is not None
            info = await self._persistence._threads.aget(thread_id)
            await self._persistence.validate_head(info)
            return info

    async def ahistory(self, thread_id: str) -> list[BaseMessage]:
        state = await self._persistence.read_state(thread_id)
        return cast(list[BaseMessage], state.get("messages", []))


class AgentApplication:
    """Wire agents, queries, and thread administration to one private backend.

    Exit conversation tasks/streams before leaving this context. Injected models
    remain caller-owned. No conversation worker or input queue is created here.
    """

    def __init__(
        self,
        database: str | Path | None = None,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        self._persistence = Persistence(database)
        self.agents = AgentFactory(self._persistence, model=model, tools=tools)
        self.query = QueryEngine(self._persistence)
        self.threads = ThreadRepository(self._persistence)
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("application is already open")
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(self._persistence)
            await stack.enter_async_context(self.agents)
            self._stack = stack
            return self
        except BaseException:
            await stack.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()
