"""Private SQLite ownership and committed-state persistence."""

from __future__ import annotations

import fcntl
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import aiosqlite
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from . import defaults
from .errors import DatabaseBusyError, PersistenceError, ThreadNotFoundError
from .store import ThreadInfo, ThreadStore


class Persistence:
    def __init__(self, database: str | Path | None = None) -> None:
        self.database = (
            Path(database) if database is not None else defaults.database_path()
        )
        self._stack: AsyncExitStack | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._threads: ThreadStore | None = None
        self._halted = False

    async def __aenter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("application is already open")
        stack = AsyncExitStack()
        try:
            self.database = self.database.expanduser().resolve()
            self.database.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(
                self.database.with_suffix(self.database.suffix + ".lock").open("a")
            )
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DatabaseBusyError(f"database is in use: {self.database}") from exc
            checkpoints = await stack.enter_async_context(
                aiosqlite.connect(self.database)
            )
            self._checkpointer = AsyncSqliteSaver(checkpoints)
            await self._checkpointer.setup()
            # Metadata transactions must not be committed by checkpoint writes.
            metadata = await stack.enter_async_context(aiosqlite.connect(self.database))
            self._threads = ThreadStore(metadata)
            await self._threads.asetup()
            self._halted = False
            self._stack = stack
            return self
        except BaseException:
            await stack.aclose()
            self._checkpointer = None
            self._threads = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stack, self._stack = self._stack, None
        try:
            if stack is not None:
                await stack.aclose()
        finally:
            self._checkpointer = None
            self._threads = None

    def check_open(self) -> None:
        if self._stack is None:
            raise RuntimeError("use AgentApplication inside an async with block")
        if self._halted:
            raise PersistenceError(
                "persistence failed; close and reopen the application"
            )

    @asynccontextmanager
    async def failure_boundary(self) -> AsyncGenerator[None]:
        """Require healthy storage and translate SQLite failures into a halt.

        This also wraps native agent execution because LangGraph writes its own
        checkpoints. It starts no transaction and acquires no lock; thread
        serialization and metadata transactions have their own scopes.
        """
        self.check_open()
        try:
            yield
        except sqlite3.Error as exc:
            self._halted = True
            raise PersistenceError(str(exc)) from exc

    async def ensure_thread(self, thread_id: str) -> ThreadInfo:
        """Caller holds the query engine's lock for this thread."""
        assert self._threads is not None
        try:
            return await self._threads.aget(thread_id)
        except ThreadNotFoundError:
            assert self._checkpointer is not None
            config = await self._checkpointer.aput(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                empty_checkpoint(),
                {"source": "update", "step": -1, "parents": {}},
                {},
            )
            checkpoint_id = cast(str, config.get("configurable", {})["checkpoint_id"])
            await self._threads.acreate(thread_id, checkpoint_id)
            return await self._threads.aget(thread_id)

    async def read_history(self, thread_id: str) -> list[BaseMessage]:
        """Read only messages from the committed native checkpoint."""
        async with self.failure_boundary():
            assert self._threads is not None
            info = await self._threads.aget(thread_id)
            saved = await self.read_checkpoint(info)
            messages = saved.checkpoint["channel_values"].get("messages", [])
            return deepcopy(cast(list[BaseMessage], messages))

    def owns_checkpointer(self, checkpointer: object) -> bool:
        return checkpointer is self._checkpointer

    async def begin(self, thread_id: str) -> ThreadInfo:
        """Select and mark a head while the engine owns the thread lock."""
        info = await self.ensure_thread(thread_id)
        await self.read_checkpoint(info)
        assert self._threads is not None
        await self._threads.abegin(info)
        return info

    async def read_checkpoint(self, info: ThreadInfo) -> CheckpointTuple:
        assert self._checkpointer is not None
        saved = await self._checkpointer.aget_tuple(info.checkpoint_config())
        if saved is None:
            raise PersistenceError("completed thread checkpoint is missing")
        return saved

    async def promote(self, info: ThreadInfo, checkpoint_id: str) -> None:
        async with self.failure_boundary():
            assert self._threads is not None
            await self._threads.apromote(info, checkpoint_id)
