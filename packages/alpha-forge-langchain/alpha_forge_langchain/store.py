"""Atomic committed thread heads alongside framework-owned checkpoints."""

import asyncio
from dataclasses import dataclass

import aiosqlite
from langchain_core.runnables import RunnableConfig

from .errors import PersistenceError, ThreadNotFoundError


@dataclass(frozen=True)
class ThreadInfo:
    thread_id: str
    checkpoint_id: str
    unfinished_work_excluded: bool

    def checkpoint_config(self) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": self.checkpoint_id,
            }
        }


class ThreadStore:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection
        self._lock = asyncio.Lock()

    async def asetup(self) -> None:
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                checkpoint_id TEXT NOT NULL,
                state_type TEXT NOT NULL,
                state BLOB NOT NULL,
                unfinished INTEGER NOT NULL DEFAULT 0
            )
        """)
        await self.connection.commit()

    async def acreate(
        self, thread_id: str, checkpoint_id: str, state: tuple[str, bytes]
    ) -> None:
        async with self._lock:
            await self.connection.execute(
                "INSERT INTO threads (thread_id, checkpoint_id, state_type, state) "
                "VALUES (?, ?, ?, ?)",
                (thread_id, checkpoint_id, *state),
            )
            await self.connection.commit()

    async def aget(self, thread_id: str) -> ThreadInfo:
        info, _ = await self.aget_snapshot(thread_id)
        return info

    async def aget_snapshot(
        self, thread_id: str
    ) -> tuple[ThreadInfo, tuple[str, bytes]]:
        async with (
            self._lock,
            self.connection.execute(
                "SELECT checkpoint_id, unfinished, state_type, state "
                "FROM threads WHERE thread_id = ?",
                (thread_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            raise ThreadNotFoundError(f"unknown thread: {thread_id}")
        return ThreadInfo(thread_id, row[0], bool(row[1])), (row[2], row[3])

    async def abegin(self, info: ThreadInfo) -> None:
        async with self._lock:
            async with self.connection.execute(
                "UPDATE threads SET unfinished = 1 "
                "WHERE thread_id = ? AND checkpoint_id = ?",
                (info.thread_id, info.checkpoint_id),
            ) as cursor:
                if cursor.rowcount != 1:
                    raise PersistenceError("thread head changed before starting turn")
            await self.connection.commit()

    async def apromote(
        self, info: ThreadInfo, checkpoint_id: str, state: tuple[str, bytes]
    ) -> None:
        async with self._lock:
            async with self.connection.execute(
                "UPDATE threads SET checkpoint_id = ?, state_type = ?, state = ?, "
                "unfinished = 0 WHERE thread_id = ? AND checkpoint_id = ?",
                (checkpoint_id, *state, info.thread_id, info.checkpoint_id),
            ) as cursor:
                if cursor.rowcount != 1:
                    raise PersistenceError("thread head changed before promotion")
            await self.connection.commit()
