"""Conversation heads alongside framework-owned SQLite checkpoints."""

from dataclasses import dataclass
from uuid import uuid4

import aiosqlite
from langchain_core.runnables import RunnableConfig


class ConversationNotFoundError(ValueError):
    """No conversation with the requested ID exists."""


@dataclass(frozen=True)
class ConversationInfo:
    conversation_id: str
    system_prompt: str
    thread_id: str
    checkpoint_id: str | None
    unfinished_work_excluded: bool

    def checkpoint_config(self) -> RunnableConfig:
        values = {"thread_id": self.thread_id}
        if self.checkpoint_id is not None:
            values["checkpoint_id"] = self.checkpoint_id
        return {"configurable": values}


class ConversationStore:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def asetup(self) -> None:
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                system_prompt TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT,
                unfinished INTEGER NOT NULL DEFAULT 0
            )
        """)
        await self.connection.commit()

    async def acreate(self, system_prompt: str) -> str:
        conversation_id = str(uuid4())
        await self.connection.execute(
            "INSERT INTO conversations (conversation_id, system_prompt, thread_id) "
            "VALUES (?, ?, ?)",
            (conversation_id, system_prompt, str(uuid4())),
        )
        await self.connection.commit()
        return conversation_id

    async def aget(self, conversation_id: str) -> ConversationInfo:
        async with self.connection.execute(
            "SELECT conversation_id, system_prompt, thread_id, checkpoint_id, "
            "unfinished FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"unknown conversation: {conversation_id}")
        return ConversationInfo(row[0], row[1], row[2], row[3], bool(row[4]))

    async def abegin(self, info: ConversationInfo) -> str:
        # An empty completed head must never select an abandoned first-turn thread.
        thread_id = str(uuid4()) if info.checkpoint_id is None else info.thread_id
        await self.connection.execute(
            "UPDATE conversations SET unfinished = 1, thread_id = ? "
            "WHERE conversation_id = ?",
            (thread_id, info.conversation_id),
        )
        await self.connection.commit()
        return thread_id

    async def apromote(self, conversation_id: str, checkpoint_id: str) -> None:
        await self.connection.execute(
            "UPDATE conversations SET checkpoint_id = ?, unfinished = 0 "
            "WHERE conversation_id = ?",
            (checkpoint_id, conversation_id),
        )
        await self.connection.commit()
