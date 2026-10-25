"""Caller-driven turns, serialized per thread, using native LangChain agents."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamPart
from langgraph.typing import ContextT, InputT, OutputT, StateT

from .errors import PersistenceError
from .persistence import Persistence


@dataclass
class _ThreadLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class QueryEngine:
    """Shared by caller-owned conversation tasks; creates no execution workers.

    No work runs until ``asubmit`` is iterated. Consume inside ``aclosing`` when
    stopping early. Finish conversation tasks before closing application storage.
    """

    def __init__(self, persistence: Persistence) -> None:
        self._persistence = persistence
        self._locks: dict[str, _ThreadLock] = {}

    @asynccontextmanager
    async def _thread(self, thread_id: str) -> AsyncGenerator[None]:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        self._persistence.check_open()
        # No awaits in registration/refcounts: atomic on the owning event loop.
        entry = self._locks.setdefault(thread_id, _ThreadLock())
        entry.users += 1
        try:
            async with entry.lock, self._persistence.failure_boundary():
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[thread_id]

    def _validate_invocation(
        self,
        thread_id: str,
        agent: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
        config: RunnableConfig | None = None,
    ) -> None:
        if not self._persistence.owns_checkpointer(agent.checkpointer):
            raise ValueError("agent must use this application's persistence backend")
        for supplied in (agent.config, config):
            configurable = (supplied or {}).get("configurable", {})
            if configurable.get("thread_id", thread_id) != thread_id:
                raise ValueError("config thread_id conflicts with submission")
            if any(
                key in {"checkpoint_id", "checkpoint_ns", "checkpoint_map"}
                or key.startswith("__pregel_")
                for key in configurable
            ):
                raise ValueError("checkpoint configuration is managed by the engine")

    async def aget_state(
        self,
        thread_id: str,
        *,
        agent: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
    ) -> StateT:
        """Project committed checkpoint state through the supplied native agent.

        This does not invoke the model or wait for an active turn. The agent's
        schema selects state channels without exposing checkpoint routing data.
        """
        async with self._persistence.failure_boundary():
            self._validate_invocation(thread_id, agent)
            assert self._persistence._threads is not None
            info = await self._persistence._threads.aget(thread_id)
            snapshot = await agent.aget_state(info.checkpoint_config())
            if snapshot.created_at is None:
                raise PersistenceError("completed thread checkpoint is missing")
            return deepcopy(cast(StateT, snapshot.values))

    async def asubmit(
        self,
        thread_id: str,
        *,
        agent: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
        input: InputT,
        config: RunnableConfig | None = None,
        context: ContextT | None = None,
    ) -> AsyncGenerator[StreamPart[StateT, OutputT]]:
        """Execute one turn; only terminal values indicate a committed result.

        The agent restores the committed checkpoint. Supply new input only,
        not the entire history. Config and context apply to this turn.
        """
        async with self._thread(thread_id):
            self._validate_invocation(thread_id, agent, config)
            info = await self._persistence.begin(thread_id)
            invocation: RunnableConfig = {**(config or {})}
            invocation["configurable"] = {
                **invocation.get("configurable", {}),
                **info.checkpoint_config().get("configurable", {}),
            }
            stream = cast(
                AsyncGenerator[StreamPart[StateT, OutputT]],
                agent.astream(
                    deepcopy(input),
                    invocation,
                    context=context,
                    stream_mode=["messages", "updates"],
                    version="v2",
                    durability="sync",
                ),
            )
            async with aclosing(stream):
                async for part in stream:
                    yield deepcopy(part)
            snapshot = await agent.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            if snapshot.next or snapshot.interrupts:
                raise RuntimeError("agent did not complete the turn")
            values = cast(OutputT, deepcopy(snapshot.values))
            checkpoint_id = cast(
                str, snapshot.config.get("configurable", {})["checkpoint_id"]
            )
            # A short commit task settles durable promotion even if the caller
            # cancels concurrently. It never executes agent work in the background.
            promotion = asyncio.create_task(
                self._persistence.promote(info, checkpoint_id)
            )
            cancellation: asyncio.CancelledError | None = None
            while not promotion.done():
                try:
                    await asyncio.shield(promotion)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            promotion.result()
            if cancellation is not None:
                raise cancellation
        yield {"type": "values", "ns": (), "data": values, "interrupts": ()}
