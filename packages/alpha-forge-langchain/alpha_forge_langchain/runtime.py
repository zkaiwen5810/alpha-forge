"""Headless LangChain agent and completed-turn recovery boundary."""

from __future__ import annotations

import asyncio
import fcntl
import os
import sqlite3
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Self, TypedDict, cast

import aiosqlite

# LangChain's Checkpointer alias omits its generic parameter in this signature.
from langchain.agents import (  # pyright: ignore[reportUnknownVariableType]
    AgentState,
    create_agent,
)
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain.agents.middleware.model_call_limit import ModelCallLimitState
from langchain.agents.middleware.types import OutputAgentState, ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, StreamPart
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr

from . import defaults
from .store import ConversationInfo, ConversationStore
from .tools import calculator


class DatabaseBusyError(RuntimeError):
    """Another runtime owns this database."""


class PersistenceError(RuntimeError):
    """Conversation persistence failed or a completed checkpoint is missing."""


@wrap_tool_call(state_schema=ModelCallLimitState)
async def _atool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
) -> ToolMessage | Command[object]:
    try:
        return await handler(request)
    # Normalize arbitrary exceptions from caller-supplied tools for the model.
    except Exception as exc:  # noqa: BLE001
        return ToolMessage(
            content=f"error: {str(exc) or type(exc).__name__}",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )


class _ClientOptions(TypedDict):
    api_key: str
    base_url: str
    timeout: float
    max_retries: int


def _validate_input(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input must be a non-empty string")
    return value


class AgentRuntime:
    """One database owner; use ``async with``. Injected models remain caller-owned."""

    def __init__(
        self,
        database: str | Path | None = None,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        self.database: Path = (
            Path(database) if database is not None else defaults.database_path()
        )
        self._provided_model: BaseChatModel | None = model
        self._model: BaseChatModel | None = model
        self._tools: list[BaseTool] = [calculator, *tools]
        names = [item.name for item in self._tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique, including calculator")
        self._operations: asyncio.Lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._conversation_store: ConversationStore | None = None
        self._halted: bool = False

    async def __aenter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("runtime is already open")
        stack = AsyncExitStack()
        try:
            self.database = self.database.expanduser().resolve()
            self.database.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.database.with_suffix(self.database.suffix + ".lock")
            lock = stack.enter_context(lock_path.open("a"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DatabaseBusyError(f"database is in use: {self.database}") from exc
            connection = await stack.enter_async_context(
                aiosqlite.connect(self.database)
            )
            self._checkpointer = AsyncSqliteSaver(connection)
            await self._checkpointer.setup()
            self._conversation_store = ConversationStore(connection)
            await self._conversation_store.asetup()
            self._stack = stack
            self._halted = False
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
        async with self._operations:
            stack, self._stack = self._stack, None
            if stack is not None:
                await stack.aclose()
            self._model = self._provided_model
            self._checkpointer = None
            self._conversation_store = None

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        async with self._operations:
            if self._stack is None:
                raise RuntimeError("use AgentRuntime inside an async with block")
            if self._halted:
                raise PersistenceError(
                    "persistence failed; close and reopen the runtime"
                )
            try:
                yield
            except sqlite3.Error as exc:
                self._halted = True
                raise PersistenceError(str(exc)) from exc

    async def acreate_conversation(self) -> str:
        async with self._operation():
            assert self._conversation_store is not None
            return await self._conversation_store.acreate(defaults.SYSTEM_PROMPT)

    async def arestore_conversation(self, conversation_id: str) -> ConversationInfo:
        async with self._operation():
            assert self._conversation_store is not None
            info = await self._conversation_store.aget(conversation_id)
            _ = await self._ahistory(info)
            return info

    async def ahistory(self, conversation_id: str) -> list[BaseMessage]:
        async with self._operation():
            assert self._conversation_store is not None
            return await self._ahistory(
                await self._conversation_store.aget(conversation_id)
            )

    async def _ahistory(self, info: ConversationInfo) -> list[BaseMessage]:
        if info.checkpoint_id is None:
            return []
        assert self._checkpointer is not None
        saved = await self._checkpointer.aget_tuple(info.checkpoint_config())
        if saved is None:
            raise PersistenceError("completed conversation checkpoint is missing")
        messages = cast(
            list[BaseMessage], saved.checkpoint["channel_values"].get("messages", [])
        )
        return deepcopy(messages)

    async def _aget_model(self) -> BaseChatModel:
        if self._model is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("set OPENAI_API_KEY before sending a prompt")
            assert self._stack is not None
            # Explicit clients prevent SDK/framework endpoint env fallbacks.
            kwargs: _ClientOptions = {
                "api_key": key,
                "base_url": defaults.BASE_URL,
                "timeout": defaults.TIMEOUT,
                "max_retries": 0,
            }
            sync = self._stack.enter_context(OpenAI(**kwargs))
            asynchronous = await self._stack.enter_async_context(AsyncOpenAI(**kwargs))
            self._model = ChatOpenAI(
                model=defaults.MODEL,
                api_key=SecretStr(key),
                base_url=defaults.BASE_URL,
                timeout=defaults.TIMEOUT,
                max_retries=0,
                client=sync.chat.completions,
                async_client=asynchronous.chat.completions,
                root_client=sync,
                root_async_client=asynchronous,
                use_responses_api=False,
            )
        return self._model

    async def asend(
        self, conversation_id: str, input: str
    ) -> AsyncGenerator[StreamPart[AgentState[object], OutputAgentState[object]]]:
        """Stream native messages/updates, then committed state as terminal values.

        Consume inside ``contextlib.aclosing`` so stopping early releases the
        operation lock and cancels unfinished graph work. Only the terminal
        ``values`` part indicates that the conversation head was promoted.
        """
        input = _validate_input(input)
        async with self._operation():
            assert self._conversation_store is not None
            assert self._checkpointer is not None
            info = await self._conversation_store.aget(conversation_id)
            await self._ahistory(info)
            agent = create_agent(
                await self._aget_model(),
                tools=self._tools,
                system_prompt=info.system_prompt,
                checkpointer=self._checkpointer,
                middleware=[
                    _atool_errors,
                    ModelCallLimitMiddleware(
                        run_limit=defaults.MAX_MODEL_CALLS,
                        exit_behavior="error",
                    ),
                ],
            )
            thread_id = await self._conversation_store.abegin(info)
            config = info.checkpoint_config()
            config.setdefault("configurable", {})["thread_id"] = thread_id
            config["recursion_limit"] = 100
            # LangGraph annotates astream as AsyncIterator, but implements an
            # async generator whose aclose is required for early-stop cleanup.
            stream = cast(
                AsyncGenerator[
                    StreamPart[AgentState[object], OutputAgentState[object]]
                ],
                agent.astream(
                    {"messages": [HumanMessage(content=input)]},
                    config,
                    stream_mode=["messages", "updates"],
                    version="v2",
                    durability="sync",
                ),
            )
            async with aclosing(stream):
                async for part in stream:
                    yield deepcopy(part)
            # No concurrent invocation exists, so latest is this invocation's tail,
            # including when it branched from an older completed checkpoint.
            snapshot = await agent.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            values = cast(OutputAgentState[object], deepcopy(snapshot.values))
            messages = values["messages"]
            final_message = messages[-1] if messages else None
            if (
                snapshot.next
                or not isinstance(final_message, AIMessage)
                or final_message.tool_calls
            ):
                raise RuntimeError("agent did not finish with an assistant response")
            checkpoint_id = cast(
                str, snapshot.config.get("configurable", {})["checkpoint_id"]
            )
            await self._conversation_store.apromote(conversation_id, checkpoint_id)
        # The caller can inspect history at completion without deadlocking
        # against a suspended iterator that still holds the operation lock.
        yield {"type": "values", "ns": (), "data": values, "interrupts": ()}
