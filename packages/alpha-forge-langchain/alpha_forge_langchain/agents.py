"""Application agent construction and owned model/tool clients."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypedDict, TypeVar, cast

from aiohttp import ClientSession, ClientTimeout
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import InputAgentState, OutputAgentState
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr

from . import defaults
from .middleware import (
    HistoryCompactionMiddleware,
    HostPathMiddleware,
    ThreadPromptMiddleware,
    filesystem_middleware,
    tool_errors,
)
from .persistence import Persistence
from .tools import calculator, web_fetch_tool

ResponseT = TypeVar("ResponseT", default=Any)


class _ClientOptions(TypedDict):
    api_key: str
    base_url: str
    timeout: float
    max_retries: int


class AgentFactory:
    """Build a fresh native agent per turn, reusing owned HTTP clients.

    Persistence is wired at application construction, not exposed by QueryEngine.
    Injected models remain caller-owned. Configuration is never cached by thread.
    """

    def __init__(
        self,
        persistence: Persistence,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        self._persistence = persistence
        self._provided_model = model
        self._model_lock = asyncio.Lock()
        self._model = model
        self.workspace = Path.cwd().resolve()
        self._web_fetch = web_fetch_tool()
        self._extra_tools = list(tools)
        self._stack: AsyncExitStack | None = None
        self._validate_tools(self._extra_tools)

    def _validate_tools(self, tools: Sequence[BaseTool]) -> None:
        names = [
            tool.name
            for tool in [
                calculator,
                self._web_fetch,
                *tools,
                *filesystem_middleware(self.workspace).tools,
            ]
        ]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique, including all built-in tools")

    async def __aenter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("agent factory is already open")
        self._persistence.check_open()
        stack = AsyncExitStack()
        try:
            self._web_fetch.requests_wrapper.aiosession = (
                await stack.enter_async_context(
                    ClientSession(timeout=ClientTimeout(total=30.0))
                )
            )
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
        try:
            if stack is not None:
                await stack.aclose()
        finally:
            self._model = self._provided_model
            self._web_fetch.requests_wrapper.aiosession = None

    async def _aget_model(self) -> BaseChatModel:
        async with self._model_lock:
            if self._model is None:
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("set OPENAI_API_KEY before sending a prompt")
                assert self._stack is not None
                kwargs: _ClientOptions = {
                    "api_key": key,
                    "base_url": defaults.BASE_URL,
                    "timeout": defaults.TIMEOUT,
                    "max_retries": 0,
                }
                sync = self._stack.enter_context(OpenAI(**kwargs))
                asynchronous = await self._stack.enter_async_context(
                    AsyncOpenAI(**kwargs)
                )
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

    async def acreate(
        self,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | SystemMessage | None = defaults.SYSTEM_PROMPT,
        middleware: Sequence[AgentMiddleware[Any, Any, Any]] = (),
        state_schema: type[AgentState[ResponseT]] | None = None,
        context_schema: type[ContextT] | None = None,
        response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    ) -> CompiledStateGraph[
        AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
    ]:
        """Build this turn's agent, optionally deriving its model context.

        Context policies belong in model-call middleware: edit a deep copy of
        request.messages and pass request.override(messages=edited) to the
        handler. Keep state.messages as the original transcript. Middleware
        returning message state updates changes that transcript permanently.
        Old complete turns are automatically removed from model input when it
        exceeds defaults.MAX_CONTEXT_TOKENS, after caller-provided policies run.
        """
        self._persistence.check_open()
        if self._stack is None:
            raise RuntimeError("agent factory is closed")
        extra = self._extra_tools if tools is None else list(tools)
        self._validate_tools(extra)
        # Heterogeneous middleware state schemas are merged by LangChain.
        layers = cast(
            Sequence[AgentMiddleware[Any, ContextT, Any]],
            [
                ThreadPromptMiddleware(),
                tool_errors,
                HostPathMiddleware(self.workspace),
                filesystem_middleware(self.workspace),
                ModelCallLimitMiddleware(
                    run_limit=defaults.MAX_MODEL_CALLS, exit_behavior="error"
                ),
                *middleware,
                HistoryCompactionMiddleware(defaults.MAX_CONTEXT_TOKENS),
            ],
        )
        return create_agent(
            model if model is not None else await self._aget_model(),
            tools=[calculator, self._web_fetch, *extra],
            system_prompt=system_prompt,
            middleware=layers,
            state_schema=state_schema,
            context_schema=context_schema,
            response_format=response_format,
            checkpointer=self._persistence._checkpointer,
        ).with_config({"recursion_limit": 100})
