"""Concurrency and native agent boundaries of the shared query engine."""

import asyncio
import inspect
import os
import tempfile
import unittest
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, cast
from unittest.mock import patch

from alpha_forge_langchain import AgentApplication
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model, wrap_model_call
from langchain.agents.middleware.types import InputAgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from test_agent_application import ScriptedModel, call
from test_streaming import wait_until


async def consume(
    engine, thread_id, agent, text, *, context=None, config=None, **state
):
    result = None
    async with aclosing(
        engine.asubmit(
            thread_id,
            agent=agent,
            input={"messages": [HumanMessage(content=text)], **state},
            context=context,
            config=config,
        )
    ) as stream:
        async for part in stream:
            if part["type"] == "values":
                result = part["data"]
    if result is None:
        raise AssertionError("no committed values")
    return result


class BlockingModel(ScriptedModel):
    entered: asyncio.Event
    release: asyncio.Event
    tasks: list = Field(default_factory=list)

    async def _agenerate(self, *args, **kwargs):
        self.tasks.append(asyncio.current_task())
        self.entered.set()
        await self.release.wait()
        return await super()._agenerate(*args, **kwargs)


class CountingState(AgentState):
    visits: NotRequired[int]


@dataclass
class TurnContext:
    user: str


class Answer(BaseModel):
    text: str


class QueryEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.app = await self.enterAsyncContext(
            AgentApplication(Path(temporary.name) / "query.sqlite3")
        )
        self.engine = self.app.query

    async def agent(self, text="done", **kwargs):
        return await self.app.agents.acreate(
            model=ScriptedModel(responses=[AIMessage(content=text)]), **kwargs
        )

    async def test_generator_is_lazy_and_exposes_no_persistence(self):
        agent = await self.agent()
        self.assertTrue(inspect.isasyncgenfunction(self.engine.asubmit))
        stream = self.engine.asubmit("lazy", agent=agent, input={"messages": []})
        self.assertEqual(self.engine._locks, {})
        self.assertFalse(hasattr(self.engine, "checkpointer"))
        await stream.aclose()
        self.assertEqual(self.engine._locks, {})

    async def test_separate_threads_overlap_and_history_reads_do_not_block(self):
        first = BlockingModel(
            responses=[AIMessage(content="A")],
            entered=asyncio.Event(),
            release=asyncio.Event(),
        )
        second = BlockingModel(
            responses=[AIMessage(content="B")],
            entered=asyncio.Event(),
            release=asyncio.Event(),
        )
        agents = [await self.app.agents.acreate(model=m) for m in (first, second)]
        tasks = [
            asyncio.create_task(consume(self.engine, tid, a, tid))
            for tid, a in zip(("A", "B"), agents)
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(first.entered.wait(), second.entered.wait()), 5
            )
            self.assertEqual(await self.engine.aget_state("A"), {"messages": []})
            second.release.set()
            result = await asyncio.wait_for(tasks[1], 5)
            self.assertFalse(tasks[0].done())
            self.assertEqual([m.text for m in result["messages"]], ["B", "B"])
            first.release.set()
            await asyncio.wait_for(tasks[0], 5)
            self.assertEqual(
                [
                    m.text
                    for m in cast(
                        list[BaseMessage],
                        (await self.engine.aget_state("A"))["messages"],
                    )
                ],
                ["A", "A"],
            )
        finally:
            first.release.set()
            second.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.engine._locks, {})

    async def test_same_thread_waiters_cancel_without_splitting_its_lock(self):
        model = BlockingModel(
            responses=[AIMessage(content="first answer")],
            entered=asyncio.Event(),
            release=asyncio.Event(),
        )
        first_agent = await self.app.agents.acreate(model=model)
        next_agent = await self.agent("next answer")
        first = asyncio.create_task(consume(self.engine, "same", first_agent, "first"))
        tasks = [first]
        try:
            await asyncio.wait_for(model.entered.wait(), 5)
            waiting = asyncio.create_task(
                consume(self.engine, "same", next_agent, "cancelled")
            )
            tasks.append(waiting)
            await wait_until(lambda: self.engine._locks["same"].users == 2)
            entry = self.engine._locks["same"]
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            self.assertIs(self.engine._locks["same"], entry)
            third = asyncio.create_task(
                consume(self.engine, "same", next_agent, "next")
            )
            tasks.append(third)
            await wait_until(lambda: entry.users == 2)
            model.release.set()
            await asyncio.wait_for(first, 5)
            result = await asyncio.wait_for(third, 5)
            self.assertEqual(
                [m.text for m in result["messages"]],
                ["first", "first answer", "next", "next answer"],
            )
        finally:
            model.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.engine._locks, {})
        await consume(self.engine, "same", await self.agent("again"), "later")
        self.assertEqual(self.engine._locks, {})

    async def test_same_native_agent_can_serve_distinct_threads(self):
        class Echo(ScriptedModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=f"answer:{messages[-1].text}")
                        )
                    ]
                )

        agent = await self.app.agents.acreate(model=Echo())
        await asyncio.gather(
            *(consume(self.engine, tid, agent, tid) for tid in ("x", "y"))
        )
        results = await asyncio.gather(
            *(consume(self.engine, tid, agent, "again") for tid in ("x", "y"))
        )
        for tid, result in zip(("x", "y"), results):
            self.assertEqual(
                [m.text for m in result["messages"]],
                [tid, f"answer:{tid}", "again", "answer:again"],
            )

    async def test_agents_tools_prompts_state_and_context_change_between_turns(self):
        contexts = []

        @before_model(state_schema=CountingState)
        async def count(state, runtime):
            return {"visits": state.get("visits", 0) + 1}

        @wrap_model_call
        async def capture(request, handler):
            contexts.append(request.runtime.context.user)
            return await handler(request)

        @tool
        def upper(text: str) -> str:
            """Uppercase text."""
            return text.upper()

        @tool
        def lower(text: str) -> str:
            """Lowercase text."""
            return text.lower()

        first_model = ScriptedModel(
            responses=[call("upper", {"text": "Ab"}), AIMessage(content="AB")]
        )
        second_model = ScriptedModel(
            responses=[call("lower", {"text": "Ab"}), AIMessage(content="ab")]
        )
        for model, custom_tool, prompt, user in (
            (first_model, upper, "First instructions", "first-user"),
            (second_model, lower, "Second instructions", "second-user"),
        ):
            agent = await self.app.agents.acreate(
                model=model,
                tools=[custom_tool],
                system_prompt=prompt,
                middleware=[count, capture],
                context_schema=TurnContext,
            )
            result = await consume(
                self.engine, "changing", agent, user, context=TurnContext(user)
            )
            self.assertTrue(
                all(messages[0].text.startswith(prompt) for messages in model.seen)
            )
        self.assertEqual(result["visits"], 4)
        self.assertEqual(
            contexts, ["first-user", "first-user", "second-user", "second-user"]
        )
        self.assertEqual(len(result["messages"]), 8)
        self.assertIn("upper", first_model.bound_tools)
        self.assertNotIn("upper", second_model.bound_tools)
        self.assertEqual(
            [m.text for m in second_model.seen[0][1:-1]], ["first-user", "", "AB", "AB"]
        )
        snapshot = await self.engine.aget_state("changing")
        cast(list[BaseMessage], snapshot["messages"]).clear()
        self.assertEqual(
            len(
                cast(
                    list[BaseMessage],
                    (await self.engine.aget_state("changing"))["messages"],
                )
            ),
            8,
        )

    async def test_thread_prompt_persists_is_isolated_and_can_be_cleared(self):
        models = [ScriptedModel(responses=[AIMessage(content="ok")]) for _ in range(4)]
        for index, (tid, prompt, updates) in enumerate(
            (
                ("one", "base one", {"system_prompt_override": "saved instructions"}),
                ("two", "base two", {}),
                ("one", "new base", {}),
                ("one", "cleared base", {"system_prompt_override": None}),
            )
        ):
            agent = await self.app.agents.acreate(
                model=models[index], system_prompt=prompt
            )
            await consume(self.engine, tid, agent, "hello", **updates)
        for model, expected in zip(
            models,
            ("saved instructions", "base two", "saved instructions", "cleared base"),
        ):
            self.assertTrue(model.seen[0][0].text.startswith(expected))
            self.assertIn("host workspace", model.seen[0][0].text)
        self.assertIsNone(
            (await self.engine.aget_state("one"))["system_prompt_override"]
        )

    async def test_direct_tool_and_structured_output_are_valid_completions(self):
        @tool(return_direct=True)
        def direct(text: str) -> str:
            """Return text directly."""
            return text

        direct_agent = await self.app.agents.acreate(
            model=ScriptedModel(responses=[call("direct", {"text": "direct answer"})]),
            tools=[direct],
        )
        result = await consume(self.engine, "direct", direct_agent, "answer")
        self.assertEqual(result["messages"][-1].type, "tool")
        structured_agent = await self.app.agents.acreate(
            model=ScriptedModel(
                responses=[call("Answer", {"text": "structured answer"})]
            ),
            response_format=ToolStrategy(Answer),
        )
        result = await consume(self.engine, "structured", structured_agent, "answer")
        self.assertEqual(result["structured_response"].text, "structured answer")

    async def test_cancellation_during_promotion_settles_before_unlocking(self):
        started, release = asyncio.Event(), asyncio.Event()
        promote = self.app._persistence.promote

        async def delayed(*args):
            started.set()
            await release.wait()
            await promote(*args)

        first_agent, next_agent = (
            await self.agent("first answer"),
            await self.agent("next answer"),
        )
        tasks = []
        with patch.object(self.app._persistence, "promote", side_effect=delayed):
            first = asyncio.create_task(
                consume(self.engine, "commit", first_agent, "first")
            )
            tasks.append(first)
            try:
                await asyncio.wait_for(started.wait(), 5)
                first.cancel()
                second = asyncio.create_task(
                    consume(self.engine, "commit", next_agent, "next")
                )
                tasks.append(second)
                await wait_until(lambda: self.engine._locks["commit"].users == 2)
                self.assertFalse(first.done())
                self.assertEqual(
                    await self.engine.aget_state("commit"), {"messages": []}
                )
                first.cancel()  # A repeated cancellation must not detach the commit.
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                result = await asyncio.wait_for(second, 5)
                self.assertEqual(
                    [m.text for m in result["messages"]],
                    ["first", "first answer", "next", "next answer"],
                )
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.engine._locks, {})

    async def test_checkpoint_overrides_and_unbound_agents_are_rejected(self):
        agent = await self.agent()
        configs = [
            {"configurable": {"thread_id": "other"}},
            {"configurable": {"checkpoint_id": "old"}},
            {"configurable": {"checkpoint_ns": "nested"}},
            {"configurable": {"__pregel_checkpointer": object()}},
        ]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                await consume(self.engine, "thread", agent, "input", config=config)
        unbound = create_agent(ScriptedModel(responses=[AIMessage(content="unused")]))
        with self.assertRaisesRegex(ValueError, "persistence backend"):
            await consume(self.engine, "thread", unbound, "input")
        self.assertEqual(self.engine._locks, {})

    async def test_runnable_config_and_input_remain_caller_owned(self):
        observed = []

        class Recorder(AsyncCallbackHandler):
            async def on_chat_model_start(self, serialized, messages, **kwargs):
                observed.append(kwargs)

        agent = await self.agent()
        message = HumanMessage(content="hello")
        input: InputAgentState = {"messages": [message]}
        config: RunnableConfig = {
            "tags": ["caller"],
            "metadata": {"source": "query-test"},
            "callbacks": [Recorder()],
        }
        async with aclosing(
            self.engine.asubmit("configured", agent=agent, input=input, config=config)
        ) as stream:
            async for _ in stream:
                pass
        self.assertIn("caller", observed[0]["tags"])
        self.assertEqual(observed[0]["metadata"]["source"], "query-test")
        self.assertNotIn("configurable", config)
        self.assertIsNone(message.id)

    async def test_concurrent_default_agent_creation_shares_client_initialization(self):
        entered, release = asyncio.Event(), asyncio.Event()
        clients = []

        class DelayedClient(AsyncOpenAI):
            async def __aenter__(self):
                clients.append(self)
                entered.set()
                await release.wait()
                return await super().__aenter__()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "offline-not-a-real-key"}),
            patch("alpha_forge_langchain.agents.AsyncOpenAI", DelayedClient),
        ):
            first = asyncio.create_task(self.app.agents.acreate())
            tasks = [first]
            try:
                await asyncio.wait_for(entered.wait(), 5)
                tasks.append(asyncio.create_task(self.app.agents.acreate()))
                release.set()
                agents = await asyncio.wait_for(asyncio.gather(*tasks), 5)
                self.assertEqual(len(clients), 1)
                self.assertIsNot(agents[0], agents[1])
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
