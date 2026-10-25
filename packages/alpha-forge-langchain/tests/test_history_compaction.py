"""Automatic request compaction keeps complete turns and original checkpoints."""

import asyncio
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from alpha_forge_langchain import AgentApplication, defaults
from alpha_forge_langchain.message_state import MessageState
from alpha_forge_langchain.middleware import HistoryCompactionMiddleware
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from test_agent_application import ScriptedModel
from test_context_views import submit


class HistoryCompactionTests(unittest.IsolatedAsyncioTestCase):
    def request(self, messages, *, system=None, tools=None):
        return ModelRequest(
            model=ScriptedModel(), messages=messages, system_message=system, tools=tools
        )

    async def compact(self, request, budget):
        handler = AsyncMock(
            return_value=ModelResponse(result=[AIMessage(content="ok")])
        )
        await HistoryCompactionMiddleware(budget).awrap_model_call(request, handler)
        handler.assert_awaited_once()
        return handler.call_args.args[0]

    async def test_below_and_at_budget_pass_through(self):
        request = self.request([HumanMessage(content="hello")])
        tokens = count_tokens_approximately(request.messages)
        for budget in (tokens, tokens + 1):
            with self.subTest(budget=budget):
                self.assertIs(await self.compact(request, budget), request)

    async def test_one_token_over_drops_whole_oldest_turn(self):
        messages = [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
            HumanMessage(content="current question"),
        ]
        budget = count_tokens_approximately(messages) - 1
        compacted = await self.compact(self.request(messages), budget)
        self.assertEqual(compacted.messages, messages[2:])
        self.assertLessEqual(count_tokens_approximately(compacted.messages), budget)

    async def test_keeps_system_and_parallel_tool_exchanges_without_aliasing(self):
        messages = [
            SystemMessage(content="history instructions"),
            HumanMessage(content="old question " * 200),
            AIMessage(content="old answer"),
            HumanMessage(content="current question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"key": key}, "id": key}
                    for key in ("a", "b")
                ],
            ),
            ToolMessage(content="first result", tool_call_id="a"),
            ToolMessage(content="second result", tool_call_id="b"),
        ]
        original = deepcopy(messages)
        expected = [messages[0], *messages[3:]]
        system = SystemMessage(content="agent prompt")
        request = self.request(messages, system=system)
        budget = count_tokens_approximately([system, *expected])
        compacted = await self.compact(request, budget)
        self.assertEqual(compacted.messages, expected)
        self.assertIs(compacted.system_message, request.system_message)
        compacted.messages[2].tool_calls[0]["args"]["key"] = "changed"
        compacted.messages[-1].content = "changed"
        self.assertEqual(messages, original)

    async def test_system_and_tool_schema_overhead_counts_toward_budget(self):
        system = SystemMessage(content="instructions " * 40)
        tools: list[BaseTool | dict[str, Any]] = [
            {"name": "lookup", "description": "schema " * 100}
        ]
        messages = [
            HumanMessage(content="old question " * 200),
            AIMessage(content="old answer"),
            HumanMessage(content="current question"),
        ]
        budget = count_tokens_approximately([system, messages[-1]], tools=tools)
        compacted = await self.compact(
            self.request(messages, system=system, tools=tools), budget
        )
        self.assertEqual(compacted.messages, messages[-1:])
        self.assertEqual(compacted.tools, tools)
        self.assertEqual(
            count_tokens_approximately(
                [compacted.system_message, *compacted.messages], tools=compacted.tools
            ),
            budget,
        )

    async def test_oversized_current_turn_never_calls_model(self):
        request = self.request(
            [HumanMessage(content="current question " * 100)],
            system=SystemMessage(content="required instructions"),
        )
        handler = AsyncMock()
        with self.assertRaisesRegex(ValueError, "current turn.*MAX_CONTEXT_TOKENS"):
            await HistoryCompactionMiddleware(20).awrap_model_call(request, handler)
        handler.assert_not_awaited()

    async def test_oversized_history_without_user_boundary_is_rejected(self):
        handler = AsyncMock()
        with self.assertRaisesRegex(ValueError, "Cannot compact model input"):
            await HistoryCompactionMiddleware(10).awrap_model_call(
                self.request([AIMessage(content="unbounded " * 100)]), handler
            )
        handler.assert_not_awaited()

    def test_nonpositive_budget_is_rejected(self):
        for budget in (0, -1):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(
                    ValueError, "MAX_CONTEXT_TOKENS must be positive"
                ),
            ):
                HistoryCompactionMiddleware(budget)


class AutomaticCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / "compaction.sqlite3"

    async def test_default_policy_budget_changes_and_original_history_survive_reopen(
        self,
    ):
        sizes = []

        @wrap_model_call
        async def measure(request, handler):
            sizes.append(
                count_tokens_approximately(
                    [request.system_message, *request.messages], tools=request.tools
                )
            )
            return await handler(request)

        old = "old question " * 2000
        new = "new question " * 2000
        async with AgentApplication(self.database) as app:
            thread = await app.threads.acreate()
            model = ScriptedModel(responses=[AIMessage(content="first answer")])
            agent = await app.agents.acreate(model=model, middleware=[measure])
            await submit(app, thread, agent, old)

            compacted_model = ScriptedModel(
                responses=[AIMessage(content="second answer")]
            )
            with patch.object(defaults, "MAX_CONTEXT_TOKENS", sizes[0]):
                agent = await app.agents.acreate(model=compacted_model)
            state, parts = await submit(app, thread, agent, new)
            self.assertEqual(
                [
                    m.text
                    for m in compacted_model.seen[0]
                    if isinstance(m, HumanMessage)
                ],
                [new],
            )
            original = await app.threads.ahistory(thread)
            self.assertEqual(
                [m.text for m in original], [old, "first answer", new, "second answer"]
            )
            self.assertEqual(state["messages"], original)
            display = MessageState(original[:-1])
            for part in parts:
                display.update(part)
            self.assertEqual(display.messages, original)

            failed_model = ScriptedModel(responses=[AIMessage(content="must not run")])
            with patch.object(defaults, "MAX_CONTEXT_TOKENS", 1):
                agent = await app.agents.acreate(model=failed_model)
            head = (await app.threads.aget(thread)).checkpoint_id
            with self.assertRaisesRegex(ValueError, "Cannot compact model input"):
                await submit(app, thread, agent, "too large")
            self.assertEqual(failed_model.seen, [])
            self.assertEqual((await app.threads.aget(thread)).checkpoint_id, head)
            self.assertEqual(await app.threads.ahistory(thread), original)

        async with AgentApplication(self.database) as app:
            self.assertEqual(await app.threads.ahistory(thread), original)
            model = ScriptedModel(responses=[AIMessage(content="restored")])
            agent = await app.agents.acreate(model=model)
            await submit(app, thread, agent, "continue")
            self.assertEqual(
                [m.text for m in model.seen[0] if isinstance(m, HumanMessage)],
                [old, new, "continue"],
            )

    async def test_shared_agent_compacts_concurrent_threads_independently(self):
        sizes = []

        @wrap_model_call
        async def measure(request, handler):
            sizes.append(
                count_tokens_approximately(
                    [request.system_message, *request.messages], tools=request.tools
                )
            )
            return await handler(request)

        async with AgentApplication(self.database) as app:
            threads = [await app.threads.acreate() for _ in range(2)]
            model = ScriptedModel(
                responses=[AIMessage(content="old answer") for _ in threads]
            )
            agent = await app.agents.acreate(model=model, middleware=[measure])
            await asyncio.gather(
                *(
                    submit(app, thread, agent, f"old {index} " * 2000)
                    for index, thread in enumerate(threads)
                )
            )

            entered = 0
            both_entered = asyncio.Event()

            @wrap_model_call
            async def overlap(request, handler):
                nonlocal entered
                entered += 1
                if entered == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), 5)
                return await handler(request)

            model = ScriptedModel(
                responses=[AIMessage(content="new answer") for _ in threads]
            )
            with patch.object(defaults, "MAX_CONTEXT_TOKENS", max(sizes)):
                agent = await app.agents.acreate(model=model, middleware=[overlap])
            await asyncio.gather(
                *(
                    submit(app, thread, agent, f"new {index} " * 2000)
                    for index, thread in enumerate(threads)
                )
            )
            self.assertEqual(
                sorted(
                    [m.text for m in seen if isinstance(m, HumanMessage)]
                    for seen in model.seen
                ),
                [[f"new {index} " * 2000] for index in range(2)],
            )
            for index, thread in enumerate(threads):
                history = await app.threads.ahistory(thread)
                self.assertEqual(
                    [m.text for m in history if isinstance(m, HumanMessage)],
                    [f"old {index} " * 2000, f"new {index} " * 2000],
                )
