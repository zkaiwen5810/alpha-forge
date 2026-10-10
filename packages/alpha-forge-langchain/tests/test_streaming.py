import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import aclosing
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from alpha_forge_langchain import AgentRuntime, PersistenceError
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from test_runtime import ScriptedModel, aconsume


class StreamingModel(ScriptedModel):
    entered: asyncio.Event | None = None
    release: asyncio.Event | None = None

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, AIMessage):
            response = [AIMessageChunk(**response.model_dump(exclude={"type"}))]
        if isinstance(response, BaseException):
            raise response
        for index, chunk in enumerate(response):
            if isinstance(chunk, BaseException):
                raise chunk
            yield ChatGenerationChunk(message=chunk)
            if index == 0 and self.release is not None:
                if self.entered is not None:
                    self.entered.set()
                await self.release.wait()


async def wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "stream.sqlite3"

    async def test_native_parts_incremental_output_and_committed_values(self):
        release = asyncio.Event()
        model = StreamingModel(
            responses=[
                [
                    AIMessageChunk(content="Hello"),
                    AIMessageChunk(content=" world"),
                ]
            ],
            release=release,
        )
        parts = []
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()

            async def consume():
                async with aclosing(runtime.asend(conversation_id, "hello")) as stream:
                    async for part in stream:
                        parts.append(part)
                        if part["type"] == "values":
                            # The iterator must release the lock before completion.
                            self.assertEqual(
                                (await runtime.ahistory(conversation_id))[-1].text,
                                "Hello world",
                            )

            task = asyncio.create_task(consume())
            try:
                await wait_until(lambda: any(p["type"] == "messages" for p in parts))
                self.assertFalse(task.done())
                first_message = next(p for p in parts if p["type"] == "messages")
                self.assertEqual(first_message["data"][0].text, "Hello")
                self.assertTrue(all(p["type"] != "values" for p in parts))
                release.set()
                await asyncio.wait_for(task, 5)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertEqual([p["type"] for p in parts].count("values"), 1)
            self.assertEqual(parts[-1]["type"], "values")
            self.assertEqual(parts[-1]["ns"], ())
            self.assertEqual(parts[-1]["interrupts"], ())
            self.assertEqual(parts[-1]["data"]["messages"][-1].text, "Hello world")
            self.assertTrue(any(p["type"] == "updates" for p in parts))
            parts[-1]["data"]["messages"][-1].content = "mutated"
            self.assertEqual(
                (await runtime.ahistory(conversation_id))[-1].text, "Hello world"
            )

    async def test_early_close_cancels_graph_without_promoting_or_leaking_lock(self):
        release = asyncio.Event()
        model = StreamingModel(
            responses=[
                [
                    AIMessageChunk(content="partial"),
                    AIMessageChunk(content=" abandoned"),
                ],
                AIMessage(content="fresh"),
            ],
            release=release,
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            async with aclosing(runtime.asend(conversation_id, "old")) as stream:
                part = await asyncio.wait_for(anext(stream), 5)
                while part["type"] != "messages":
                    part = await asyncio.wait_for(anext(stream), 5)
                self.assertEqual(part["type"], "messages")
            self.assertEqual(
                await asyncio.wait_for(runtime.ahistory(conversation_id), 5), []
            )
            self.assertTrue(
                (
                    await runtime.arestore_conversation(conversation_id)
                ).unfinished_work_excluded
            )
            release.set()
            await asyncio.wait_for(aconsume(runtime, conversation_id, "new"), 5)
            self.assertEqual(
                [m.text for m in await runtime.ahistory(conversation_id)],
                ["new", "fresh"],
            )

    async def test_partial_failure_emits_no_values_and_next_run_uses_completed_history(
        self,
    ):
        model = StreamingModel(
            responses=[
                AIMessage(content="baseline"),
                [AIMessageChunk(content="partial"), RuntimeError("provider failed")],
                AIMessage(content="fresh"),
            ]
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            await aconsume(runtime, conversation_id, "first")
            parts = []
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                async with aclosing(runtime.asend(conversation_id, "failed")) as stream:
                    async for part in stream:
                        parts.append(part)
            self.assertTrue(any(p["type"] == "messages" for p in parts))
            self.assertFalse(any(p["type"] == "values" for p in parts))
            self.assertEqual(
                [m.text for m in await runtime.ahistory(conversation_id)],
                ["first", "baseline"],
            )
            await aconsume(runtime, conversation_id, "next")
            self.assertEqual(
                [m.text for m in model.seen[-1][1:]], ["first", "baseline", "next"]
            )

    async def test_failed_promotion_never_emits_values(self):
        model = StreamingModel(responses=[AIMessage(content="uncommitted")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            parts = []
            with (
                patch.object(
                    runtime._conversation_store,
                    "apromote",
                    side_effect=sqlite3.OperationalError("disk failed"),
                ),
                self.assertRaises(PersistenceError),
            ):
                async with aclosing(runtime.asend(conversation_id, "hello")) as stream:
                    async for part in stream:
                        parts.append(part)
            self.assertFalse(any(p["type"] == "values" for p in parts))
        async with AgentRuntime(self.database) as runtime:
            self.assertEqual(await runtime.ahistory(conversation_id), [])
