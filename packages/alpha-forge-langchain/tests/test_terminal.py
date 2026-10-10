import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge_langchain import AgentRuntime
from alpha_forge_langchain.session import ChatSession
from alpha_forge_langchain.terminal import ChatApplication
from langchain_core.messages import AIMessage, AIMessageChunk
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from test_runtime import ScriptedModel, aconsume, call
from test_streaming import StreamingModel, wait_until


class TerminalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "terminal.sqlite3"

    async def test_input_stays_responsive_and_queued_runs_are_fifo(self):
        release = asyncio.Event()
        model = StreamingModel(
            responses=[
                [AIMessageChunk(content="Hello"), AIMessageChunk(content=" world")],
                AIMessage(content="second answer"),
            ],
            release=release,
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            session = ChatSession(runtime, conversation_id)
            with create_pipe_input() as input:
                app = ChatApplication(session, input=input, output=DummyOutput())
                task = asyncio.create_task(app.arun())
                try:
                    await wait_until(lambda: app.app.is_running)
                    input.send_text("first\r")
                    await wait_until(lambda: "Hello" in app.transcript.text)
                    self.assertIn("pending", app.transcript.text)
                    input.send_text("second\rtyped draft")
                    await wait_until(lambda: app.editor.text == "typed draft")
                    self.assertEqual(session.queue.qsize(), 1)
                    self.assertEqual(len(model.seen), 1)
                    release.set()
                    await asyncio.wait_for(session.queue.join(), 5)
                    self.assertEqual(app.editor.text, "typed draft")
                    self.assertEqual(
                        [m[-1].text for m in model.seen], ["first", "second"]
                    )
                    self.assertEqual(app.transcript.text.count("Hello world"), 1)
                    self.assertNotIn("pending", app.transcript.text)
                    self.assertEqual(
                        [m.text for m in await runtime.ahistory(conversation_id)],
                        ["first", "Hello world", "second", "second answer"],
                    )
                    input.send_text("\x04")
                    self.assertEqual(await asyncio.wait_for(task, 5), 0)
                finally:
                    release.set()
                    session.request_exit()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_ctrl_c_cancels_partial_run_discards_queue_and_releases_database(
        self,
    ):
        model = StreamingModel(
            responses=[
                [
                    AIMessageChunk(content="partial"),
                    AIMessageChunk(content="unfinished"),
                ],
                AIMessage(content="must not run"),
            ],
            release=asyncio.Event(),
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            session = ChatSession(runtime, conversation_id)
            with create_pipe_input() as input:
                app = ChatApplication(session, input=input, output=DummyOutput())
                task = asyncio.create_task(app.arun())
                try:
                    await wait_until(lambda: app.app.is_running)
                    input.send_text("first\r")
                    await wait_until(lambda: "partial" in app.transcript.text)
                    input.send_text("queued\r")
                    await wait_until(lambda: session.queue.qsize() == 1)
                    input.send_text("\x03")
                    self.assertEqual(await asyncio.wait_for(task, 5), 130)
                    await asyncio.wait_for(session.queue.join(), 5)
                    self.assertEqual(len(model.seen), 1)
                    self.assertIn("cancelled", session.message_status.values())
                    self.assertEqual(await runtime.ahistory(conversation_id), [])
                finally:
                    session.request_exit()
                    await asyncio.gather(task, return_exceptions=True)
        async with AgentRuntime(self.database) as runtime:
            self.assertTrue(
                (
                    await runtime.arestore_conversation(conversation_id)
                ).unfinished_work_excluded
            )

    async def test_resume_renders_tool_history_and_exit_command(self):
        model = ScriptedModel(responses=[call(), AIMessage(content="20")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            await aconsume(runtime, conversation_id, "calculate")
        async with AgentRuntime(self.database) as runtime:
            session = ChatSession(
                runtime, conversation_id, await runtime.ahistory(conversation_id)
            )
            with create_pipe_input() as input:
                app = ChatApplication(session, input=input, output=DummyOutput())
                self.assertIn("Tool call calculator", app.transcript.text)
                self.assertIn("Tool calculator (success): 20", app.transcript.text)
                self.assertIn("Assistant: 20", app.transcript.text)
                task = asyncio.create_task(app.arun())
                try:
                    await wait_until(lambda: app.app.is_running)
                    input.send_text("/exit\r")
                    self.assertEqual(await asyncio.wait_for(task, 5), 0)
                finally:
                    session.request_exit()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_model_failure_is_visible_and_next_queued_input_runs(self):
        model = StreamingModel(
            responses=[
                [AIMessageChunk(content="partial"), RuntimeError("provider failed")],
                AIMessage(content="fresh"),
            ]
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            session = ChatSession(runtime, conversation_id)
            session.submit("first")
            session.submit("next")
            worker = asyncio.create_task(session.arun())
            try:
                await asyncio.wait_for(session.queue.join(), 5)
                self.assertIn("failed", session.message_status.values())
                self.assertIn("provider failed", session.notices[0])
                self.assertEqual(
                    [m.text for m in await runtime.ahistory(conversation_id)],
                    ["next", "fresh"],
                )
            finally:
                session.request_exit()
                await worker

    async def test_persistence_failure_stops_the_app_and_remaining_queue(self):
        model = ScriptedModel(
            responses=[
                AIMessage(content="uncommitted"),
                AIMessage(content="must not run"),
            ]
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            session = ChatSession(runtime, conversation_id)
            with create_pipe_input() as input:
                app = ChatApplication(session, input=input, output=DummyOutput())
                with patch.object(
                    runtime._conversation_store,
                    "apromote",
                    side_effect=sqlite3.OperationalError("disk failed"),
                ):
                    task = asyncio.create_task(app.arun())
                    try:
                        await wait_until(lambda: app.app.is_running)
                        input.send_text("first\rsecond\r")
                        self.assertEqual(await asyncio.wait_for(task, 5), 1)
                        await asyncio.wait_for(session.queue.join(), 5)
                        self.assertEqual(len(model.seen), 1)
                        self.assertIn("Persistence error", session.notices[0])
                    finally:
                        session.request_exit()
                        await asyncio.gather(task, return_exceptions=True)

    async def test_eof_cancels_active_run(self):
        model = StreamingModel(
            responses=[[AIMessageChunk(content="partial")]], release=asyncio.Event()
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            session = ChatSession(runtime, conversation_id)
            with create_pipe_input() as input:
                app = ChatApplication(session, input=input, output=DummyOutput())
                task = asyncio.create_task(app.arun())
                try:
                    await wait_until(lambda: app.app.is_running)
                    input.send_text("first\r")
                    await wait_until(lambda: "partial" in app.transcript.text)
                    input.close()
                    self.assertEqual(await asyncio.wait_for(task, 5), 0)
                    self.assertEqual(await runtime.ahistory(conversation_id), [])
                finally:
                    session.request_exit()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_exit_waits_for_async_stream_cleanup(self):
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        cleaned = asyncio.Event()

        class CleanupModel(StreamingModel):
            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                try:
                    async for chunk in super()._astream(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    ):
                        yield chunk
                finally:
                    cleanup_started.set()
                    await cleanup_release.wait()
                    cleaned.set()

        model = CleanupModel(
            responses=[[AIMessageChunk(content="partial")]], release=asyncio.Event()
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation_id = await runtime.acreate_conversation()
            session = ChatSession(runtime, conversation_id)
            with create_pipe_input() as input:
                app = ChatApplication(session, input=input, output=DummyOutput())
                task = asyncio.create_task(app.arun())
                try:
                    await wait_until(lambda: app.app.is_running)
                    input.send_text("first\r")
                    await wait_until(lambda: "partial" in app.transcript.text)
                    input.send_text("\x03")
                    await asyncio.wait_for(cleanup_started.wait(), 5)
                    await asyncio.sleep(0.05)
                    self.assertFalse(task.done())
                    cleanup_release.set()
                    self.assertEqual(await asyncio.wait_for(task, 5), 130)
                    self.assertTrue(cleaned.is_set())
                    self.assertEqual(await runtime.ahistory(conversation_id), [])
                finally:
                    cleanup_release.set()
                    session.request_exit()
                    await asyncio.gather(task, return_exceptions=True)
