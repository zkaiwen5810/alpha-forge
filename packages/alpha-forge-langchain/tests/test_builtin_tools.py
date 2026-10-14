"""Exercise ecosystem tools through the real streaming/checkpoint runtime."""

import os
import shlex
import sys
import tempfile
import unittest
from contextlib import aclosing
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from alpha_forge_langchain import AgentRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from test_runtime import ScriptedModel, aconsume, call

BUILT_INS = {
    "calculator",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "requests_get",
}


class BuiltinToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.database = self.root / "history.sqlite3"
        self.cwd = patch(
            "alpha_forge_langchain.runtime.Path.cwd", return_value=self.workspace
        )
        self.cwd.start()
        self.addCleanup(self.cwd.stop)

    async def invoke_steps(self, steps):
        model = ScriptedModel(
            responses=[
                *[
                    call(name, args, f"call-{index}")
                    for index, (name, args) in enumerate(steps)
                ],
                AIMessage(content="done"),
            ]
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.acreate_conversation()
            parts = []
            async with aclosing(runtime.asend(conversation, "Use the tools")) as stream:
                async for part in stream:
                    parts.append(part)
            self.assertEqual(model.bound_tools and set(model.bound_tools), BUILT_INS)
            self.assertEqual(parts[-1]["type"], "values")
            messages = parts[-1]["data"]["messages"]
            self.assertEqual(messages, await runtime.ahistory(conversation))
            results = [
                message for message in messages if isinstance(message, ToolMessage)
            ]
            self.assertEqual(len(results), len(steps))
            self.assertEqual(
                [message.name for message in results], [name for name, _ in steps]
            )
            self.assertEqual(
                [message.tool_call_id for message in results],
                [f"call-{index}" for index in range(len(steps))],
            )
            self.assertTrue(any(part["type"] == "updates" for part in parts))
        return results

    async def test_write_overwrite_edit_and_paginated_read(self):
        results = await self.invoke_steps(
            [
                (
                    "write_file",
                    {"file_path": "nested/note.txt", "content": "old\nold\n"},
                ),
                (
                    "write_file",
                    {
                        "file_path": "nested/note.txt",
                        "content": "first\nsecond\nthird\n",
                    },
                ),
                (
                    "edit_file",
                    {
                        "file_path": "nested/note.txt",
                        "old_string": "second",
                        "new_string": "changed",
                    },
                ),
                (
                    "read_file",
                    {
                        "file_path": str(self.workspace / "nested/note.txt"),
                        "offset": 1,
                        "limit": 1,
                    },
                ),
            ]
        )
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertEqual(
            (self.workspace / "nested/note.txt").read_text(), "first\nchanged\nthird\n"
        )
        self.assertIn("changed", results[-1].text)
        self.assertNotIn("third", results[-1].text)

    async def test_parent_home_and_launch_directory_paths(self):
        (self.root / "outside.txt").write_text("parent content")
        with patch.dict(os.environ, {"HOME": str(self.root)}):
            results = await self.invoke_steps(
                [
                    ("read_file", {"file_path": "../outside.txt"}),
                    ("read_file", {"file_path": "~/outside.txt"}),
                    ("ls", {"path": ".."}),
                ]
            )
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertIn("parent content", results[0].text)
        self.assertIn("parent content", results[1].text)
        self.assertIn("outside.txt", results[2].text)

    async def test_edit_errors_do_not_change_file_and_replace_all(self):
        target = self.workspace / "note.txt"
        target.write_text("same same")
        results = await self.invoke_steps(
            [
                (
                    "edit_file",
                    {
                        "file_path": "note.txt",
                        "old_string": "absent",
                        "new_string": "x",
                    },
                ),
                (
                    "edit_file",
                    {"file_path": "note.txt", "old_string": "same", "new_string": "x"},
                ),
                ("read_file", {"file_path": "note.txt"}),
                (
                    "edit_file",
                    {
                        "file_path": "note.txt",
                        "old_string": "same",
                        "new_string": "new",
                        "replace_all": True,
                    },
                ),
                ("read_file", {"file_path": "missing.txt"}),
            ]
        )
        self.assertEqual(
            [result.status for result in results],
            ["error", "error", "success", "success", "error"],
        )
        self.assertIn("same same", results[2].text)
        self.assertEqual(target.read_text(), "new new")

    async def test_parallel_mutations_to_equivalent_paths_are_rejected(self):
        model = ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "note.txt", "content": "first"},
                            "id": "first",
                        },
                        {
                            "name": "write_file",
                            "args": {"file_path": "./note.txt", "content": "second"},
                            "id": "second",
                        },
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.acreate_conversation()
            result = await aconsume(runtime, conversation, "write")
        messages = {
            message.tool_call_id: message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        }
        self.assertEqual(messages["first"].status, "success")
        self.assertEqual(messages["second"].status, "error")
        self.assertIn("parallel", messages["second"].text)
        self.assertEqual((self.workspace / "note.txt").read_text(), "first")

    async def test_glob_and_literal_grep(self):
        (self.workspace / "src").mkdir()
        (self.workspace / "src/main.py").write_text("needle.*\nneedle.*\n")
        (self.workspace / "other.txt").write_text("needleXYZ\n")
        results = await self.invoke_steps(
            [
                ("glob", {"pattern": "**/*.py"}),
                (
                    "grep",
                    {"pattern": "needle.*", "glob": "*.py", "output_mode": "content"},
                ),
                (
                    "grep",
                    {"pattern": "needle.*", "path": "src", "output_mode": "count"},
                ),
                ("grep", {"pattern": "missing"}),
                ("glob", {"pattern": "*.missing"}),
            ]
        )
        self.assertIn("main.py", results[0].text)
        self.assertIn("needle.*", results[1].text)
        self.assertNotIn("other.txt", results[1].text)
        self.assertIn("2", results[2].text)
        self.assertTrue(all(result.status == "success" for result in results))

    async def test_shell_host_directory_environment_and_exit(self):
        with patch.dict(os.environ, {"AF_TOOL_MARKER": "inherited"}):
            results = await self.invoke_steps(
                [
                    (
                        "execute",
                        {
                            "command": 'pwd; echo "$AF_TOOL_MARKER"; echo failure >&2; exit 7'
                        },
                    ),
                    ("execute", {"command": "cd .."}),
                    ("execute", {"command": "pwd"}),
                ]
            )
        self.assertIn(str(self.workspace), results[0].text)
        self.assertIn("inherited", results[0].text)
        self.assertIn("failure", results[0].text)
        self.assertIn("7", results[0].text)
        self.assertIn(str(self.workspace), results[2].text)

    async def test_shell_timeout_and_output_limit(self):
        executable = shlex.quote(sys.executable)
        results = await self.invoke_steps(
            [
                (
                    "execute",
                    {
                        "command": f'{executable} -c "import time; time.sleep(1.5)"',
                        "timeout": 1,
                    },
                ),
                ("execute", {"command": f"{executable} -c \"print('x' * 100100)\""}),
            ]
        )
        self.assertIn("timed out", results[0].text.lower())
        self.assertIn("truncated", results[1].text.lower())
        self.assertLess(len(results[1].text), 100300)

    async def test_get_bodies_redirect_and_session_cleanup(self):
        async def body(request):
            return web.Response(text='{"value": 42}', content_type="application/json")

        async def redirect(request):
            raise web.HTTPFound("/body")

        async def missing(request):
            return web.Response(text="missing body", status=404)

        app = web.Application()
        app.router.add_get("/body", body)
        app.router.add_get("/redirect", redirect)
        app.router.add_get("/missing", missing)
        async with TestServer(app) as server:
            results = await self.invoke_steps(
                [
                    ("requests_get", {"url": str(server.make_url("/body"))}),
                    ("requests_get", {"url": str(server.make_url("/redirect"))}),
                    ("requests_get", {"url": str(server.make_url("/missing"))}),
                ]
            )
        self.assertEqual(
            [result.text for result in results],
            ['{"value": 42}', '{"value": 42}', "missing body"],
        )
        runtime = AgentRuntime(self.database)
        async with runtime:
            session = runtime._web_fetch.requests_wrapper.aiosession
            self.assertEqual(session.timeout.total, 30.0)
        self.assertTrue(session.closed)
        self.assertIsNone(runtime._web_fetch.requests_wrapper.aiosession)
        async with runtime:
            self.assertIsNot(runtime._web_fetch.requests_wrapper.aiosession, session)

    async def test_get_connection_and_timeout_errors(self):
        model = ScriptedModel(
            responses=[
                call("requests_get", {"url": "https://example.invalid/"}),
                AIMessage(content="failed"),
            ]
        )
        for error in (
            ConnectionError("connection failed"),
            TimeoutError("fetch timed out"),
        ):
            model.responses = [
                call("requests_get", {"url": "https://example.invalid/"}),
                AIMessage(content="failed"),
            ]
            async with AgentRuntime(self.database, model=model) as runtime:
                conversation = await runtime.acreate_conversation()
                with patch.object(
                    runtime._web_fetch.requests_wrapper.aiosession,
                    "request",
                    side_effect=error,
                ):
                    result = await aconsume(runtime, conversation, "fetch")
                message = result["messages"][-2]
                self.assertEqual(message.status, "error")
                self.assertIn(str(error), message.text)

    def test_custom_tool_names_cannot_override_builtins(self):
        for name in BUILT_INS:

            @tool(name)
            def duplicate(value: str) -> str:
                """Echo text."""
                return value

            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unique"):
                AgentRuntime(self.database, tools=[duplicate])

    async def test_failed_write_is_not_replayed_after_restore(self):
        model = ScriptedModel(
            responses=[
                AIMessage(content="baseline"),
                call(
                    "write_file", {"file_path": "effect.txt", "content": "tool effect"}
                ),
                RuntimeError("provider failed"),
            ]
        )
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.acreate_conversation()
            await aconsume(runtime, conversation, "baseline")
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await aconsume(runtime, conversation, "write")
        target = self.workspace / "effect.txt"
        self.assertEqual(target.read_text(), "tool effect")
        target.write_text("external change")
        restored = ScriptedModel(responses=[AIMessage(content="fresh")])
        async with AgentRuntime(self.database, model=restored) as runtime:
            self.assertTrue(
                (
                    await runtime.arestore_conversation(conversation)
                ).unfinished_work_excluded
            )
            self.assertEqual(
                [message.text for message in await runtime.ahistory(conversation)],
                ["baseline", "baseline"],
            )
            self.assertEqual(restored.seen, [])
            await aconsume(runtime, conversation, "continue")
        self.assertEqual(target.read_text(), "external change")
