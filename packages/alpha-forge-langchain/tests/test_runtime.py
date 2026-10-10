import asyncio
import os
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from alpha_forge_langchain import (
    AgentRuntime,
    ConversationNotFoundError,
    DatabaseBusyError,
    PersistenceError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        """Override BaseChatModel._llm_type to identify this offline model."""
        return "scripted-offline-model"

    def bind_tools(self, tools, **kwargs):
        """Override BaseChatModel.bind_tools; scripted responses specify tool calls."""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override BaseChatModel._generate to record inputs and return scripted output."""
        self.seen.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override BaseChatModel._agenerate using the same offline response script."""
        return self._generate(messages, stop, run_manager, **kwargs)


def call(name="calculator", arguments=None, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": arguments or {"expression": "(2 + 3) * 4"},
        "id": call_id,
    }])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Override IsolatedAsyncioTestCase.asyncSetUp to create a fresh database path."""
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "conversations.sqlite3"

    async def asyncTearDown(self):
        """Override IsolatedAsyncioTestCase.asyncTearDown to remove temporary files."""
        self.temporary.cleanup()

    async def test_multi_turn_tools_and_restart(self):
        model = ScriptedModel(responses=[call(), AIMessage(content="20"), AIMessage(content="Again: 20")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.create_conversation()
            result = await runtime.send(conversation, "Calculate")
            self.assertEqual(result.assistant_text, "20")
            self.assertEqual([m.type for m in result.messages], ["human", "ai", "tool", "ai"])
            self.assertEqual(result.messages[2].content, "20")
            await runtime.send(conversation, "Repeat")
            history = await runtime.history(conversation)
            self.assertEqual([m.type for m in history], ["human", "ai", "tool", "ai", "human", "ai"])
            self.assertEqual([m.content for m in model.seen[-1][1:]], [m.content for m in history[:-1]])
            # Returned history/results must not expose persisted mutable objects.
            result.messages[0].content = "mutated"
            history[0].content = "mutated"
            self.assertEqual((await runtime.history(conversation))[0].content, "Calculate")
        next_model = ScriptedModel(responses=[AIMessage(content="Still 20")])
        async with AgentRuntime(self.database, model=next_model) as runtime:
            info = await runtime.restore_conversation(conversation)
            self.assertFalse(info.unfinished_work_excluded)
            self.assertEqual(next_model.seen, [])
            self.assertEqual(len(await runtime.history(conversation)), 6)
            result = await runtime.send(conversation, "Remember?")
            self.assertEqual(len(next_model.seen[0]), 8)  # system + six saved + new user
            self.assertEqual(result.assistant_text, "Still 20")
            self.assertEqual(len(result.messages), 2)

    async def test_injected_tool_and_error_result(self):
        @tool
        def explode(value: str) -> str:
            """Raise an ordinary tool failure."""
            raise ValueError(f"cannot process {value}")

        model = ScriptedModel(responses=[
            call("explode", {"value": "test"}), AIMessage(content="Tool failed"),
            call(arguments={"expression": "1 / 0"}, call_id="call-2"), AIMessage(content="Division failed"),
        ])
        async with AgentRuntime(self.database, model=model, tools=[explode]) as runtime:
            conversation = await runtime.create_conversation()
            result = await runtime.send(conversation, "Use custom tool")
            self.assertEqual(result.messages[2].status, "error")
            self.assertIn("cannot process test", result.messages[2].content)
            result = await runtime.send(conversation, "Divide")
            self.assertEqual(result.messages[2].status, "error")
            self.assertEqual(result.messages[2].tool_call_id, "call-2")

    async def test_failures_restore_completed_head_without_retry(self):
        actions = []

        @tool
        def record(value: str) -> str:
            """Record an action for testing recovery."""
            actions.append(value)
            return "recorded"

        model = ScriptedModel(responses=[
            AIMessage(content="baseline"), call("record", {"value": "once"}), RuntimeError("provider failed"),
        ])
        async with AgentRuntime(self.database, model=model, tools=[record]) as runtime:
            conversation = await runtime.create_conversation()
            await runtime.send(conversation, "First")
            head = (await runtime.restore_conversation(conversation)).checkpoint_id
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await runtime.send(conversation, "Unfinished")
            self.assertEqual(actions, ["once"])
            self.assertEqual((await runtime.restore_conversation(conversation)).checkpoint_id, head)
        model = ScriptedModel(responses=[AIMessage(content="fresh")])
        async with AgentRuntime(self.database, model=model, tools=[record]) as runtime:
            self.assertTrue((await runtime.restore_conversation(conversation)).unfinished_work_excluded)
            self.assertEqual(model.seen, [])
            self.assertEqual([m.content for m in await runtime.history(conversation)], ["First", "baseline"])
            await runtime.send(conversation, "New prompt")
            self.assertEqual(actions, ["once"])
            self.assertEqual([m.content for m in model.seen[0]], [
                "You are Alpha Forge, a concise and helpful assistant.", "First", "baseline", "New prompt",
            ])
            self.assertFalse((await runtime.restore_conversation(conversation)).unfinished_work_excluded)

    async def test_failed_first_turn_gets_empty_history_and_fresh_thread(self):
        model = ScriptedModel(responses=[RuntimeError("failed"), AIMessage(content="new")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.create_conversation()
            with self.assertRaises(RuntimeError):
                await runtime.send(conversation, "old")
            abandoned_thread = (await runtime.restore_conversation(conversation)).thread_id
            self.assertEqual(await runtime.history(conversation), [])
            await runtime.send(conversation, "fresh")
            self.assertNotEqual((await runtime.restore_conversation(conversation)).thread_id, abandoned_thread)
            self.assertEqual([m.content for m in model.seen[-1][1:]], ["fresh"])

    async def test_limit_and_budget_reset(self):
        model = ScriptedModel(responses=[*[call(call_id=f"call-{i}") for i in range(10)], AIMessage(content="recovered")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.create_conversation()
            with self.assertRaisesRegex(Exception, "limit"):
                await runtime.send(conversation, "loop")
            self.assertEqual(len(model.seen), 10)
            self.assertEqual(await runtime.history(conversation), [])
            await runtime.send(conversation, "fresh")
            self.assertEqual(len(model.seen), 11)

    async def test_promotion_failure_excludes_finished_unpromoted_checkpoint(self):
        model = ScriptedModel(responses=[AIMessage(content="baseline"), AIMessage(content="not promoted")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.create_conversation()
            await runtime.send(conversation, "first")
            with patch.object(runtime._store, "promote", side_effect=sqlite3.OperationalError("disk failed")):
                with self.assertRaises(PersistenceError):
                    await runtime.send(conversation, "second")
            with self.assertRaisesRegex(PersistenceError, "reopen"):
                await runtime.send(conversation, "blocked")
        async with AgentRuntime(self.database) as runtime:
            info = await runtime.restore_conversation(conversation)
            self.assertTrue(info.unfinished_work_excluded)
            self.assertEqual([m.content for m in await runtime.history(conversation)], ["first", "baseline"])

    async def test_checkpoint_write_failure_preserves_head(self):
        model = ScriptedModel(responses=[AIMessage(content="baseline"), AIMessage(content="next")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.create_conversation()
            await runtime.send(conversation, "first")
            with patch.object(runtime._saver, "aput", side_effect=sqlite3.OperationalError("checkpoint failed")):
                with self.assertRaises(PersistenceError):
                    await runtime.send(conversation, "second")
        async with AgentRuntime(self.database) as runtime:
            self.assertEqual([m.content for m in await runtime.history(conversation)], ["first", "baseline"])

    async def test_cancelled_turn_excluded_and_resources_released(self):
        entered = asyncio.Event()
        never = asyncio.Event()

        class WaitingModel(ScriptedModel):
            async def _agenerate(self, messages, **kwargs):
                """Override ScriptedModel._agenerate to pause until the test cancels it."""
                entered.set()
                await never.wait()

        async with AgentRuntime(self.database, model=WaitingModel()) as runtime:
            conversation = await runtime.create_conversation()
            task = asyncio.create_task(runtime.send(conversation, "wait"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        async with AgentRuntime(self.database) as runtime:
            self.assertTrue((await runtime.restore_conversation(conversation)).unfinished_work_excluded)
            self.assertEqual(await runtime.history(conversation), [])

    async def test_isolation_validation_and_keyless_inspection(self):
        with patch.dict(os.environ, {}, clear=True):
            async with AgentRuntime(self.database) as runtime:
                conversation = await runtime.create_conversation()
                self.assertEqual(await runtime.history(conversation), [])
                await runtime.restore_conversation(conversation)
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                    await runtime.send(conversation, "hello")
                self.assertFalse((await runtime.restore_conversation(conversation)).unfinished_work_excluded)
                with self.assertRaises(ConversationNotFoundError):
                    await runtime.history("missing")
                with self.assertRaises(ValueError):
                    await runtime.send(conversation, "  ")
        model = ScriptedModel(responses=[AIMessage(content="private"), AIMessage(content="separate")])
        async with AgentRuntime(self.database, model=model) as runtime:
            a, b = await runtime.create_conversation(), await runtime.create_conversation()
            await runtime.send(a, "secret")
            await runtime.send(b, "hello")
            self.assertEqual([m.content for m in model.seen[-1][1:]], ["hello"])

    async def test_lock_lifecycle_and_missing_checkpoint(self):
        runtime = AgentRuntime(self.database, model=ScriptedModel(responses=[AIMessage(content="ok")]))
        with self.assertRaises(RuntimeError):
            await runtime.create_conversation()
        async with runtime:
            with self.assertRaises(DatabaseBusyError):
                async with AgentRuntime(self.database):
                    pass
            conversation = await runtime.create_conversation()
            await runtime.send(conversation, "hello")
            with patch.object(runtime._saver, "aget_tuple", return_value=None):
                with self.assertRaisesRegex(PersistenceError, "missing"):
                    await runtime.restore_conversation(conversation)
        async with AgentRuntime(self.database) as runtime:
            await runtime.restore_conversation(conversation)

    async def test_operations_are_serialized(self):
        model = ScriptedModel(responses=[AIMessage(content="one"), AIMessage(content="two")])
        async with AgentRuntime(self.database, model=model) as runtime:
            conversation = await runtime.create_conversation()
            await asyncio.gather(runtime.send(conversation, "first"), runtime.send(conversation, "second"))
            self.assertEqual([m.content for m in await runtime.history(conversation)], ["first", "one", "second", "two"])

    async def test_real_openai_adapter_offline_and_client_cleanup(self):
        import json

        import httpx
        from openai import AsyncOpenAI, OpenAI

        requests = []
        clients = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
                "model": "DeepSeek-V4-Flash", "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": "SDK response"},
                    "finish_reason": "stop",
                }], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            })

        def sync_client(**kwargs):
            client = httpx.Client(transport=httpx.MockTransport(respond))
            clients.append(client)
            return OpenAI(**kwargs, http_client=client)

        def async_client(**kwargs):
            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            clients.append(client)
            return AsyncOpenAI(**kwargs, http_client=client)

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-not-a-real-key",
            "OPENAI_BASE_URL": "https://ignored.example/v1",
            "OPENAI_MODEL": "ignored-model",
        }), patch("alpha_forge_langchain.runtime.OpenAI", side_effect=sync_client), patch(
            "alpha_forge_langchain.runtime.AsyncOpenAI", side_effect=async_client,
        ):
            async with AgentRuntime(self.database) as runtime:
                conversation = await runtime.create_conversation()
                result = await runtime.send(conversation, "hello")
                self.assertEqual(result.assistant_text, "SDK response")
                self.assertEqual(result.messages[-1].usage_metadata["total_tokens"], 6)
                self.assertEqual(str(requests[0].url), "http://litellm:4000/v1/chat/completions")
                payload = json.loads(requests[0].content)
                self.assertEqual(payload["model"], "DeepSeek-V4-Flash")
                self.assertEqual(payload["tools"][0]["function"]["name"], "calculator")
            self.assertTrue(all(client.is_closed for client in clients))

    async def test_completed_model_budget_resets_and_runtime_can_reopen(self):
        model = ScriptedModel(responses=[
            *[call(call_id=f"call-{i}") for i in range(9)],
            AIMessage(content="ten calls"), AIMessage(content="next turn"),
        ])
        runtime = AgentRuntime(self.database, model=model)
        async with runtime:
            conversation = await runtime.create_conversation()
            await runtime.send(conversation, "ten")
            self.assertEqual(len(model.seen), 10)
        async with runtime:
            await runtime.send(conversation, "next")
            self.assertEqual(len(model.seen), 11)


if __name__ == "__main__":
    unittest.main()
