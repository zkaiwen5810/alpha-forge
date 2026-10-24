import asyncio
import os
import sqlite3
import tempfile
import unittest
from contextlib import aclosing
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from alpha_forge_langchain import (
    AgentApplication,
    DatabaseBusyError,
    PersistenceError,
    ThreadNotFoundError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        """Override BaseChatModel._llm_type to identify this offline model."""
        return "scripted-offline-model"

    def bind_tools(self, tools, **kwargs):
        """Override BaseChatModel.bind_tools; scripted responses specify tool calls."""
        self.bound_tools = [item.name for item in tools]
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
        return self._generate(
            messages, stop, run_manager.get_sync() if run_manager else None, **kwargs
        )


async def astream(runtime, conversation_id, input):
    """Adapt text fixtures to the application factory and native query input."""
    if not isinstance(input, str) or not input.strip():
        raise ValueError("input must be a non-empty string")
    agent = await runtime.agents.acreate()
    async with aclosing(
        runtime.query.asubmit(
            conversation_id,
            agent=agent,
            input={"messages": [HumanMessage(content=input)]},
        )
    ) as stream:
        async for part in stream:
            yield part


async def aconsume(runtime, conversation_id, input):
    """Collect committed values in tests without a second runtime execution API."""
    output = None
    async with aclosing(astream(runtime, conversation_id, input)) as stream:
        async for part in stream:
            if part["type"] == "values":
                output = part["data"]
    if output is None:
        raise AssertionError("no committed values")
    return output


def call(name="calculator", arguments=None, call_id="call-1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments or {"expression": "(2 + 3) * 4"},
                "id": call_id,
            }
        ],
    )


class AgentApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Override IsolatedAsyncioTestCase.asyncSetUp to create a fresh database path."""
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "conversations.sqlite3"

    async def asyncTearDown(self):
        """Override IsolatedAsyncioTestCase.asyncTearDown to remove temporary files."""
        self.temporary.cleanup()

    async def test_multi_turn_tools_and_restart(self):
        model = ScriptedModel(
            responses=[call(), AIMessage(content="20"), AIMessage(content="Again: 20")]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            conversation = await runtime.threads.acreate()
            result = await aconsume(runtime, conversation, "Calculate")
            self.assertEqual(result["messages"][-1].text, "20")
            self.assertEqual(
                [m.type for m in result["messages"]], ["human", "ai", "tool", "ai"]
            )
            self.assertEqual(result["messages"][2].content, "20")
            await aconsume(runtime, conversation, "Repeat")
            history = await runtime.threads.ahistory(conversation)
            self.assertEqual(
                [m.type for m in history], ["human", "ai", "tool", "ai", "human", "ai"]
            )
            self.assertEqual(
                [m.content for m in model.seen[-1][1:]],
                [m.content for m in history[:-1]],
            )
            # Returned history/results must not expose persisted mutable objects.
            result["messages"][0].content = "mutated"
            history[0].content = "mutated"
            self.assertEqual(
                (await runtime.threads.ahistory(conversation))[0].content, "Calculate"
            )
        next_model = ScriptedModel(responses=[AIMessage(content="Still 20")])
        async with AgentApplication(self.database, model=next_model) as runtime:
            info = await runtime.threads.aget(conversation)
            self.assertFalse(info.unfinished_work_excluded)
            self.assertEqual(next_model.seen, [])
            self.assertEqual(len(await runtime.threads.ahistory(conversation)), 6)
            result = await aconsume(runtime, conversation, "Remember?")
            self.assertEqual(
                len(next_model.seen[0]), 8
            )  # system + six saved + new user
            self.assertEqual(result["messages"][-1].text, "Still 20")
            self.assertEqual(len(result["messages"]), 8)

    async def test_injected_tool_and_error_result(self):
        @tool
        def explode(value: str) -> str:
            """Raise an ordinary tool failure."""
            raise ValueError(f"cannot process {value}")

        model = ScriptedModel(
            responses=[
                call("explode", {"value": "test"}),
                AIMessage(content="Tool failed"),
                call(arguments={"expression": "1 / 0"}, call_id="call-2"),
                AIMessage(content="Division failed"),
            ]
        )
        async with AgentApplication(
            self.database, model=model, tools=[explode]
        ) as runtime:
            conversation = await runtime.threads.acreate()
            result = await aconsume(runtime, conversation, "Use custom tool")
            self.assertEqual(result["messages"][2].status, "error")
            self.assertIn("cannot process test", result["messages"][2].content)
            result = await aconsume(runtime, conversation, "Divide")
            self.assertEqual(result["messages"][-2].status, "error")
            self.assertEqual(result["messages"][-2].tool_call_id, "call-2")

    async def test_failures_restore_completed_head_without_retry(self):
        actions = []

        @tool
        def record(value: str) -> str:
            """Record an action for testing recovery."""
            actions.append(value)
            return "recorded"

        model = ScriptedModel(
            responses=[
                AIMessage(content="baseline"),
                call("record", {"value": "once"}),
                RuntimeError("provider failed"),
            ]
        )
        async with AgentApplication(
            self.database, model=model, tools=[record]
        ) as runtime:
            conversation = await runtime.threads.acreate()
            await aconsume(runtime, conversation, "First")
            head = (await runtime.threads.aget(conversation)).checkpoint_id
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await aconsume(runtime, conversation, "Unfinished")
            self.assertEqual(actions, ["once"])
            self.assertEqual(
                (await runtime.threads.aget(conversation)).checkpoint_id, head
            )
        model = ScriptedModel(responses=[AIMessage(content="fresh")])
        async with AgentApplication(
            self.database, model=model, tools=[record]
        ) as runtime:
            self.assertTrue(
                (await runtime.threads.aget(conversation)).unfinished_work_excluded
            )
            self.assertEqual(model.seen, [])
            self.assertEqual(
                [m.content for m in await runtime.threads.ahistory(conversation)],
                ["First", "baseline"],
            )
            await aconsume(runtime, conversation, "New prompt")
            self.assertEqual(actions, ["once"])
            self.assertEqual(
                [m.content for m in model.seen[0][1:]],
                [
                    "First",
                    "baseline",
                    "New prompt",
                ],
            )
            self.assertTrue(
                model.seen[0][0].text.startswith(
                    "You are Alpha Forge, a concise and helpful assistant."
                )
            )
            self.assertFalse(
                (await runtime.threads.aget(conversation)).unfinished_work_excluded
            )

    async def test_failed_first_turn_gets_empty_history_and_stable_thread(self):
        model = ScriptedModel(
            responses=[RuntimeError("failed"), AIMessage(content="new")]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            conversation = await runtime.threads.acreate()
            with self.assertRaises(RuntimeError):
                await aconsume(runtime, conversation, "old")
            abandoned_thread = (await runtime.threads.aget(conversation)).thread_id
            self.assertEqual(await runtime.threads.ahistory(conversation), [])
            await aconsume(runtime, conversation, "fresh")
            self.assertEqual(
                (await runtime.threads.aget(conversation)).thread_id,
                abandoned_thread,
            )
            self.assertEqual([m.content for m in model.seen[-1][1:]], ["fresh"])

    async def test_limit_and_budget_reset(self):
        model = ScriptedModel(
            responses=[
                *[call(call_id=f"call-{i}") for i in range(10)],
                AIMessage(content="recovered"),
            ]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            conversation = await runtime.threads.acreate()
            with self.assertRaisesRegex(Exception, "limit"):
                await aconsume(runtime, conversation, "loop")
            self.assertEqual(len(model.seen), 10)
            self.assertEqual(await runtime.threads.ahistory(conversation), [])
            await aconsume(runtime, conversation, "fresh")
            self.assertEqual(len(model.seen), 11)

    async def test_promotion_failure_excludes_finished_unpromoted_checkpoint(self):
        model = ScriptedModel(
            responses=[AIMessage(content="baseline"), AIMessage(content="not promoted")]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            conversation = await runtime.threads.acreate()
            await aconsume(runtime, conversation, "first")
            with (
                patch.object(
                    runtime._persistence._threads,
                    "apromote",
                    side_effect=sqlite3.OperationalError("disk failed"),
                ),
                self.assertRaises(PersistenceError),
            ):
                await aconsume(runtime, conversation, "second")
            with self.assertRaisesRegex(PersistenceError, "reopen"):
                await aconsume(runtime, conversation, "blocked")
        async with AgentApplication(self.database) as runtime:
            info = await runtime.threads.aget(conversation)
            self.assertTrue(info.unfinished_work_excluded)
            self.assertEqual(
                [m.content for m in await runtime.threads.ahistory(conversation)],
                ["first", "baseline"],
            )

    async def test_checkpoint_write_failure_preserves_head(self):
        model = ScriptedModel(
            responses=[AIMessage(content="baseline"), AIMessage(content="next")]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            conversation = await runtime.threads.acreate()
            await aconsume(runtime, conversation, "first")
            with (
                patch.object(
                    runtime._persistence._checkpointer,
                    "aput",
                    side_effect=sqlite3.OperationalError("checkpoint failed"),
                ),
                self.assertRaises(PersistenceError),
            ):
                await aconsume(runtime, conversation, "second")
        async with AgentApplication(self.database) as runtime:
            self.assertEqual(
                [m.content for m in await runtime.threads.ahistory(conversation)],
                ["first", "baseline"],
            )

    async def test_cancelled_turn_excluded_and_resources_released(self):
        entered = asyncio.Event()
        never = asyncio.Event()

        class WaitingModel(ScriptedModel):
            async def _agenerate(
                self, messages, stop=None, run_manager=None, **kwargs
            ) -> ChatResult:
                """Override ScriptedModel._agenerate to pause until the test cancels it."""
                entered.set()
                await never.wait()
                raise AssertionError("waiting model should always be cancelled")

        async with AgentApplication(self.database, model=WaitingModel()) as runtime:
            conversation = await runtime.threads.acreate()
            task = asyncio.create_task(aconsume(runtime, conversation, "wait"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        async with AgentApplication(self.database) as runtime:
            self.assertTrue(
                (await runtime.threads.aget(conversation)).unfinished_work_excluded
            )
            self.assertEqual(await runtime.threads.ahistory(conversation), [])

    async def test_isolation_validation_and_keyless_inspection(self):
        with patch.dict(os.environ, {}, clear=True):
            async with AgentApplication(self.database) as runtime:
                conversation = await runtime.threads.acreate()
                self.assertEqual(await runtime.threads.ahistory(conversation), [])
                await runtime.threads.aget(conversation)
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                    await aconsume(runtime, conversation, "hello")
                self.assertFalse(
                    (await runtime.threads.aget(conversation)).unfinished_work_excluded
                )
                with self.assertRaises(ThreadNotFoundError):
                    await runtime.threads.ahistory("missing")
                with self.assertRaises(ValueError):
                    await aconsume(runtime, conversation, "  ")
        model = ScriptedModel(
            responses=[AIMessage(content="private"), AIMessage(content="separate")]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            a, b = (
                await runtime.threads.acreate(),
                await runtime.threads.acreate(),
            )
            await aconsume(runtime, a, "secret")
            await aconsume(runtime, b, "hello")
            self.assertEqual([m.content for m in model.seen[-1][1:]], ["hello"])

    async def test_lock_lifecycle_and_missing_checkpoint(self):
        runtime = AgentApplication(
            self.database, model=ScriptedModel(responses=[AIMessage(content="ok")])
        )
        with self.assertRaises(RuntimeError):
            await runtime.threads.acreate()
        async with runtime:
            with self.assertRaises(DatabaseBusyError):
                async with AgentApplication(self.database):
                    pass
            conversation = await runtime.threads.acreate()
            await aconsume(runtime, conversation, "hello")
            with (
                patch.object(
                    runtime._persistence._checkpointer, "aget_tuple", return_value=None
                ),
                self.assertRaisesRegex(PersistenceError, "missing"),
            ):
                await runtime.threads.aget(conversation)
        async with AgentApplication(self.database) as runtime:
            await runtime.threads.aget(conversation)

    async def test_operations_are_serialized(self):
        model = ScriptedModel(
            responses=[AIMessage(content="one"), AIMessage(content="two")]
        )
        async with AgentApplication(self.database, model=model) as runtime:
            conversation = await runtime.threads.acreate()
            await asyncio.gather(
                aconsume(runtime, conversation, "first"),
                aconsume(runtime, conversation, "second"),
            )
            self.assertEqual(
                [m.content for m in await runtime.threads.ahistory(conversation)],
                ["first", "one", "second", "two"],
            )

    async def test_real_openai_adapter_offline_and_client_cleanup(self):
        import json

        import httpx
        from openai import AsyncOpenAI, OpenAI

        requests = []
        clients = []

        def respond(request):
            requests.append(request)
            chunks = []
            for text, finish_reason in [
                ("SDK ", None),
                ("response", None),
                ("", "stop"),
            ]:
                chunks.append(
                    {
                        "id": "chatcmpl-offline",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "DeepSeek-V4-Flash",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                )
            chunks.append(
                {
                    "id": "chatcmpl-offline",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "DeepSeek-V4-Flash",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                }
            )
            content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content + "data: [DONE]\n\n",
            )

        def sync_client(**kwargs):
            client = httpx.Client(transport=httpx.MockTransport(respond))
            clients.append(client)
            return OpenAI(**kwargs, http_client=client)

        def async_client(**kwargs):
            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            clients.append(client)
            return AsyncOpenAI(**kwargs, http_client=client)

        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "offline-not-a-real-key",
                    "OPENAI_BASE_URL": "https://ignored.example/v1",
                    "OPENAI_MODEL": "ignored-model",
                },
            ),
            patch("alpha_forge_langchain.agents.OpenAI", side_effect=sync_client),
            patch(
                "alpha_forge_langchain.agents.AsyncOpenAI",
                side_effect=async_client,
            ),
        ):
            async with AgentApplication(self.database) as runtime:
                conversation = await runtime.threads.acreate()
                result = await aconsume(runtime, conversation, "hello")
                self.assertEqual(result["messages"][-1].text, "SDK response")
                self.assertEqual(
                    result["messages"][-1].usage_metadata["total_tokens"], 6
                )
                self.assertEqual(
                    str(requests[0].url), "http://litellm:4000/v1/chat/completions"
                )
                payload = json.loads(requests[0].content)
                self.assertEqual(payload["model"], "DeepSeek-V4-Flash")
                self.assertTrue(payload["stream"])
            self.assertEqual(
                {item["function"]["name"] for item in payload["tools"]},
                {
                    "calculator",
                    "ls",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "glob",
                    "grep",
                    "execute",
                    "requests_get",
                },
            )
            self.assertTrue(all(client.is_closed for client in clients))

    async def test_completed_model_budget_resets_and_runtime_can_reopen(self):
        model = ScriptedModel(
            responses=[
                *[call(call_id=f"call-{i}") for i in range(9)],
                AIMessage(content="ten calls"),
                AIMessage(content="next turn"),
            ]
        )
        runtime = AgentApplication(self.database, model=model)
        async with runtime:
            conversation = await runtime.threads.acreate()
            await aconsume(runtime, conversation, "ten")
            self.assertEqual(len(model.seen), 10)
        async with runtime:
            await aconsume(runtime, conversation, "next")
            self.assertEqual(len(model.seen), 11)


if __name__ == "__main__":
    unittest.main()
