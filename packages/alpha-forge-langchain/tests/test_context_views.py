"""Model-input edits must preserve original transcript messages and checkpoints."""

import asyncio
import tempfile
import unittest
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from alpha_forge_langchain import AgentApplication
from alpha_forge_langchain.message_state import MessageState
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    wrap_model_call,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from test_agent_application import ScriptedModel, call


@tool(response_format="content_and_artifact")
async def lookup(record_id: str) -> tuple[str, dict]:
    """Retrieve the original text and structured artifact for a record."""
    return f"Original output for {record_id}. " * 40, {
        "nested": {"record": record_id, "tags": ["original"]}
    }


def editing(placeholder="[cleared]"):
    return ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=1,
                keep=0,
                clear_tool_inputs=True,
                placeholder=placeholder,
            )
        ]
    )


def lookup_response(record="record", call_id="lookup-1"):
    response = call("lookup", {"record_id": record}, call_id)
    response.response_metadata = {"nested": {"source": "original"}}
    return response


def tool_result(messages):
    return next(message for message in messages if isinstance(message, ToolMessage))


def tool_request(messages):
    return next(
        message
        for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
    )


async def submit(app, thread_id, agent, text, *, context=None):
    parts = []
    async with aclosing(
        app.query.asubmit(
            thread_id,
            agent=agent,
            input={"messages": [HumanMessage(content=text)]},
            context=context,
        )
    ) as stream:
        async for part in stream:
            parts.append(part)
    if not parts or parts[-1]["type"] != "values":
        raise AssertionError("no committed values")
    return parts[-1]["data"], parts


@dataclass
class ViewContext:
    marker: str


class ContextViewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / "context.sqlite3"

    async def test_native_edits_preserve_transcript_state_and_display(self):
        response = lookup_response()
        model = ScriptedModel(responses=[response, AIMessage(content="answer")])
        async with AgentApplication(self.database) as app:
            thread = await app.threads.acreate()
            agent = await app.agents.acreate(
                model=model, tools=[lookup], middleware=[editing()]
            )
            state, parts = await submit(app, thread, agent, "look up record")
            model_tool = tool_result(model.seen[-1])
            model_request = tool_request(model.seen[-1])
            self.assertEqual(model_tool.text, "[cleared]")
            self.assertIsNone(model_tool.artifact)
            self.assertEqual(model_request.tool_calls[0]["args"], {})
            self.assertEqual(model_request.tool_calls[0]["id"], model_tool.tool_call_id)

            history = await app.threads.ahistory(thread)
            self.assertEqual(
                [message.type for message in history], ["human", "ai", "tool", "ai"]
            )
            self.assertTrue(
                tool_result(history).text.startswith("Original output for record.")
            )
            self.assertEqual(
                tool_request(history).tool_calls[0]["args"], {"record_id": "record"}
            )
            self.assertEqual(
                tool_result(history).artifact,
                {"nested": {"record": "record", "tags": ["original"]}},
            )
            self.assertNotIn("context_editing", tool_result(history).response_metadata)
            self.assertEqual(state["messages"], history)
            self.assertEqual(
                (await app.query.aget_state(thread, agent=agent))["messages"], history
            )

            display = MessageState()
            for part in parts:
                display.update(part)
            self.assertEqual(tool_result(display.messages), tool_result(history))
            self.assertEqual(
                tool_request(display.messages).tool_calls,
                tool_request(history).tool_calls,
            )

    async def test_reopen_and_policy_changes_rederive_context_from_originals(self):
        async with AgentApplication(self.database) as app:
            thread = await app.threads.acreate()
            model = ScriptedModel(
                responses=[lookup_response(), AIMessage(content="first")]
            )
            agent = await app.agents.acreate(
                model=model, tools=[lookup], middleware=[editing()]
            )
            await submit(app, thread, agent, "first")
            original = await app.threads.ahistory(thread)

        async with AgentApplication(self.database) as app:
            self.assertEqual(await app.threads.ahistory(thread), original)
            unedited_model = ScriptedModel(responses=[AIMessage(content="second")])
            agent = await app.agents.acreate(model=unedited_model, tools=[lookup])
            await submit(app, thread, agent, "second")
            self.assertEqual(tool_result(unedited_model.seen[0]), tool_result(original))
            self.assertEqual(
                tool_request(unedited_model.seen[0]).tool_calls[0]["args"],
                {"record_id": "record"},
            )

            edited_model = ScriptedModel(responses=[AIMessage(content="third")])
            agent = await app.agents.acreate(
                model=edited_model, tools=[lookup], middleware=[editing("[new policy]")]
            )
            state, _ = await submit(app, thread, agent, "third")
            self.assertEqual(tool_result(edited_model.seen[0]).text, "[new policy]")
            history = await app.threads.ahistory(thread)
            self.assertEqual(tool_result(history), tool_result(original))
            self.assertEqual(state["messages"], history)
            self.assertEqual(len(history), 8)

    async def test_custom_async_edits_do_not_mutate_nested_transcript_data(self):
        @wrap_model_call
        async def custom_view(request, handler):
            messages = deepcopy(request.messages)
            for message in messages:
                if isinstance(message, ToolMessage):
                    message.content = "model-only output"
                    message.artifact["nested"]["tags"].append("model-only")
                elif isinstance(message, AIMessage) and message.tool_calls:
                    message.tool_calls[0]["args"]["record_id"] = "model-only argument"
                    message.response_metadata["nested"]["source"] = (
                        "model-only metadata"
                    )
            await asyncio.sleep(0)
            return await handler(request.override(messages=messages))

        model = ScriptedModel(
            responses=[lookup_response(), AIMessage(content="answer")]
        )
        async with AgentApplication(self.database) as app:
            thread = await app.threads.acreate()
            agent = await app.agents.acreate(
                model=model, tools=[lookup], middleware=[custom_view]
            )
            await submit(app, thread, agent, "input")
            self.assertEqual(tool_result(model.seen[-1]).text, "model-only output")
            self.assertEqual(
                tool_result(model.seen[-1]).artifact["nested"]["tags"],
                ["original", "model-only"],
            )
            self.assertEqual(
                tool_request(model.seen[-1]).tool_calls[0]["args"],
                {"record_id": "model-only argument"},
            )
            history = await app.threads.ahistory(thread)
            self.assertEqual(
                tool_result(history).artifact["nested"]["tags"], ["original"]
            )
            self.assertEqual(
                tool_request(history).response_metadata["nested"],
                {"source": "original"},
            )
            self.assertEqual(
                tool_request(history).tool_calls[0]["args"], {"record_id": "record"}
            )
            self.assertTrue(tool_result(history).text.startswith("Original output"))

    async def test_shared_policy_keeps_concurrent_thread_views_isolated(self):
        entered = {marker: asyncio.Event() for marker in ("left", "right")}
        release = asyncio.Event()

        @wrap_model_call
        async def per_thread_view(request, handler):
            messages = deepcopy(request.messages)
            if any(isinstance(message, ToolMessage) for message in messages):
                marker = request.runtime.context.marker
                tool_result(messages).content = f"context for {marker}"
                entered[marker].set()
                await release.wait()
            return await handler(request.override(messages=messages))

        async with AgentApplication(self.database) as app:
            threads = [await app.threads.acreate() for _ in entered]
            models = [
                ScriptedModel(
                    responses=[lookup_response(marker), AIMessage(content=marker)]
                )
                for marker in entered
            ]
            agents = [
                await app.agents.acreate(
                    model=model,
                    tools=[lookup],
                    middleware=[per_thread_view],
                    context_schema=ViewContext,
                )
                for model in models
            ]
            tasks = [
                asyncio.create_task(
                    submit(app, thread, agent, marker, context=ViewContext(marker))
                )
                for thread, agent, marker in zip(threads, agents, entered)
            ]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in entered.values())), 5
                )
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 5)
                for marker, model, thread in zip(entered, models, threads):
                    self.assertEqual(
                        tool_result(model.seen[-1]).text, f"context for {marker}"
                    )
                    history = await app.threads.ahistory(thread)
                    self.assertEqual(history[0].text, marker)
                    self.assertTrue(
                        tool_result(history).text.startswith(
                            f"Original output for {marker}."
                        )
                    )
            finally:
                release.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_policy_failure_preserves_transcript_and_committed_head(self):
        @wrap_model_call
        async def broken_policy(request, handler):
            messages = deepcopy(request.messages)
            tool_result(messages).content = "discarded context"
            raise RuntimeError("context policy failed")

        async with AgentApplication(self.database) as app:
            thread = await app.threads.acreate()
            agent = await app.agents.acreate(
                model=ScriptedModel(
                    responses=[lookup_response(), AIMessage(content="baseline")]
                ),
                tools=[lookup],
            )
            await submit(app, thread, agent, "first")
            original = await app.threads.ahistory(thread)
            head = (await app.threads.aget(thread)).checkpoint_id

            failed_model = ScriptedModel(responses=[AIMessage(content="must not run")])
            agent = await app.agents.acreate(
                model=failed_model, tools=[lookup], middleware=[broken_policy]
            )
            with self.assertRaisesRegex(RuntimeError, "context policy failed"):
                await submit(app, thread, agent, "failed input")
            self.assertEqual(failed_model.seen, [])
            self.assertEqual((await app.threads.aget(thread)).checkpoint_id, head)
            self.assertEqual(await app.threads.ahistory(thread), original)

            retry_model = ScriptedModel(responses=[AIMessage(content="recovered")])
            agent = await app.agents.acreate(model=retry_model, tools=[lookup])
            state, _ = await submit(app, thread, agent, "retry")
            self.assertEqual(
                [
                    message.text
                    for message in retry_model.seen[0]
                    if isinstance(message, HumanMessage)
                ],
                ["first", "retry"],
            )
            self.assertEqual(tool_result(state["messages"]), tool_result(original))
