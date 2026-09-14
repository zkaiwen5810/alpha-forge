import asyncio
import unittest

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.config import Config
from alpha_forge.context import (
    ContextPipeline,
    ToolResultBudgetPolicy,
    ToolResultContext,
)
from alpha_forge.hooks import (
    Hook,
    HookRegistry,
    PermissionAction,
    match_tool_names,
)
from alpha_forge.providers import (
    OutputMessage,
    OutputText,
    ProviderOutput,
    StreamCompleted,
    TextDelta,
    ToolCall,
)
from alpha_forge.query import QueryEngine
from alpha_forge.sessions import Session
from alpha_forge.sessions.service import INTERRUPTED_TOOL_RESULT
from alpha_forge.tools import Tool, ToolRegistry
from alpha_forge.transcript import ContextEdited, QueryFailed, ToolResult


def _text(text: str) -> ProviderOutput:
    return ProviderOutput((OutputMessage((OutputText(text),)),), "stop")


class ScriptedProvider:
    def __init__(self, outputs: list[ProviderOutput | Exception]) -> None:
        self.outputs = outputs
        self.contexts = []
        self.tools = []

    def list_models(self) -> list[str]:
        return ["gpt-test"]

    async def stream(self, context, *, tools):
        self.contexts.append(context)
        self.tools.append(tools)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if output.output_text:
            yield TextDelta(output.output_text)
        yield StreamCompleted(output)


async def _consume_one(
    coordinator: ApplicationCoordinator,
    prompt: str | None,
) -> None:
    if prompt is not None:
        coordinator.submit(prompt)
    coordinator.request_exit()
    await coordinator.consume()


class QueryFlowTests(unittest.TestCase):
    def test_permission_denial_is_committed_as_error_tool_result(self) -> None:
        provider = ScriptedProvider(
            [
                ProviderOutput(
                    (ToolCall("call", "file_writer", '{"value":"x"}'),)
                ),
                _text("continued after denial"),
            ]
        )
        invoked: list[str] = []
        registry = ToolRegistry(
            [
                Tool(
                    name="file_writer",
                    description="write",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    handler=lambda _arguments: invoked.append("handler") or "done",
                )
            ]
        )

        async def deny(_event):  # type: ignore[no-untyped-def]
            return False

        hook_registry = HookRegistry(
            [
                Hook(
                    match_tool_names("file_writer"),
                    PermissionAction(deny),
                )
            ]
        )
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            tool_registry=registry,
            session=session,
            hook_registry=hook_registry,
        )

        asyncio.run(_consume_one(coordinator, "write"))

        result = next(
            event
            for event in session.transcript.events
            if isinstance(event, ToolResult)
        )
        self.assertEqual(result.status, "error")
        self.assertIn("permission denied", result.content)
        self.assertEqual(invoked, [])
        self.assertEqual(len(provider.contexts), 2)

    def test_multiple_provider_requests_use_fresh_committed_projection(self) -> None:
        provider = ScriptedProvider(
            [
                ProviderOutput(
                    (
                        OutputMessage((OutputText("checking"),)),
                        ToolCall("one", "echo", '{"value":"a"}'),
                        ToolCall("two", "echo", '{"value":"b"}'),
                    ),
                    "tool_calls",
                ),
                _text("done"),
            ]
        )
        registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    description="echo",
                    input_schema={"type": "object"},
                    handler=lambda arguments: str(arguments["value"]),
                )
            ]
        )
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key", model="gpt-test"),
            provider=provider,
            tool_registry=registry,
            session=session,
        )

        asyncio.run(_consume_one(coordinator, "go"))

        self.assertEqual(len(provider.contexts), 2)
        first_revision, second_revision = [
            context.revision for context in provider.contexts
        ]
        self.assertGreater(second_revision, first_revision)
        second_results = [
            item
            for item in provider.contexts[1].items
            if isinstance(item, ToolResultContext)
        ]
        self.assertEqual(
            [(result.call_id, result.content) for result in second_results],
            [("one", "a"), ("two", "b")],
        )
        durable = [type(event).__name__ for event in session.transcript.events]
        self.assertEqual(
            durable,
            [
                "SessionOpened",
                "InputAccepted",
                "ModelOutput",
                "ToolResult",
                "ToolResult",
                "ModelOutput",
            ],
        )
        self.assertFalse(
            any(isinstance(event, ContextEdited) for event in session.transcript.events)
        )

    def test_activation_interrupts_without_calling_provider_or_tools(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("recover")
        output = session.record_model_output(
            prompt.event_id,
            ProviderOutput(
                (
                    ToolCall("recorded", "echo", "{}"),
                    ToolCall("missing", "echo", "{}"),
                )
            ),
        )
        session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="recorded",
            status="success",
            content="already durable",
        )
        revision_before_open = session.revision
        invoked = []
        provider = ScriptedProvider([])
        registry = ToolRegistry([
            Tool(
                name="echo",
                description="echo",
                input_schema={"type": "object"},
                handler=lambda args: invoked.append(args) or "done",
            ),
        ])
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=session,
            tool_registry=registry,
            context_pipeline=ContextPipeline(),
        )
        self.assertEqual(session.revision, revision_before_open)

        asyncio.run(_consume_one(coordinator, None))

        results = [
            event
            for event in session.transcript.events
            if isinstance(event, ToolResult)
        ]
        self.assertEqual(
            [(result.call_id, result.status) for result in results],
            [("recorded", "success"), ("missing", "interrupted")],
        )
        self.assertEqual(results[-1].content, INTERRUPTED_TOOL_RESULT)
        self.assertEqual(provider.contexts, [])
        self.assertEqual(invoked, [])
        self.assertIsInstance(session.transcript.events[-1], QueryFailed)
        self.assertEqual(session.transcript.events[-1].stage, "interrupted")
        self.assertIsNone(session.transcript.state.active_prompt_event_id)

    def test_new_prompt_after_activation_gets_fresh_round_budget(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("old request")
        for index in range(2):
            output = session.record_model_output(
                prompt.event_id,
                ProviderOutput((ToolCall(str(index), "echo", "{}"),)),
            )
            session.record_tool_result(
                model_output_event_id=output.event_id,
                call_id=str(index),
                status="success",
                content="saved",
            )
        provider = ScriptedProvider([
            ProviderOutput((ToolCall("new", "echo", "{}"),)),
            _text("done"),
        ])
        invoked = []
        registry = ToolRegistry([
            Tool(
                name="echo",
                description="echo",
                input_schema={"type": "object"},
                handler=lambda args: invoked.append(args) or "new result",
            ),
        ])
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=session,
            tool_registry=registry,
            query_engine=QueryEngine(provider, max_intermediate_rounds=2),
        )
        asyncio.run(_consume_one(coordinator, "continue"))
        self.assertEqual(len(provider.contexts), 2)
        self.assertEqual(invoked, [{}])
        self.assertEqual(
            [item.content for item in provider.contexts[0].items
             if isinstance(item, ToolResultContext)],
            ["saved", "saved"],
        )
        self.assertEqual(session.transcript.events[-1].items, _text("done").items)
        failures = [e for e in session.transcript.events if isinstance(e, QueryFailed)]
        self.assertEqual([e.stage for e in failures], ["interrupted"])

    def test_provider_failure_is_durable_and_discards_stream_draft(self) -> None:
        provider = ScriptedProvider([RuntimeError("network down")])
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=session,
        )

        asyncio.run(_consume_one(coordinator, "hello"))

        failure = session.transcript.events[-1]
        self.assertIsInstance(failure, QueryFailed)
        self.assertEqual(failure.stage, "provider")
        self.assertEqual(failure.message, "network down")
        self.assertIsNone(session.transcript.state.active_prompt_event_id)

    def test_context_edit_is_committed_before_the_next_provider_request(self) -> None:
        provider = ScriptedProvider(
            [
                ProviderOutput((ToolCall("call", "large", "{}"),)),
                _text("done"),
            ]
        )
        registry = ToolRegistry(
            [
                Tool(
                    name="large",
                    description="return a large result",
                    input_schema={"type": "object"},
                    handler=lambda _arguments: "x" * 1000,
                )
            ]
        )
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=session,
            tool_registry=registry,
            context_pipeline=ContextPipeline(
                (
                    ToolResultBudgetPolicy(
                        individual_limit=300,
                        aggregate_limit=300,
                    ),
                )
            ),
        )

        asyncio.run(_consume_one(coordinator, "large"))

        self.assertEqual(len(provider.contexts), 2)
        projected_result = next(
            item
            for item in provider.contexts[1].items
            if isinstance(item, ToolResultContext)
        )
        self.assertEqual(len(projected_result.content), 300)
        event_names = [
            type(event).__name__ for event in session.transcript.events
        ]
        self.assertEqual(
            event_names,
            [
                "SessionOpened",
                "InputAccepted",
                "ModelOutput",
                "ToolResult",
                "ContextEdited",
                "ModelOutput",
            ],
        )

    def test_intermediate_round_limit_fails_after_complete_exchange(self) -> None:
        provider = ScriptedProvider(
            [ProviderOutput((ToolCall("call", "echo", '{"value":"x"}'),))]
        )
        registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    description="echo",
                    input_schema={"type": "object"},
                    handler=lambda arguments: str(arguments["value"]),
                )
            ]
        )
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=session,
            tool_registry=registry,
            query_engine=QueryEngine(provider, max_intermediate_rounds=1),
        )

        asyncio.run(_consume_one(coordinator, "loop"))

        self.assertIsInstance(session.transcript.events[-1], QueryFailed)
        self.assertEqual(
            session.transcript.events[-1].stage,
            "intermediate_round_limit",
        )
        result = next(
            event
            for event in session.transcript.events
            if isinstance(event, ToolResult)
        )
        self.assertEqual(result.status, "success")


if __name__ == "__main__":
    unittest.main()
