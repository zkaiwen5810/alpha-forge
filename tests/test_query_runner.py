"""The query runner translates protocol messages into presentation notifications."""

import unittest

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.application import events as application_events
from alpha_forge.config import Config
from alpha_forge.providers import ProviderOutput, TextDelta, ToolCall
from alpha_forge.query import protocol as query
from alpha_forge.sessions import Session
from tests.test_query import ScriptedProvider, _text


class EmittingEngine:
    def __init__(self, messages):
        self.messages = messages

    async def run(self, request):
        for message in self.messages:
            feedback = yield message
            if feedback is not None:
                raise AssertionError("progress must not receive feedback")


class QueryRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_commit_feedback_follows_persistence_view_and_notification(self):
        session = Session.create(in_memory=True)
        self.addCleanup(session.close)
        prompt = session.accept_prompt("hello")
        received = []
        checked = []
        test = self

        class CommittingEngine:
            async def run(self, request):
                context = yield query.PrepareContext(request.prompt_event_id)
                test.assertIsInstance(context, query.ContextPrepared)
                output = yield query.CommitModelOutput(
                    request.prompt_event_id,
                    ProviderOutput((ToolCall("call", "echo", "{}"),)),
                )
                test.assertIsInstance(output, query.ModelOutputCommitted)
                record = session.transcript.records[-1]
                test.assertEqual(output.output_event_id, record.event_id)
                test.assertEqual(output.revision, session.revision)
                test.assertEqual(
                    received[-2:],
                    [
                        application_events.SessionViewChanged(
                            coordinator.session_view, reset_active=True
                        ),
                        application_events.ModelOutputRecorded(record.event_id),
                    ],
                )
                checked.append("model feedback")
                result = yield query.CommitToolResult(
                    output.output_event_id, "call", "success", "done"
                )
                test.assertIsInstance(result, query.ToolResultCommitted)
                record = session.transcript.records[-1]
                test.assertEqual(result.result_event_id, record.event_id)
                test.assertEqual(result.revision, session.revision)
                test.assertEqual(
                    received[-2:],
                    [
                        application_events.SessionViewChanged(
                            coordinator.session_view, reset_active=True
                        ),
                        application_events.ToolResultRecorded(
                            record.event_id, output.output_event_id, "call"
                        ),
                    ],
                )
                checked.append("tool feedback")

        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=ScriptedProvider([]),
            session=session,
            query_engine=CommittingEngine(),
        )
        coordinator.event_router.subscribe(
            application_events.ApplicationEvent, received.append
        )
        runner = coordinator._query_runner
        await runner.run(session, runner.prepare_request(session, prompt.event_id))
        self.assertEqual(checked, ["model feedback", "tool feedback"])

    async def test_all_progress_is_translated_for_application_subscribers(self):
        output = _text("hello")
        delta = TextDelta("hello")
        call = ToolCall("call", "echo", "{}")
        messages = [
            query.ProviderRequestStarted("prompt", "request"),
            query.ProviderDeltaReceived("request", delta),
            query.ProviderResponseCompleted("request", output),
            query.ToolCallProcessingStarted("output", call),
            query.QueryCompleted("prompt", "output"),
        ]
        session = Session.create(in_memory=True)
        self.addCleanup(session.close)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=ScriptedProvider([]),
            session=session,
            query_engine=EmittingEngine(messages),
        )
        received = []
        coordinator.event_router.subscribe(
            application_events.ApplicationEvent, received.append
        )
        runner = coordinator._query_runner
        await runner.run(session, runner.prepare_request(session, "prompt"))

        self.assertEqual(
            received,
            [
                application_events.ResponseStreamStarted("prompt", "request"),
                application_events.ResponseStreamUpdated("request", delta),
                application_events.ResponseStreamCompleted("request", output),
                application_events.ToolCallProcessingStarted("output", call),
                application_events.StatusChanged("Ready"),
            ],
        )
        self.assertTrue(
            all(not isinstance(event, query.QueryMessage) for event in received)
        )
        self.assertEqual(session.revision, 1)

    async def test_unknown_messages_and_progress_fail_explicitly(self):
        for message, description in (
            (query.QueryMessage(), "unsupported query message"),
            (query.QueryProgress(), "unsupported query progress"),
            (query.QueryEffect(), "unsupported query effect"),
        ):
            with self.subTest(message=type(message).__name__):
                session = Session.create(in_memory=True)
                self.addCleanup(session.close)
                coordinator = ApplicationCoordinator(
                    Config("key"),
                    provider=ScriptedProvider([]),
                    session=session,
                    query_engine=EmittingEngine([message]),
                )
                runner = coordinator._query_runner
                with self.assertRaisesRegex(
                    query.QueryExecutionError, description
                ) as caught:
                    await runner.run(session, runner.prepare_request(session, "prompt"))
                self.assertEqual(caught.exception.stage, "internal")
