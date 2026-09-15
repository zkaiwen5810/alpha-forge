"""Regression coverage for application ownership and failure boundaries."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.application.events import (
    ApplicationEvent,
    ExitReady,
    ModelOutputRecorded,
    PersistenceFailed,
    SessionViewChanged,
    ToolPermissionRequested,
    ToolPermissionResolved,
    ToolResultRecorded,
)
from alpha_forge.config import Config
from alpha_forge.context import ToolResultContext, UserMessage
from alpha_forge.hooks import PreToolExecution
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.providers import ProviderOutput, ToolCall
from alpha_forge.sessions import Session
from alpha_forge.sessions.tool_result_reader import ToolResultReader
from alpha_forge.tools import Tool, ToolRegistry
from alpha_forge.transcript import InputAccepted, QueryFailed, TranscriptPersistenceError
from tests.test_query import ScriptedProvider, _text


class ApplicationLifecycleTests(unittest.TestCase):
    def test_failed_model_or_tool_commit_stops_later_work_without_acknowledgment(self):
        for method, expected_invocations, expected_model_acks in (
            ("record_model_output", [], 0),
            ("record_tool_result", ["first"], 1),
        ):
            with self.subTest(method=method):
                invoked = []
                provider = ScriptedProvider(
                    [
                        ProviderOutput(
                            (
                                ToolCall("first", "echo", '{"value":"first"}'),
                                ToolCall("second", "echo", '{"value":"second"}'),
                            )
                        )
                    ]
                )
                session = Session.create(in_memory=True)
                registry = ToolRegistry(
                    [
                        Tool(
                            name="echo",
                            description="echo",
                            input_schema={"type": "object"},
                            handler=lambda args: (
                                invoked.append(args["value"]) or "done"
                            ),
                        )
                    ]
                )
                coordinator = ApplicationCoordinator(
                    Config("key"),
                    provider=provider,
                    session=session,
                    tool_registry=registry,
                )
                events = []
                coordinator.event_router.subscribe(ApplicationEvent, events.append)

                async def run():
                    coordinator.submit("first prompt")
                    coordinator.submit("must not run")
                    coordinator.request_exit()
                    await coordinator.consume()

                with patch.object(
                    session, method, side_effect=TranscriptPersistenceError("disk full")
                ):
                    asyncio.run(run())
                self.assertEqual(invoked, expected_invocations)
                self.assertEqual(len(provider.contexts), 1)
                self.assertEqual(
                    [
                        e.text
                        for e in session.transcript.events
                        if isinstance(e, InputAccepted)
                    ],
                    ["first prompt"],
                )
                self.assertEqual(
                    sum(isinstance(e, ModelOutputRecorded) for e in events),
                    expected_model_acks,
                )
                self.assertFalse(any(isinstance(e, ToolResultRecorded) for e in events))
                failures = [e for e in events if isinstance(e, PersistenceFailed)]
                self.assertEqual(
                    [(e.stage, e.message) for e in failures], [("query", "disk full")]
                )
                self.assertIsInstance(events[-1], ExitReady)
                self.assertFalse(coordinator.accepting)
                with self.assertRaisesRegex(TranscriptPersistenceError, "closed"):
                    session.accept_prompt("closed")

    def test_committed_view_precedes_model_acknowledgment(self):
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=ScriptedProvider([_text("done")]),
            session=session,
        )
        events = []
        coordinator.event_router.subscribe(ApplicationEvent, events.append)

        commit_snapshots = []

        def capture_commit(event):
            commit_snapshots.append(
                (
                    events[-2],
                    session.revision,
                    session.transcript.records[-1].event_id,
                    event.output_event_id,
                )
            )

        coordinator.event_router.subscribe(ModelOutputRecorded, capture_commit)

        async def run():
            coordinator.submit("hello")
            coordinator.request_exit()
            await coordinator.consume()

        asyncio.run(run())
        self.assertEqual(len(commit_snapshots), 1)
        view_event, revision, stored_id, acknowledged_id = commit_snapshots[0]
        self.assertIsInstance(view_event, SessionViewChanged)
        self.assertEqual(view_event.view.revision, revision)
        self.assertTrue(view_event.reset_active)
        self.assertEqual(stored_id, acknowledged_id)

    def test_shutdown_denies_pending_permission_once(self):
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"), provider=ScriptedProvider([]), session=session
        )
        events = []
        coordinator.event_router.subscribe(ApplicationEvent, events.append)
        lifecycle = PreToolExecution(
            call_id="call",
            tool_name="bash",
            tool_input=FrozenJsonObject({"cmd": "pwd"}),
        )

        async def run():
            pending = asyncio.create_task(
                coordinator.request_tool_permission(lifecycle)
            )
            await asyncio.sleep(0)
            request = next(e for e in events if isinstance(e, ToolPermissionRequested))
            self.assertFalse(coordinator.resolve_tool_permission("stale", True))
            with self.assertRaisesRegex(RuntimeError, "already pending"):
                await coordinator.request_tool_permission(lifecycle)
            coordinator.request_exit()
            coordinator.request_exit()
            self.assertFalse(await pending)
            self.assertFalse(
                coordinator.resolve_tool_permission(request.request_id, True)
            )
            await coordinator.consume()

        asyncio.run(run())
        resolutions = [e for e in events if isinstance(e, ToolPermissionResolved)]
        self.assertEqual([e.allowed for e in resolutions], [False])
        self.assertEqual(session.revision, 1)

    def test_resume_interrupts_before_queued_prompt_and_reader_uses_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.jsonl"
            saved = Session.create(transcript_path=path)
            prompt = saved.accept_prompt("saved prompt")
            output = saved.record_model_output(
                prompt.event_id,
                ProviderOutput(
                    (
                        ToolCall("stored", "echo", "{}"),
                        ToolCall("missing", "echo", "{}"),
                    )
                ),
            )
            result = saved.record_tool_result(
                model_output_event_id=output.event_id,
                call_id="stored",
                status="success",
                content="destination result",
            )
            saved.close()
            provider = ScriptedProvider(
                [
                    ProviderOutput(
                        (
                            ToolCall(
                                "read",
                                "tool_result_reader",
                                json.dumps(
                                    {
                                        "result_event_id": result.event_id,
                                    }
                                ),
                            ),
                        )
                    ),
                    _text("next answer"),
                ]
            )
            source = Session.create(in_memory=True)
            coordinator = ApplicationCoordinator(
                Config("key"), provider=provider, session=source
            )

            async def run():
                coordinator.submit(f"/resume {path}")
                coordinator.submit("next prompt")
                coordinator.request_exit()
                await coordinator.consume()

            asyncio.run(run())
            contexts = provider.contexts
            self.assertEqual(len(contexts), 2)
            self.assertEqual(
                [m.content for m in contexts[0].items if isinstance(m, UserMessage)],
                ["saved prompt", "next prompt"],
            )
            recovered = [
                m for m in contexts[0].items if isinstance(m, ToolResultContext)
            ]
            self.assertEqual([m.status for m in recovered], ["success", "interrupted"])
            read = next(
                m
                for m in contexts[1].items
                if isinstance(m, ToolResultContext) and m.call_id == "read"
            )
            self.assertTrue(read.content.startswith("destination result\n"))
            self.assertEqual(read.status, "success")
            self.assertEqual(
                [m.content for m in contexts[1].items if isinstance(m, UserMessage)],
                ["saved prompt", "next prompt"],
            )
            with self.assertRaisesRegex(TranscriptPersistenceError, "closed"):
                source.accept_prompt("closed")

    def test_resume_without_prompt_does_not_request_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            saved = Session.create(transcript_path=path)
            saved.accept_prompt("unfinished")
            saved.close()
            provider = ScriptedProvider([])
            coordinator = ApplicationCoordinator(
                Config("key"), provider=provider, session=Session.create(in_memory=True)
            )

            async def run():
                coordinator.submit(f"/resume {path}")
                coordinator.request_exit()
                await coordinator.consume()

            asyncio.run(run())
            self.assertEqual(provider.contexts, [])
            self.assertIsNone(coordinator.session.transcript.state.active_prompt_event_id)
            failures = [
                e for e in coordinator.session.transcript.events
                if isinstance(e, QueryFailed)
            ]
            self.assertEqual([e.stage for e in failures], ["interrupted"])

    def test_startup_finalization_failure_halts_queued_work(self):
        session = Session.create(in_memory=True)
        session.accept_prompt("unfinished")
        provider = ScriptedProvider([])
        coordinator = ApplicationCoordinator(Config("key"), provider=provider, session=session)
        events = []
        coordinator.event_router.subscribe(ApplicationEvent, events.append)

        async def run():
            coordinator.submit("must not run")
            coordinator.request_exit()
            await coordinator.consume()

        with patch.object(session, "fail_query", side_effect=TranscriptPersistenceError("disk full")):
            asyncio.run(run())
        self.assertEqual(provider.contexts, [])
        self.assertEqual(session.revision, 2)
        self.assertFalse(coordinator.accepting)
        self.assertEqual(
            [(e.stage, e.message) for e in events if isinstance(e, PersistenceFailed)],
            [("session activation", "disk full")],
        )
        self.assertIsInstance(events[-1], ExitReady)

    def test_destination_finalization_failure_retains_source_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            saved = Session.create(transcript_path=path)
            prompt = saved.accept_prompt("unfinished")
            saved.record_model_output(
                prompt.event_id, ProviderOutput((ToolCall("missing", "echo", "{}"),))
            )
            saved.close()
            source = Session.create(in_memory=True)
            provider = ScriptedProvider([_text("done")])
            coordinator = ApplicationCoordinator(Config("key"), provider=provider, session=source)

            async def run():
                coordinator.submit(f"/resume {path}")
                coordinator.submit("still here")
                coordinator.request_exit()
                await coordinator.consume()

            with patch.object(Session, "fail_query", side_effect=TranscriptPersistenceError("disk full")):
                asyncio.run(run())
            self.assertIs(coordinator.session, source)
            self.assertEqual(len(provider.contexts), 1)
            self.assertEqual(source.transcript.events[2].status, "error")
            self.assertIn("disk full", source.transcript.events[2].messages[0].content)
            resumed = Session.resume(path)
            self.addCleanup(resumed.close)
            self.assertEqual(resumed.transcript.events[-1].status, "interrupted")
            resumed.interrupt_open_query()
            self.assertIsNone(resumed.transcript.state.active_prompt_event_id)

    def test_failed_resume_keeps_source_for_next_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Session.create(in_memory=True)
            provider = ScriptedProvider([_text("done")])
            coordinator = ApplicationCoordinator(
                Config("key"), provider=provider, session=source
            )

            async def run():
                coordinator.submit(f"/resume {Path(tmp) / 'missing.jsonl'}")
                coordinator.submit("still here")
                coordinator.request_exit()
                await coordinator.consume()

            asyncio.run(run())
            self.assertIs(coordinator.session, source)
            self.assertEqual(len(provider.contexts), 1)
            self.assertEqual(source.transcript.events[2].status, "error")


class ToolResultReaderTests(unittest.TestCase):
    def setUp(self):
        self.session = Session.create(in_memory=True)
        self.addCleanup(self.session.close)
        prompt = self.session.accept_prompt("read")
        output = self.session.record_model_output(
            prompt.event_id, ProviderOutput((ToolCall("call", "echo", "{}"),))
        )
        self.result = self.session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="call",
            status="success",
            content="abcdef",
        )
        self.reader = ToolResultReader(self.session.transcript)

    def test_tool_pages_and_end_marker_match_direct_reader(self):
        tool = self.reader.as_tool()
        page = tool.handler(
            {"result_event_id": self.result.event_id, "offset": 4, "limit": 2}
        )
        self.assertEqual(
            page, self.reader.read(self.result.event_id, offset=4, limit=2)
        )
        self.assertEqual(
            page,
            f"ef\n[alpha-forge transcript-result]\nresult_event_id: {self.result.event_id}\nnext_offset: 6\neof: true",
        )
        self.assertTrue(
            self.reader.read(self.result.event_id, offset=6).startswith(
                "\n[alpha-forge"
            )
        )

    def test_invalid_paging_and_foreign_results_are_rejected(self):
        for options in (
            {"offset": -1},
            {"offset": True},
            {"limit": 0},
            {"limit": True},
            {"limit": 16001},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.reader.read(self.result.event_id, **options)
        with self.assertRaisesRegex(ValueError, "unknown transcript result"):
            self.reader.read("foreign-result")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            self.reader.as_tool().handler({"result_event_id": ""})
