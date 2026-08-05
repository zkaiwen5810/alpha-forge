import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.config import Config
from alpha_forge.context import UserMessage
from alpha_forge.providers import (
    OutputMessage,
    OutputText,
    ProviderOutput,
    StreamCompleted,
)
from alpha_forge.sessions import Session
from alpha_forge.transcript import (
    CommandCompleted,
    InputAccepted,
    SessionLinked,
)


class EchoProvider:
    def __init__(self) -> None:
        self.contexts = []

    def list_models(self) -> list[str]:
        return ["gpt-test"]

    async def stream(self, context, *, tools):
        self.contexts.append(context)
        prompt = [
            item.content for item in context.items if isinstance(item, UserMessage)
        ][-1]
        yield StreamCompleted(
            ProviderOutput(
                (OutputMessage((OutputText(f"reply:{prompt}"),)),),
                "stop",
            )
        )


class SessionAndCoordinatorTests(unittest.TestCase):
    def test_clear_destination_is_not_persisted_until_it_accepts_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"XDG_DATA_HOME": tmp}):
                source = Session.create()
                coordinator = ApplicationCoordinator(
                    Config("key"),
                    provider=EchoProvider(),
                    session=source,
                )

                async def run():
                    coordinator.submit("/clear")
                    coordinator.request_exit()
                    await coordinator.consume()

                asyncio.run(run())

                self.assertTrue(source.transcript_path.exists())
                self.assertFalse(coordinator.session.transcript_path.exists())

    def test_fifo_accepts_next_prompt_only_after_prior_query_completes(self) -> None:
        provider = EchoProvider()
        session = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=session,
        )

        async def run():
            coordinator.submit("first")
            coordinator.submit("second")
            coordinator.request_exit()
            await coordinator.consume()

        asyncio.run(run())

        self.assertEqual(len(provider.contexts), 2)
        prompts = [
            [
                item.content
                for item in context.items
                if isinstance(item, UserMessage)
            ]
            for context in provider.contexts
        ]
        self.assertEqual(prompts, [["first"], ["first", "second"]])
        accepted = [
            event.text
            for event in session.transcript.events
            if isinstance(event, InputAccepted)
        ]
        self.assertEqual(accepted, ["first", "second"])

    def test_clear_switches_session_before_later_queued_prompt(self) -> None:
        provider = EchoProvider()
        source = Session.create(in_memory=True)
        coordinator = ApplicationCoordinator(
            Config("key"),
            provider=provider,
            session=source,
        )

        async def run():
            coordinator.submit("/clear")
            coordinator.submit("after clear")
            coordinator.request_exit()
            await coordinator.consume()

        asyncio.run(run())

        self.assertIsNot(coordinator.session, source)
        self.assertTrue(
            any(isinstance(event, CommandCompleted) for event in source.transcript.events)
        )
        self.assertTrue(
            any(
                isinstance(event, SessionLinked) and event.kind == "clear"
                for event in coordinator.session.transcript.events
            )
        )
        destination_prompts = [
            event.text
            for event in coordinator.session.transcript.events
            if isinstance(event, InputAccepted) and event.kind == "prompt"
        ]
        self.assertEqual(destination_prompts, ["after clear"])

    def test_resume_validates_existing_schema_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "saved.jsonl"
            original = Session.create(transcript_path=path)
            prompt = original.accept_prompt("saved")
            original.record_model_output(
                prompt.event_id,
                ProviderOutput(
                    (OutputMessage((OutputText("answer"),)),),
                    "stop",
                ),
            )
            original.close()

            resumed = Session.resume(path)
            self.assertEqual(resumed.revision, 3)
            self.assertIsNone(resumed.open_query())
            resumed.close()

    def test_session_tool_result_reader_pages_raw_content(self) -> None:
        session = Session.create(in_memory=True)
        prompt = session.accept_prompt("tool")
        from alpha_forge.providers import ToolCall

        output = session.record_model_output(
            prompt.event_id,
            ProviderOutput((ToolCall("call", "echo", "{}"),)),
        )
        result = session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id="call",
            status="success",
            content="abcdefghij",
        )

        page = session.read_tool_result(result.event_id, offset=2, limit=4)

        self.assertTrue(page.startswith("cdef"))
        self.assertIn("next_offset: 6", page)
        self.assertIn("eof: false", page)


if __name__ == "__main__":
    unittest.main()
