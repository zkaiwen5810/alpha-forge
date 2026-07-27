import unittest

from alpha_forge.context import (
    ContextPipeline,
    ModelContextSnapshot,
    ToolResultBudgetError,
    ToolResultBudgetPolicy,
    ToolResultContext,
)
from alpha_forge.context.policies import ContextPolicyDecision
from alpha_forge.providers import ProviderOutput, ToolCall
from alpha_forge.sessions import Session
from alpha_forge.transcript import ContextEdited, HeadTailPreview, PolicyInvocation


def _tool_exchange(contents: tuple[str, ...]) -> Session:
    session = Session.create(in_memory=True)
    prompt = session.accept_prompt("run tools")
    calls = tuple(
        ToolCall(f"call-{index}", "echo", "{}")
        for index in range(len(contents))
    )
    output = session.record_model_output(
        prompt.event_id,
        ProviderOutput(calls, "tool_calls"),
    )
    for call, content in zip(calls, contents, strict=True):
        session.record_tool_result(
            model_output_event_id=output.event_id,
            call_id=call.call_id,
            status="success",
            content=content,
        )
    return session


class _ObservingPolicy:
    def __init__(self) -> None:
        self.seen_revision = None
        self.seen_content = None

    def evaluate(
        self,
        snapshot: ModelContextSnapshot,
    ) -> ContextPolicyDecision:
        self.seen_revision = snapshot.revision
        self.seen_content = next(
            item.content
            for item in snapshot.items
            if isinstance(item, ToolResultContext)
        )
        return ContextPolicyDecision(PolicyInvocation("observer", 1, {}))


class ContextPolicyTests(unittest.TestCase):
    def test_results_within_budget_emit_no_context_event(self) -> None:
        session = _tool_exchange(("short",))
        revision = session.revision

        snapshot = session.prepare_context(
            ContextPipeline((ToolResultBudgetPolicy(),))
        )

        self.assertEqual(session.revision, revision)
        self.assertEqual(snapshot.revision, revision)
        self.assertFalse(
            any(isinstance(event, ContextEdited) for event in session.transcript.events)
        )

    def test_oversized_result_records_metadata_without_copying_content(self) -> None:
        session = _tool_exchange(("a" * 1000,))
        snapshot = session.prepare_context(
            ContextPipeline(
                (
                    ToolResultBudgetPolicy(
                        individual_limit=300,
                        aggregate_limit=400,
                    ),
                )
            )
        )

        edit = session.transcript.events[-1]
        self.assertIsInstance(edit, ContextEdited)
        operation = edit.operations[0]
        self.assertIsInstance(operation.representation, HeadTailPreview)
        self.assertEqual(operation.representation.rendered_chars, 300)
        self.assertNotIn("a" * 100, repr(edit))
        projected = next(
            item for item in snapshot.items if isinstance(item, ToolResultContext)
        )
        self.assertEqual(len(projected.content), 300)
        self.assertEqual(projected.original_chars, 1000)
        self.assertFalse(hasattr(projected, "raw_content"))

    def test_policies_chain_serially_with_committed_reprojection(self) -> None:
        session = _tool_exchange(("x" * 1000,))
        observer = _ObservingPolicy()

        session.prepare_context(
            ContextPipeline(
                (
                    ToolResultBudgetPolicy(
                        individual_limit=300,
                        aggregate_limit=300,
                    ),
                    observer,
                )
            )
        )

        self.assertEqual(observer.seen_revision, session.revision)
        self.assertEqual(len(observer.seen_content), 300)
        self.assertEqual(
            sum(
                isinstance(event, ContextEdited)
                for event in session.transcript.events
            ),
            1,
        )

    def test_aggregate_budget_water_fills_results(self) -> None:
        session = _tool_exchange(("a" * 100, "b" * 1000))
        snapshot = session.prepare_context(
            ContextPipeline(
                (
                    ToolResultBudgetPolicy(
                        individual_limit=1000,
                        aggregate_limit=500,
                    ),
                )
            )
        )

        results = [
            item for item in snapshot.items if isinstance(item, ToolResultContext)
        ]
        self.assertEqual([len(result.content) for result in results], [100, 400])

    def test_budget_too_small_for_metadata_does_not_append(self) -> None:
        session = _tool_exchange(("x" * 1000,))
        revision = session.revision
        with self.assertRaises(ToolResultBudgetError):
            session.prepare_context(
                ContextPipeline(
                    (
                        ToolResultBudgetPolicy(
                            individual_limit=1,
                            aggregate_limit=1,
                        ),
                    )
                )
            )
        self.assertEqual(session.revision, revision)


if __name__ == "__main__":
    unittest.main()
