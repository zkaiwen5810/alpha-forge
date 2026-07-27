import unittest

from alpha_forge.model_messages import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)
from alpha_forge.models import (
    PromptEdit,
    PromptEditDecision,
    RawToolResult,
    ToolCall,
)
from alpha_forge.prompt_editor import (
    INTERRUPTED_TOOL_RESULT_CONTENT,
    PromptDraft,
    PromptEditBudgetError,
    ToolResultPromptEditor,
)


def _result(
    call_id: str,
    content: str,
    *,
    failed: bool = False,
) -> RawToolResult:
    return RawToolResult(
        result_id=f"result-{call_id}",
        call_id=call_id,
        content=content,
        failed=failed,
    )


class ToolResultPromptEditorTests(unittest.TestCase):
    def test_default_limits_match_builtin_policy(self) -> None:
        editor = ToolResultPromptEditor()

        self.assertEqual(editor.individual_limit, 16_000)
        self.assertEqual(editor.aggregate_limit, 32_000)

    def test_results_within_budgets_are_unchanged(self) -> None:
        editor = ToolResultPromptEditor(
            individual_limit=500,
            aggregate_limit=800,
        )
        results = (_result("one", "first"), _result("two", "second", failed=True))

        applied = editor.edit_results(results)
        rendered = [
            editor.render(result, decision)
            for result, decision in zip(results, applied.decisions, strict=True)
        ]

        self.assertEqual(rendered, ["first", "second"])
        self.assertTrue(all(decision.reason is None for decision in applied.decisions))
        self.assertTrue(results[1].failed)

    def test_prompt_strategy_is_a_noop_without_raw_tool_messages(self) -> None:
        editor = ToolResultPromptEditor()
        messages = (UserMessage("hello"),)

        edited = editor.edit(PromptDraft(messages))

        self.assertEqual(edited.messages, messages)
        self.assertIsNone(edited.tool_batch_edit)

    def test_prompt_strategy_edits_raw_message_tail_in_call_order(self) -> None:
        editor = ToolResultPromptEditor()
        calls = (
            ToolCall("one", "tool", "{}"),
            ToolCall("two", "tool", "{}"),
        )
        draft = PromptDraft(
            (
                UserMessage("run"),
                AssistantMessage(None, calls, output_id="output"),
                ToolMessage(
                    "second",
                    "two",
                    result_id="result-two",
                    raw=True,
                ),
                ToolMessage(
                    "first",
                    "one",
                    result_id="result-one",
                    raw=True,
                ),
            ),
        )

        edited = editor.edit(draft)

        self.assertIsNotNone(edited.tool_batch_edit)
        assert edited.tool_batch_edit is not None
        self.assertEqual(
            [message.to_openai()["role"] for message in edited.messages],
            ["user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [result.call_id for result in edited.tool_batch_edit.results],
            ["one", "two"],
        )
        self.assertTrue(
            all(
                not message.raw
                for message in edited.messages
                if isinstance(message, ToolMessage)
            )
        )

    def test_missing_result_policy_synthesizes_only_absent_calls(self) -> None:
        editor = ToolResultPromptEditor()
        calls = (
            ToolCall("one", "tool", "{}"),
            ToolCall("two", "tool", "{}"),
        )
        draft = PromptDraft(
            (
                UserMessage("run"),
                AssistantMessage(None, calls, output_id="output"),
                ToolMessage(
                    "first",
                    "one",
                    result_id="result-one",
                    raw=True,
                ),
            )
        )

        edited = editor.edit(draft)

        assert edited.tool_batch_edit is not None
        effect = edited.tool_batch_edit
        self.assertEqual(
            [result.call_id for result in effect.existing_results],
            ["one"],
        )
        self.assertEqual(
            [result.call_id for result in effect.synthesized_results],
            ["two"],
        )
        self.assertTrue(effect.synthesized_results[0].failed)
        self.assertEqual(
            effect.synthesized_results[0].content,
            INTERRUPTED_TOOL_RESULT_CONTENT,
        )

    def test_individual_overflow_creates_stable_head_tail_preview(self) -> None:
        content = "HEAD" + ("x" * 1_000) + "TAIL"
        editor = ToolResultPromptEditor(
            individual_limit=500,
            aggregate_limit=800,
        )
        result = _result("one", content)

        applied = editor.edit_results((result,))
        preview = editor.render(result, applied.decisions[0])

        self.assertEqual(len(preview), 500)
        self.assertIn("reason: individual_limit", preview)
        self.assertIn('transcript_ref: "result-one"', preview)
        self.assertIn("HEAD", preview)
        self.assertIn("TAIL", preview)

    def test_aggregate_budget_water_fills_and_preserves_short_result(self) -> None:
        editor = ToolResultPromptEditor(
            individual_limit=1_000,
            aggregate_limit=1_100,
        )
        results = (
            _result("short", "s" * 100),
            _result("a", "a" * 900),
            _result("b", "b" * 900),
        )

        applied = editor.edit_results(results)
        rendered = [
            editor.render(result, decision)
            for result, decision in zip(results, applied.decisions, strict=True)
        ]

        self.assertEqual(
            [decision.allocated_chars for decision in applied.decisions],
            [100, 500, 500],
        )
        self.assertEqual(sum(map(len, rendered)), 1_100)
        self.assertIsNone(applied.decisions[0].reason)
        self.assertEqual(applied.decisions[1].reason, "aggregate_limit")
        self.assertEqual(applied.decisions[2].reason, "aggregate_limit")

    def test_individual_and_aggregate_reason_is_recorded(self) -> None:
        editor = ToolResultPromptEditor(
            individual_limit=700,
            aggregate_limit=1_000,
        )
        results = (_result("a", "a" * 900), _result("b", "b" * 900))

        applied = editor.edit_results(results)

        self.assertEqual(
            [decision.reason for decision in applied.decisions],
            [
                "individual_and_aggregate_limits",
                "individual_and_aggregate_limits",
            ],
        )

    def test_too_small_preview_budget_fails_before_event_is_returned(self) -> None:
        editor = ToolResultPromptEditor(
            individual_limit=10,
            aggregate_limit=10,
        )

        with self.assertRaises(PromptEditBudgetError):
            editor.edit_results((_result("one", "x" * 20),))

    def test_rejects_nonpositive_limits(self) -> None:
        with self.assertRaises(ValueError):
            ToolResultPromptEditor(individual_limit=0)
        with self.assertRaises(ValueError):
            ToolResultPromptEditor(aggregate_limit=0)

    def test_render_rejects_a_decision_for_another_result(self) -> None:
        editor = ToolResultPromptEditor()
        result = _result("one", "content")
        edit = PromptEdit(
            policy_version="head_tail_v1",
            individual_limit=16_000,
            aggregate_limit=32_000,
            decisions=(
                PromptEditDecision(
                    "result-other",
                    "one",
                    len(result.content),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "must match"):
            editor.render_edit((result,), edit)


if __name__ == "__main__":
    unittest.main()
