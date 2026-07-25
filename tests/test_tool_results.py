import unittest

from alpha_forge.tool_results import (
    ToolResultBudgetError,
    TranscriptToolResultLimiter,
)
from alpha_forge.transcript import ToolResult


def _result(
    call_id: str,
    content: str,
    *,
    failed: bool = False,
) -> ToolResult:
    return ToolResult(
        result_id=f"result-{call_id}",
        output_id="output",
        call_id=call_id,
        content=content,
        failed=failed,
    )


class ToolResultLimiterTests(unittest.TestCase):
    def test_default_limits_match_builtin_policy(self) -> None:
        limiter = TranscriptToolResultLimiter()

        self.assertEqual(limiter.individual_limit, 16_000)
        self.assertEqual(limiter.aggregate_limit, 32_000)

    def test_results_within_budgets_are_unchanged(self) -> None:
        limiter = TranscriptToolResultLimiter(
            individual_limit=500,
            aggregate_limit=800,
        )
        results = (_result("one", "first"), _result("two", "second", failed=True))

        applied = limiter.apply(
            output_id="output",
            results=results,
        )
        rendered = [
            limiter.render(result, decision)
            for result, decision in zip(results, applied.decisions, strict=True)
        ]

        self.assertEqual(rendered, ["first", "second"])
        self.assertTrue(all(decision.reason is None for decision in applied.decisions))
        self.assertTrue(results[1].failed)

    def test_individual_overflow_creates_stable_head_tail_preview(self) -> None:
        content = "HEAD" + ("x" * 1_000) + "TAIL"
        limiter = TranscriptToolResultLimiter(
            individual_limit=500,
            aggregate_limit=800,
        )
        result = _result("one", content)

        applied = limiter.apply(
            output_id="output",
            results=(result,),
        )
        preview = limiter.render(result, applied.decisions[0])

        self.assertEqual(len(preview), 500)
        self.assertIn("reason: individual_limit", preview)
        self.assertIn('transcript_ref: "result-one"', preview)
        self.assertIn("HEAD", preview)
        self.assertIn("TAIL", preview)

    def test_aggregate_budget_water_fills_and_preserves_short_result(self) -> None:
        limiter = TranscriptToolResultLimiter(
            individual_limit=1_000,
            aggregate_limit=1_100,
        )
        results = (
            _result("short", "s" * 100),
            _result("a", "a" * 900),
            _result("b", "b" * 900),
        )

        applied = limiter.apply(
            output_id="output",
            results=results,
        )
        rendered = [
            limiter.render(result, decision)
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
        limiter = TranscriptToolResultLimiter(
            individual_limit=700,
            aggregate_limit=1_000,
        )
        results = (_result("a", "a" * 900), _result("b", "b" * 900))

        applied = limiter.apply(
            output_id="output",
            results=results,
        )

        self.assertEqual(
            [decision.reason for decision in applied.decisions],
            [
                "individual_and_aggregate_limits",
                "individual_and_aggregate_limits",
            ],
        )

    def test_too_small_preview_budget_fails_before_event_is_returned(self) -> None:
        limiter = TranscriptToolResultLimiter(
            individual_limit=10,
            aggregate_limit=10,
        )

        with self.assertRaises(ToolResultBudgetError):
            limiter.apply(
                output_id="output",
                results=(_result("one", "x" * 20),),
            )

    def test_rejects_nonpositive_limits(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptToolResultLimiter(individual_limit=0)
        with self.assertRaises(ValueError):
            TranscriptToolResultLimiter(aggregate_limit=0)


if __name__ == "__main__":
    unittest.main()
