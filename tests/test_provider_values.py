import unittest

from alpha_forge.providers import (
    OutputMessage,
    OutputRefusal,
    OutputText,
    ProviderOutput,
    ProviderOutputAccumulator,
    ReasoningDelta,
    ReasoningItem,
    RefusalDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
)


class ProviderOutputTests(unittest.TestCase):
    def test_provider_oriented_items_preserve_atomic_tool_batch(self) -> None:
        output = ProviderOutput(
            (
                ReasoningItem("reason"),
                OutputMessage(
                    (OutputText("answer"), OutputRefusal("cannot continue"))
                ),
                ToolCall("one", "reader", "{}"),
                ToolCall("two", "calculator", '{"expression":"2+2"}'),
            ),
            "tool_calls",
        )

        self.assertEqual(output.output_text, "answer")
        self.assertEqual(output.reasoning, "reason")
        self.assertEqual(output.refusal, "cannot continue")
        self.assertEqual(
            [call.call_id for call in output.tool_calls],
            ["one", "two"],
        )

    def test_stream_accumulator_is_only_a_precommit_draft(self) -> None:
        accumulator = ProviderOutputAccumulator()
        for event in (
            ReasoningDelta("rea"),
            ReasoningDelta("son"),
            TextDelta("answer"),
            RefusalDelta("no"),
            ToolCallDelta(0, "call-", "calc", '{"x":'),
            ToolCallDelta(0, "1", "ulator", "1}"),
        ):
            accumulator.apply(event)

        self.assertEqual(
            accumulator.build("tool_calls"),
            ProviderOutput(
                (
                    ReasoningItem("reason"),
                    OutputMessage((OutputText("answer"), OutputRefusal("no"))),
                    ToolCall("call-1", "calculator", '{"x":1}'),
                ),
                "tool_calls",
            ),
        )


if __name__ == "__main__":
    unittest.main()
