import asyncio
import unittest

from alpha_forge.providers import ToolCall
from alpha_forge.tools import Tool, ToolExecutor, ToolRegistry


class ToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler_calls = 0
        self.registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    handler=self._echo,
                    description="echo",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                )
            ]
        )

    def _echo(self, arguments):  # type: ignore[no-untyped-def]
        self.handler_calls += 1
        return str(arguments["value"])

    def test_executes_valid_calls(self) -> None:
        result = asyncio.run(
            ToolExecutor(self.registry).execute(
                ToolCall("call", "echo", '{"value":"ok"}')
            )
        )
        self.assertEqual(result.content, "ok")
        self.assertEqual(result.status, "success")

    def test_normalizes_bad_json_and_unknown_tools(self) -> None:
        executor = ToolExecutor(self.registry)
        invalid = asyncio.run(
            executor.execute(ToolCall("one", "echo", "not-json"))
        )
        unknown = asyncio.run(
            executor.execute(ToolCall("two", "missing", "{}"))
        )
        self.assertEqual(invalid.status, "error")
        self.assertEqual(unknown.status, "error")
        self.assertIn("error:", invalid.content)
        self.assertIn("unknown tool", unknown.content)

    def test_rejects_schema_violations_before_invoking_handler(self) -> None:
        executor = ToolExecutor(self.registry)
        calls = (
            ToolCall("missing", "echo", "{}"),
            ToolCall("wrong-type", "echo", '{"value":2}'),
            ToolCall("extra", "echo", '{"value":"ok","extra":true}'),
            ToolCall("not-object", "echo", "[]"),
            ToolCall("constant", "echo", '{"value":NaN}'),
        )

        results = [asyncio.run(executor.execute(call)) for call in calls]

        self.assertTrue(all(result.status == "error" for result in results))
        self.assertIn("does not match schema", results[0].content)
        self.assertIn("does not match schema", results[1].content)
        self.assertIn("does not match schema", results[2].content)
        self.assertIn("JSON object", results[3].content)
        self.assertIn("invalid JSON constant", results[4].content)
        self.assertEqual(self.handler_calls, 0)


if __name__ == "__main__":
    unittest.main()
