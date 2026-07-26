import asyncio
import unittest

from alpha_forge.models import ToolCall
from alpha_forge.tool_execution import ToolExecutor
from alpha_forge.tools import Tool, ToolRegistry


class ToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    function=lambda arguments: str(arguments["value"]),
                    description="echo",
                    prompt="echo",
                    input_schema={"type": "object"},
                )
            ]
        )

    def test_executes_valid_calls(self) -> None:
        result = asyncio.run(
            ToolExecutor(self.registry).execute(
                ToolCall("call", "echo", '{"value":"ok"}')
            )
        )
        self.assertEqual(result.content, "ok")
        self.assertFalse(result.failed)

    def test_normalizes_bad_json_and_unknown_tools(self) -> None:
        executor = ToolExecutor(self.registry)
        invalid = asyncio.run(
            executor.execute(ToolCall("one", "echo", "not-json"))
        )
        unknown = asyncio.run(
            executor.execute(ToolCall("two", "missing", "{}"))
        )
        self.assertTrue(invalid.failed)
        self.assertTrue(unknown.failed)
        self.assertIn("error:", invalid.content)
        self.assertIn("unknown tool", unknown.content)


if __name__ == "__main__":
    unittest.main()
