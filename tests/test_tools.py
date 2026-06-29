import unittest

from alpha_forge.tools import (
    Tool,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    load_builtin_tools,
)


def _echo(arguments):  # type: ignore[no-untyped-def]
    return str(arguments["value"])


class ToolRegistryTests(unittest.TestCase):
    def test_registers_gets_aliases_and_builds_openai_definition(self) -> None:
        validator_calls: list[object] = []
        tool = Tool(
            name="echo",
            aliases=("repeat",),
            is_mcp=True,
            description="Human-facing description.",
            prompt="Model-facing instructions.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            validate_input=validator_calls.append,  # type: ignore[arg-type]
            function=_echo,
        )
        registry = ToolRegistry([tool])

        self.assertIs(registry.get("echo"), tool)
        self.assertIs(registry.get("repeat"), tool)
        self.assertEqual(registry.execute("repeat", {"value": "hello"}), "hello")
        self.assertEqual(validator_calls, [])
        self.assertTrue(tool.is_mcp)
        self.assertEqual(
            registry.definitions(),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Model-facing instructions.",
                        "parameters": tool.input_schema,
                    },
                }
            ],
        )

    def test_rejects_collisions_and_unknown_tools(self) -> None:
        first = Tool(
            name="first",
            aliases=("shared",),
            description="First",
            prompt="First",
            input_schema={"type": "object"},
            function=lambda _arguments: "first",
        )
        registry = ToolRegistry([first])
        second = Tool(
            name="shared",
            description="Second",
            prompt="Second",
            input_schema={"type": "object"},
            function=lambda _arguments: "second",
        )

        with self.assertRaises(ValueError):
            registry.register(second)
        with self.assertRaises(ToolNotFoundError):
            registry.get("missing")


class CalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_builtin_tools()

    def test_builtin_loader_exposes_calculator_and_alias(self) -> None:
        self.assertEqual(self.registry.get("calculator").name, "calculator")
        self.assertEqual(self.registry.get("calc").name, "calculator")

    def test_calculates_precedence_parentheses_unary_and_decimals(self) -> None:
        cases = {
            "2 + 3 * 4": "14",
            "(2 + 3) * 4": "20",
            "-5 + +2": "-3",
            "7 / 2": "3.5",
            "2 ** 8": "256",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(
                    self.registry.execute(
                        "calculator",
                        {"expression": expression},
                    ),
                    expected,
                )

    def test_rejects_unsafe_invalid_and_unbounded_expressions(self) -> None:
        expressions = [
            "__import__('os').getcwd()",
            "1 / 0",
            "2 ** 101",
            "1e101",
            "2 +",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ToolExecutionError):
                    self.registry.execute(
                        "calculator",
                        {"expression": expression},
                    )

        with self.assertRaises(ToolExecutionError):
            self.registry.execute("calculator", {"expression": ""})
