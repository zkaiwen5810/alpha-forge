import unittest

from alpha_forge.tools import (
    Tool,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    load_builtin_tools,
)
from alpha_forge.tools.calculator import calculate


def _echo(arguments):  # type: ignore[no-untyped-def]
    return str(arguments["value"])


class ToolRegistryTests(unittest.TestCase):
    def test_registers_gets_aliases_and_exposes_provider_neutral_spec(self) -> None:
        tool = Tool(
            name="echo",
            aliases=("repeat",),
            display_description="Human-facing description.",
            description="Model-facing instructions.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=_echo,
        )
        registry = ToolRegistry([tool])

        self.assertIs(registry.get("echo"), tool)
        self.assertIs(registry.get("repeat"), tool)
        self.assertEqual(registry.specs(), (tool.spec,))
        self.assertEqual(tool.spec.name, "echo")
        self.assertEqual(tool.spec.description, "Model-facing instructions.")

    def test_rejects_collisions_and_unknown_tools(self) -> None:
        first = Tool(
            name="first",
            aliases=("shared",),
            description="First",
            input_schema={"type": "object"},
            handler=lambda _arguments: "first",
        )
        registry = ToolRegistry([first])
        second = Tool(
            name="shared",
            description="Second",
            input_schema={"type": "object"},
            handler=lambda _arguments: "second",
        )

        with self.assertRaises(ValueError):
            registry.register(second)
        with self.assertRaises(ToolNotFoundError):
            registry.get("missing")

    def test_tool_spec_copies_and_freezes_its_json_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        tool = Tool(
            name="echo",
            description="echo",
            input_schema=schema,
            handler=_echo,
        )
        schema["properties"]["value"]["type"] = "number"

        self.assertEqual(
            tool.input_schema["properties"]["value"]["type"],
            "string",
        )
        with self.assertRaises(TypeError):
            tool.input_schema["new"] = {}  # type: ignore[index]

    def test_rejects_invalid_json_schema_at_definition_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid tool input schema"):
            Tool(
                name="invalid",
                description="invalid",
                input_schema={"type": "not-a-json-schema-type"},
                handler=lambda _arguments: "never",
            )


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
                    calculate({"expression": expression}),
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
                    calculate({"expression": expression})

        with self.assertRaises(ToolExecutionError):
            calculate({"expression": ""})
