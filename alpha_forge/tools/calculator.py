"""Safe built-in arithmetic calculator."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

from alpha_forge.tools.base import Tool, ToolExecutionError

Number = int | float

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_NODES = 100
_MAX_EXPONENT = 100
_MAX_ABSOLUTE_RESULT = 1e100


def _checked(number: Number) -> Number:
    try:
        finite = math.isfinite(number)
        within_limit = abs(number) <= _MAX_ABSOLUTE_RESULT
    except OverflowError as exc:
        raise ToolExecutionError("result is too large or non-finite") from exc
    if not finite or not within_limit:
        raise ToolExecutionError("result is too large or non-finite")
    return number


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolExecutionError("expression may contain only numeric literals")
        return _checked(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolExecutionError(
                f"exponent magnitude cannot exceed {_MAX_EXPONENT}"
            )
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except (ArithmeticError, OverflowError) as exc:
            raise ToolExecutionError(str(exc)) from exc
        if isinstance(result, complex):
            raise ToolExecutionError("complex results are not supported")
        return _checked(result)
    raise ToolExecutionError(f"unsupported expression element: {type(node).__name__}")


def calculate(arguments: Mapping[str, Any]) -> str:
    """Evaluate the calculator's ``expression`` argument."""
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolExecutionError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError("invalid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolExecutionError("expression is too complex")
    result = _evaluate(tree)
    return str(result)


CALCULATOR_TOOL = Tool(
    name="calculator",
    aliases=("calc",),
    display_description="Safely evaluates a basic arithmetic expression.",
    description=(
        "Evaluate a basic arithmetic expression. Use this tool when exact "
        "arithmetic is needed instead of calculating mentally."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, for example: (2 + 3) * 4",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=calculate,
)
