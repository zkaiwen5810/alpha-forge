"""Freshly implemented calculator; no application-runtime dependencies."""

import ast
import math
import operator

from langchain_core.tools import tool

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _bounded(value: int | float) -> int | float:
    if isinstance(value, complex) or abs(value) > 1e100 or not math.isfinite(value):
        raise ValueError("result is too large or non-finite")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _bounded(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        number = _evaluate(node.operand)
        return _bounded(-number if isinstance(node.op, ast.USub) else number)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent magnitude cannot exceed 100")
        return _bounded(_BINARY[type(node.op)](left, right))
    raise ValueError("only arithmetic with numeric literals is supported")


@tool
def calculator(expression: str) -> str:
    """Evaluate basic arithmetic, for example (2 + 3) * 4."""
    if not expression.strip() or len(expression) > 2000:
        raise ValueError("expression must contain 1 to 2000 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("expression is too complex")
    return str(_evaluate(tree.body))
