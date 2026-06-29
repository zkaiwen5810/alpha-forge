"""Loading for tools shipped with Alpha Forge."""

from alpha_forge.tools.calculator import CALCULATOR_TOOL
from alpha_forge.tools.registry import ToolRegistry


def load_builtin_tools() -> ToolRegistry:
    """Build a fresh registry containing all packaged tools."""
    return ToolRegistry([CALCULATOR_TOOL])
