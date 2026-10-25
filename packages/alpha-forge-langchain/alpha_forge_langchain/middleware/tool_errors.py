"""Normalize tool failures into model-visible error results."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.model_call_limit import ModelCallLimitState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@wrap_tool_call(state_schema=ModelCallLimitState)
async def tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
) -> ToolMessage | Command[object]:
    try:
        return await handler(request)
    # Normalize arbitrary exceptions from caller-supplied tools for the model.
    except Exception as exc:  # noqa: BLE001
        return ToolMessage(
            content=f"error: {str(exc) or type(exc).__name__}",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )
