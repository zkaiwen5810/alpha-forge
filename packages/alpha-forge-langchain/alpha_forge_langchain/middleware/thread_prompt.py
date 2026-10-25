"""Apply persisted thread prompt overrides to model requests."""

from collections.abc import Awaitable, Callable
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage


class ThreadPromptState(AgentState):
    system_prompt_override: NotRequired[str | None]


class ThreadPromptMiddleware(AgentMiddleware):
    """Apply an optional persisted thread prompt to each native model request."""

    state_schema = ThreadPromptState

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        prompt = request.state.get("system_prompt_override")
        if prompt is not None:
            if not isinstance(prompt, str):
                raise ValueError("system_prompt_override must be a string or None")
            request = request.override(system_message=SystemMessage(content=prompt))
        return await handler(request)
