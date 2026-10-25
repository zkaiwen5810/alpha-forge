"""Derive bounded model context while preserving the original transcript."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import cast, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


class HistoryCompactionMiddleware(AgentMiddleware):
    """Bound model input by dropping old turns without changing the transcript."""

    def __init__(self, max_context_tokens: int) -> None:
        super().__init__()
        if max_context_tokens <= 0:
            raise ValueError("MAX_CONTEXT_TOKENS must be positive")
        self.max_context_tokens = max_context_tokens

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        overhead = count_tokens_approximately(
            [request.system_message] if request.system_message is not None else [],
            tools=request.tools,
        )
        message_budget = self.max_context_tokens - overhead
        messages = request.messages
        if count_tokens_approximately(messages) <= message_budget:
            return await handler(request)

        current_start = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        required = messages[current_start:] if current_start is not None else messages
        if current_start and isinstance(messages[0], SystemMessage):
            required = [messages[0], *required]
        required_tokens = overhead + count_tokens_approximately(required)
        if current_start is None or required_tokens > self.max_context_tokens:
            raise ValueError(
                "Cannot compact model input: the current turn, system instructions, "
                f"and tool definitions require {required_tokens} estimated tokens; "
                f"MAX_CONTEXT_TOKENS is {self.max_context_tokens}."
            )

        compacted = trim_messages(
            deepcopy(messages),
            max_tokens=message_budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
        # Trimming retains the concrete message types from ModelRequest.
        return await handler(
            request.override(messages=cast(list[AnyMessage], compacted))
        )
