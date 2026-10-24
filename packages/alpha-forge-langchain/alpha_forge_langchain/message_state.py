"""Reconcile native streamed message chunks and completed node updates."""

from collections.abc import Sequence
from copy import deepcopy

from langchain.agents import AgentState
from langchain.agents.middleware.types import OutputAgentState
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    message_chunk_to_message,
)
from langgraph.types import StreamPart


class MessageState:
    """Reconcile message chunks and completed node updates by message ID."""

    def __init__(self, messages: Sequence[BaseMessage] = ()) -> None:
        self.messages: list[BaseMessage] = deepcopy(list(messages))
        self._indices = {m.id: i for i, m in enumerate(self.messages) if m.id}
        self._chunks: dict[str, BaseMessageChunk] = {}

    def add_message(self, message: BaseMessage) -> None:
        message = deepcopy(message)
        index = self._indices.get(message.id) if message.id else None
        if index is None:
            if message.id:
                self._indices[message.id] = len(self.messages)
            self.messages.append(message)
        else:
            self.messages[index] = message

    def update(
        self, part: StreamPart[AgentState[object], OutputAgentState[object]]
    ) -> None:
        if part["type"] == "messages":
            chunk, _metadata = part["data"]
            if isinstance(chunk, BaseMessageChunk):
                message_id = chunk.id
                if message_id:
                    previous = self._chunks.get(message_id)
                    chunk = (
                        previous + chunk if previous is not None else deepcopy(chunk)
                    )
                    self._chunks[message_id] = chunk
                self.add_message(message_chunk_to_message(chunk))
            else:
                self.add_message(chunk)
        elif part["type"] == "updates":
            for update in part["data"].values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", ()):
                    if isinstance(message, BaseMessage):
                        self.add_message(message)
                        if message.id:
                            self._chunks.pop(message.id, None)
