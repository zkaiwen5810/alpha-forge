"""Per-session conversation state and tool-result coordination."""

from __future__ import annotations

import re
from uuid import uuid4

from alpha_forge.conversation import Conversation, Message, ToolCall, ToolMessage
from alpha_forge.tool_results import RawToolResult, ToolResultManager

DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")


class Session:
    """Own one conversation and the persistence identity for its tool results."""

    def __init__(
        self,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        tool_result_manager: ToolResultManager | None = None,
        session_id: str | None = None,
    ) -> None:
        resolved_session_id = session_id or uuid4().hex
        if not _SAFE_SESSION_ID.fullmatch(resolved_session_id):
            raise ValueError(
                "session ID must contain only letters, digits, _ or -"
            )

        self.session_id = resolved_session_id
        self._system_prompt = system_prompt
        self._tool_result_manager = (
            tool_result_manager
            if tool_result_manager is not None
            else ToolResultManager()
        )
        self._conversation = Conversation(system_prompt=system_prompt)

    @property
    def messages(self) -> list[Message]:
        """Return a snapshot of the messages in this session."""
        return self._conversation.messages

    def add_user(self, content: str) -> None:
        self._conversation.add_user(content)

    def add_assistant(self, content: str | None) -> None:
        self._conversation.add_assistant(content)

    def record_tool_iteration(
        self,
        assistant_content: str | None,
        *,
        tool_calls: tuple[ToolCall, ...],
        raw_results: tuple[RawToolResult, ...],
    ) -> tuple[ToolMessage, ...]:
        """Persist and commit one complete assistant tool-call iteration.

        Result processing happens before protocol history is mutated so a
        persistence failure cannot leave tool calls without matching outputs.
        """
        tool_messages = self._tool_result_manager.process(
            raw_results,
            session_id=self.session_id,
        )
        self._conversation.add_assistant(
            assistant_content,
            tool_calls=tool_calls,
        )
        for tool_message in tool_messages:
            self._conversation.add_tool(tool_message)
        return tool_messages

    def fresh(self) -> Session:
        """Return an empty session with the same prompt and result policy."""
        return Session(
            system_prompt=self._system_prompt,
            tool_result_manager=self._tool_result_manager,
        )


__all__ = ["DEFAULT_SYSTEM_PROMPT", "Session"]
