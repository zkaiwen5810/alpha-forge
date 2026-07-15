"""Compatibility imports for the split REPL controller and UI state modules.

New code should import orchestration from :mod:`alpha_forge.repl_controller`
and rendering state from :mod:`alpha_forge.ui_state`. Keeping this facade makes
the structural refactor behavior-preserving for existing embedders.
"""

from alpha_forge.repl_controller import (
    DEFAULT_SYSTEM_PROMPT,
    MAX_TOOL_ITERATIONS,
    ChatReplController,
    WorkItem,
)
from alpha_forge.ui_state import (
    ChatUiState,
    ConversationTurnBlock,
    HistoryBlock,
    HistoryLine,
    HistoryRole,
    IterationOutput,
    StandaloneBlock,
    TokenUsage,
    ToolExchange,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "MAX_TOOL_ITERATIONS",
    "ChatReplController",
    "ChatUiState",
    "ConversationTurnBlock",
    "HistoryBlock",
    "HistoryLine",
    "HistoryRole",
    "IterationOutput",
    "StandaloneBlock",
    "TokenUsage",
    "ToolExchange",
    "WorkItem",
]
