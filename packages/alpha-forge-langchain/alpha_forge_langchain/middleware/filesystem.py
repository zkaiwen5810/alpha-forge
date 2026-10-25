"""Configure host filesystem tools and normalize their paths."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import override

from deepagents.backends import LocalShellBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import Command


class HostPathMiddleware(AgentMiddleware):
    """Resolve host paths before the filesystem tools' virtual-path validation."""

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace

    def _normalize_call(self, call: ToolCall) -> ToolCall:
        name = call["name"]
        field = (
            "file_path"
            if name in {"read_file", "write_file", "edit_file"}
            else "path"
            if name in {"ls", "glob", "grep"}
            else None
        )
        arguments = call["args"]
        if field is not None and isinstance(arguments.get(field), str):
            path = Path(arguments[field]).expanduser()
            if not path.is_absolute():
                path = self.workspace / path
            return {
                **call,
                "args": {**arguments, field: str(path.resolve())},
            }
        return call

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        call = self._normalize_call(request.tool_call)
        if call is not request.tool_call:
            # FilesystemMiddleware (Deep Agents 0.7.25) unconditionally rejects later
            # write/edit/delete calls to the same path within one assistant batch;
            # there is no public setting for this check. Give the next middleware's
            # parallel mutation check canonical paths without changing persisted
            # assistant arguments. It does not cover shell commands or separate runs.
            messages = list(request.state.get("messages", []))
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if isinstance(message, AIMessage):
                    calls = []
                    for previous in message.tool_calls:
                        try:
                            calls.append(self._normalize_call(previous))
                        except OSError, RuntimeError, ValueError:
                            calls.append(previous)  # Invalid calls fail individually.
                    messages[index] = message.model_copy(update={"tool_calls": calls})
                    break
            request = request.override(
                tool_call=call, state={**request.state, "messages": messages}
            )
        return await handler(request)


def filesystem_middleware(workspace: Path) -> FilesystemMiddleware:
    """Reuse file/search/shell implementations without virtual storage or offloading."""
    return FilesystemMiddleware(
        backend=LocalShellBackend(
            root_dir=workspace, virtual_mode=False, inherit_env=True
        ),
        tools=["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"],
        system_prompt=(
            f"The host workspace is {workspace}. File tools accept normal host paths. "
            "Search defaults to this workspace. Shell commands execute directly on the "
            "host, without isolation or approval, and start in the workspace on each call. "
            "Shell environment and directory changes do not persist between calls. "
            "Use separate tool-call steps for dependent operations."
        ),
        custom_tool_descriptions={
            "execute": (
                "Run a shell command directly on the host in the workspace. Returns "
                "combined stdout/stderr and exit code. Each call starts a new shell; "
                "cd and export changes do not persist. Uses a 120-second default timeout "
                "and truncates output beyond 100,000 characters. No isolation or approvals."
            ),
        },
        tool_token_limit_before_evict=None,
        human_message_token_limit_before_evict=None,
        offload_binary_content=False,
    )
