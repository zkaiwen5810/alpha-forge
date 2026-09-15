"""Coordinate one ephemeral tool approval at a time."""

import asyncio
from uuid import uuid4

from alpha_forge.application.events import (
    ToolPermissionRequested,
    ToolPermissionResolved,
)
from alpha_forge.application.router import ApplicationEventRouter
from alpha_forge.hooks import PreToolExecution


class PermissionBroker:
    def __init__(self, event_router: ApplicationEventRouter) -> None:
        self.event_router = event_router
        self._pending_permission: tuple[str, asyncio.Future[bool]] | None = None

    def deny_pending(self) -> None:
        if self._pending_permission is not None:
            request_id, _future = self._pending_permission
            self.resolve(request_id, False)

    async def request(self, event: PreToolExecution) -> bool:
        """Publish one ephemeral approval request and await its resolution."""

        if self._pending_permission is not None:
            raise RuntimeError("another tool permission request is already pending")
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending_permission = (request_id, future)
        self.event_router.publish(
            ToolPermissionRequested(
                request_id=request_id,
                call_id=event.call_id,
                tool_name=event.tool_name,
                tool_input=event.tool_input,
            )
        )
        try:
            return await future
        finally:
            if self._pending_permission is not None:
                pending_id, _pending_future = self._pending_permission
                if pending_id == request_id:
                    self._pending_permission = None

    def resolve(self, request_id: str, allowed: bool) -> bool:
        """Resolve the current approval request exactly once."""

        if self._pending_permission is None:
            return False
        pending_id, future = self._pending_permission
        if pending_id != request_id or future.done():
            return False
        future.set_result(allowed)
        self.event_router.publish(ToolPermissionResolved(request_id, allowed))
        return True
