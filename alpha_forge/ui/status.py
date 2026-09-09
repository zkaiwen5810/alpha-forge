"""Status messages derived from application progress."""

from alpha_forge.application.events import (
    ExitRequested,
    InputQueued,
    InputStarted,
    ModelOutputRecorded,
    PersistenceFailed,
    ProviderDeltaReceived,
    ProviderRequestStarted,
    ProviderResponseCompleted,
    RequestFailed,
    SessionViewChanged,
    StatusChanged,
    ToolPermissionRequested,
    ToolPermissionResolved,
    ToolResultRecorded,
    ToolStarted,
)
from alpha_forge.events import Event


class StatusState:
    def __init__(self) -> None:
        self.message = "Ready"
        self.exiting = False
        self.persistence_error: str | None = None

    def handle(self, event: Event, *, queued_count: int) -> bool:
        """Application state reducer called by its owner; no toolkit hook involved."""
        if isinstance(
            event,
            (
                SessionViewChanged,
                InputQueued,
                InputStarted,
                ModelOutputRecorded,
                ToolResultRecorded,
            ),
        ):
            self.message = self._queue_status(queued_count)
        elif isinstance(event, (ProviderRequestStarted, ProviderDeltaReceived)):
            self.message = "Streaming response"
        elif isinstance(event, ProviderResponseCompleted):
            self.message = "Saving response"
        elif isinstance(event, ToolStarted):
            self.message = f"Running tool: {event.call.name}"
        elif isinstance(event, ToolPermissionRequested):
            self.message = f"Approval required: {event.event.tool_name}"
        elif isinstance(event, ToolPermissionResolved):
            self.message = "Running approved tool" if event.allowed else "Denying tool"
        elif isinstance(event, PersistenceFailed):
            self.persistence_error = event.message
            self.message = f"Cannot persist {event.stage}: {event.message}"
        elif isinstance(event, RequestFailed):
            self.message = f"Request failed: {event.message}"
        elif isinstance(event, StatusChanged):
            self.message = event.message
        elif isinstance(event, ExitRequested):
            self.exiting = True
            self.message = self._queue_status(queued_count)
        else:
            return False
        return True

    def _queue_status(self, queued_count: int) -> str:
        if self.persistence_error is not None:
            return "Persistence failed; input processing stopped"
        if self.exiting:
            return "Exiting after queued inputs"
        if not queued_count:
            return "Ready"
        if queued_count == 1:
            return "1 input queued"
        return f"{queued_count} inputs queued"
