"""Status messages derived from application progress."""

from alpha_forge.application.events import (
    ApplicationEvent,
    ExitRequested,
    InputQueued,
    InputStarted,
    ModelOutputRecorded,
    PersistenceFailed,
    RequestFailed,
    ResponseStreamCompleted,
    ResponseStreamStarted,
    ResponseStreamUpdated,
    SessionViewChanged,
    StatusChanged,
    ToolCallProcessingStarted,
    ToolPermissionRequested,
    ToolPermissionResolved,
    ToolResultRecorded,
)


class StatusState:
    def __init__(self) -> None:
        self.message = "Ready"
        self.exiting = False
        self.persistence_error: str | None = None

    def handle(self, event: ApplicationEvent, *, queued_count: int) -> bool:
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
        elif isinstance(event, (ResponseStreamStarted, ResponseStreamUpdated)):
            self.message = "Streaming response"
        elif isinstance(event, ResponseStreamCompleted):
            self.message = "Saving response"
        elif isinstance(event, ToolCallProcessingStarted):
            self.message = f"Running tool: {event.call.name}"
        elif isinstance(event, ToolPermissionRequested):
            self.message = f"Approval required: {event.tool_name}"
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
