"""Tool approval dialog and its ephemeral request state."""

import json
from collections.abc import Callable

from prompt_toolkit.layout import Container
from prompt_toolkit.layout.containers import to_container
from prompt_toolkit.layout.controls import UIControl
from prompt_toolkit.utils import Event as WidgetEvent
from prompt_toolkit.widgets import Button, Dialog, Label

from alpha_forge.application.events import (
    ApplicationEvent,
    RequestFailed,
    ToolPermissionRequested,
    ToolPermissionResolved,
    ToolResultRecorded,
)
from alpha_forge.json_values import thaw_json
from alpha_forge.ui.component import UiComponent

MAX_PERMISSION_PREVIEW_CHARS = 2_000


class PermissionPanel:
    def __init__(
        self,
        resolve: Callable[[str, bool], bool],
        *,
        focus: Callable[[], None],
        restore_focus: Callable[[], None],
    ) -> None:
        self.on_change: WidgetEvent[UiComponent] = WidgetEvent(self)
        self._resolve = resolve
        self._focus = focus
        self._restore_focus = restore_focus
        self.pending_request: ToolPermissionRequested | None = None
        # Button handlers are registered callbacks, not PermissionPanel overrides.
        self.deny_button = Button("Deny", handler=lambda: self.resolve(False))
        self.allow_button = Button("Allow once", handler=lambda: self.resolve(True))
        self.dialog = Dialog(
            title="Tool Permission",
            body=Label(text=self.render_request, style="class:permission"),
            buttons=[self.deny_button, self.allow_button],
            modal=True,
        )

    def __pt_container__(self) -> Container:
        """prompt-toolkit widget protocol: return the root container (not an override)."""
        return to_container(self.dialog)

    @property
    def focus_target(self) -> UIControl:
        """Application widget API: the control our parent passes to Layout.focus()."""
        return self.deny_button.control

    def has_pending_request(self) -> bool:
        """Application predicate passed by BottomArea to prompt-toolkit Condition."""
        return self.pending_request is not None

    def handle(self, event: ApplicationEvent) -> None:
        """Application event entry point called by our parent, not prompt-toolkit."""
        if isinstance(event, ToolPermissionRequested):
            self.pending_request = event
            self._focus()
        elif isinstance(event, ToolPermissionResolved):
            if (
                self.pending_request is not None
                and self.pending_request.request_id == event.request_id
            ):
                self.pending_request = None
            self._restore_focus()
        elif isinstance(event, (ToolResultRecorded, RequestFailed)):
            self.pending_request = None
        else:
            return
        self.on_change.fire()

    def resolve(self, allowed: bool) -> None:
        if self.pending_request is not None:
            self._resolve(self.pending_request.request_id, allowed)

    def render_request(self) -> str:
        """Dynamic text callback passed to Label(text=...), evaluated when rendered."""
        pending = self.pending_request
        if pending is None:
            return ""
        serialized = json.dumps(
            thaw_json(pending.tool_input),
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(serialized) > MAX_PERMISSION_PREVIEW_CHARS:
            omitted = len(serialized) - MAX_PERMISSION_PREVIEW_CHARS
            serialized = (
                serialized[:MAX_PERMISSION_PREVIEW_CHARS]
                + f"… [{omitted} characters omitted]"
            )
        return (
            f"Tool: {pending.tool_name}\n"
            f"Arguments: {serialized}\n"
            "Select Deny or Allow once. Escape denies."
        )
