"""Own the interaction region: queue, input, permission, status, and focus policy."""

from collections.abc import Callable

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl, UIControl
from prompt_toolkit.utils import Event as WidgetEvent

from alpha_forge.application.events import ApplicationEvent
from alpha_forge.ui.component import UiComponent
from alpha_forge.ui.input import InputPanel
from alpha_forge.ui.permission import PermissionPanel
from alpha_forge.ui.queued_inputs import QueuedInputsPanel
from alpha_forge.ui.status import StatusState


class BottomArea:
    def __init__(
        self,
        *,
        model_name: str,
        submit: Callable[[str], None],
        resolve_permission: Callable[[str, bool], bool],
        request_exit: Callable[[], None],
        request_focus: Callable[[UIControl], None],
    ) -> None:
        self.on_change: WidgetEvent[UiComponent] = WidgetEvent(self)
        self._submit = submit
        self._resolve_permission = resolve_permission
        self._request_exit = request_exit
        self._request_focus = request_focus
        self._status_state = StatusState()
        self._status_override = ""
        self._status_window = Window(
            # Dynamic text callback evaluated by prompt-toolkit when rendering.
            FormattedTextControl(lambda: self.status_message),
            height=1,
            wrap_lines=False,
            style="class:status",
        )
        self.queued_inputs = QueuedInputsPanel()
        self.input_panel = InputPanel(self._submit_input, model_name=model_name)
        self.permission_panel = PermissionPanel(
            self._resolve_tool_permission,
            focus=self._focus_permission,
            restore_focus=self._focus_input,
        )
        permission_pending = Condition(self.permission_panel.has_pending_request)
        self._permission_container = ConditionalContainer(
            self.permission_panel, filter=permission_pending
        )
        self._input_container = ConditionalContainer(
            self.input_panel, filter=~permission_pending
        )
        self._container = HSplit(
            [
                self._status_window,
                self.queued_inputs,
                self._permission_container,
                self._input_container,
            ]
        )
        for child in (self.queued_inputs, self.input_panel, self.permission_panel):
            child.on_change += self._child_changed
        self.key_bindings = self._key_bindings()

    def __pt_container__(self) -> Container:
        """prompt-toolkit widget protocol: return the root container (not an override)."""
        return self._container

    @property
    def status_message(self) -> str:
        return self._status_override or self._status_state.message

    def show_status_message(self, message: str) -> None:
        """Display UI feedback with the existing persistent override behavior."""
        self._status_override = message
        self.on_change.fire()

    @property
    def focus_target(self) -> UIControl:
        """Application widget API: the control our parent passes to Layout.focus()."""
        return (
            self.permission_panel.focus_target
            if self.permission_panel.has_pending_request()
            else self.input_panel.focus_target
        )

    def handle(self, event: ApplicationEvent) -> None:
        """Application event entry point called by our parent, not prompt-toolkit."""
        self.queued_inputs.handle(event)
        status_changed = self._status_state.handle(
            event, queued_count=len(self.queued_inputs.pending_inputs)
        )
        # Permission hooks see updated status and request state before focusing.
        self.permission_panel.handle(event)
        if status_changed:
            self.on_change.fire()

    def _child_changed(self, _child: UiComponent) -> None:
        """Subscriber to child widget notifications; requests a parent notification."""
        self.on_change.fire()

    def _submit_input(self, text: str) -> None:
        self._submit(text)

    def _resolve_tool_permission(self, request_id: str, allowed: bool) -> bool:
        return self._resolve_permission(request_id, allowed)

    def _focus_permission(self) -> None:
        self._request_focus(self.permission_panel.focus_target)

    def _focus_input(self) -> None:
        self._request_focus(self.input_panel.focus_target)

    def _key_bindings(self) -> KeyBindings:
        """Build bindings explicitly; @bindings.add registers toolkit callbacks."""
        bindings = KeyBindings()
        permission_pending = Condition(self.permission_panel.has_pending_request)

        @bindings.add("c-d")
        def exit_if_empty(_event) -> None:
            if not self.input_panel.has_draft:
                self._request_exit()

        @bindings.add("tab", filter=~permission_pending)
        def complete_command(_event) -> None:
            self.input_panel.complete_slash_command()

        @bindings.add("escape", filter=permission_pending)
        def deny_permission(_event) -> None:
            self.permission_panel.resolve(False)

        return bindings
