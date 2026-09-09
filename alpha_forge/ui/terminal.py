"""Compose terminal components and route application events and global input."""

from __future__ import annotations

import base64

from prompt_toolkit.application import Application
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.controls import UIControl
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style

from alpha_forge.application.coordinator import ApplicationCoordinator
from alpha_forge.application.events import ExitReady
from alpha_forge.events import Event
from alpha_forge.ui.bottom import BottomArea
from alpha_forge.ui.component import UiComponent
from alpha_forge.ui.history import HistoryArea


class TerminalChatUi:
    def __init__(
        self,
        coordinator: ApplicationCoordinator,
        *,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.history_area = HistoryArea(coordinator.session_view)
        self.bottom_area = BottomArea(
            model_name=coordinator.config.model,
            submit=self._submit_input,
            resolve_permission=self._resolve_permission,
            request_exit=self._request_exit,
            request_focus=self._focus,
        )
        self.app = Application(
            layout=Layout(
                self._root_container(), focused_element=self.bottom_area.focus_target
            ),
            key_bindings=merge_key_bindings(
                [
                    self.history_area.key_bindings,
                    self.bottom_area.key_bindings,
                    self._key_bindings(),
                ]
            ),
            style=self._style(),
            full_screen=True,
            mouse_support=True,
            input=input,
            output=output,
        )
        self.history_area.on_change += self._component_changed
        self.bottom_area.on_change += self._component_changed
        self._event_subscription = coordinator.event_router.subscribe(
            Event,
            self._handle_application_event,
        )

    def _redraw(self) -> None:
        if self.app.is_running:
            self.app.invalidate()

    def _component_changed(self, _component: UiComponent) -> None:
        """Subscriber to our components' on_change notifications; request repaint."""
        self._redraw()

    def _handle_application_event(self, event: Event) -> None:
        """Subscriber to the coordinator event router, not a toolkit callback."""
        self.history_area.handle(event)
        self.bottom_area.handle(event)
        if isinstance(event, ExitReady):
            self.exit(event.exit_code)

    def _submit_input(self, text: str) -> None:
        self.coordinator.submit(text)

    def _resolve_permission(self, request_id: str, allowed: bool) -> bool:
        return self.coordinator.resolve_tool_permission(request_id, allowed)

    def _request_exit(self) -> None:
        self.coordinator.request_exit()

    def _focus(self, target: UIControl) -> None:
        self.app.layout.focus(target)

    async def run_async(self) -> int:
        """Application API: delegate to Application.run_async(); not an override."""
        result = await self.app.run_async()
        return int(result or 0)

    def exit(self, exit_code: int) -> None:
        """Application API: delegate to Application.exit(); not an override."""
        if self.app.is_running:
            self.app.exit(result=exit_code)

    def _root_container(self) -> HSplit:
        return HSplit([self.history_area, self.bottom_area])

    def _key_bindings(self) -> KeyBindings:
        """Build bindings explicitly; @bindings.add registers toolkit callbacks."""
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _handle_ctrl_c(_event) -> None:  # type: ignore[no-untyped-def]
            self._request_exit()

        @bindings.add(Keys.ScrollUp)
        @bindings.add(Keys.ScrollDown)
        def _ignore_positionless_scroll(_event) -> None:
            # These keys carry no pointer position. Suppress prompt-toolkit's
            # default translation into Up/Down, which recalls editor history.
            pass

        @bindings.add("f3")
        def _handle_f3(_event) -> None:  # type: ignore[no-untyped-def]
            self._copy_history_to_terminal_clipboard()

        return bindings

    def _copy_history_to_terminal_clipboard(self) -> None:
        encoded = base64.b64encode(
            self.history_area.transcript_text().encode()
        ).decode()
        self.app.output.write_raw(f"\x1b]52;c;{encoded}\a")
        self.app.output.flush()
        self.bottom_area.show_status_message("transcript copied")

    @staticmethod
    def _style() -> Style:
        return Style.from_dict(
            {
                "status": "italic #888888",
                "section-title": "bold",
                "history": "",
                "active": "",
                "pending": "#aaaaaa",
                "queue-heading": "bold #888888",
                "model-footer": "#888888",
                "permission": "#ffdf5f",
                "dialog.body": "",
                "frame.label": "bold #ffdf5f",
                "button": "#bbbbbb",
                "button.focused": "reverse",
                "input": "",
                "slash-suggestions": "#bbbbbb",
                "scrollbar.background": "#666666",
                "scrollbar.button": "bg:#bbbbbb",
                "user-message": "bg:#2f4f4f #ffffff",
                "assistant-message": "",
                "assistant-note-message": "italic #5fd7ff",
                "tool-call-message": "#5fafff",
                "tool-result-message": "#5faf87",
                "token-usage-message": "italic #87af87",
                "notice-message": "#888888",
                "error-message": "#ff5f5f",
            }
        )
