"""Prompt-toolkit terminal UI for Alpha Forge chat."""

from __future__ import annotations

import base64
import textwrap

from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from alpha_forge.session import ChatReplController
from alpha_forge.slash_commands import SLASH_COMMANDS


class HistoryControl(UIControl):
    def __init__(self, ui: "TerminalChatUi") -> None:
        self.ui = ui

    def is_focusable(self) -> bool:
        """Tell prompt-toolkit this custom control can receive focus."""
        return True

    def create_content(self, _width: int, _height: int | None) -> UIContent:
        """Build the renderable content prompt-toolkit asks a UIControl for."""
        lines = self.ui._history_line_fragments(_width, _height)
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=self.ui._history_cursor_position(),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent):
        """Handle mouse events that prompt-toolkit routes to this control."""
        return self.ui._handle_mouse_scroll(mouse_event)


class TerminalChatUi:
    def __init__(
        self,
        controller: ChatReplController,
        *,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.controller = controller
        self._history_follow_tail = True
        self._history_scroll_offset = 0
        self._history_total_lines = 1
        self._history_view_height = 1
        self._mouse_enabled = True
        self._ui_status = ""
        self.history_control = HistoryControl(self)
        self.history_window = Window(
            content=self.history_control,
            wrap_lines=False,
            style="class:history",
            right_margins=[ScrollbarMargin(display_arrows=False)],
            get_vertical_scroll=self._get_history_vertical_scroll,
            always_hide_cursor=True,
        )
        self.history_container = self.history_window
        self.pending_area = TextArea(
            read_only=True,
            focusable=True,
            focus_on_click=True,
            scrollbar=True,
            wrap_lines=True,
            height=Dimension(preferred=4, max=6),
            style="class:pending",
        )
        self.slash_suggestions_area = TextArea(
            read_only=True,
            focusable=True,
            focus_on_click=True,
            scrollbar=True,
            wrap_lines=True,
            height=Dimension(min=3, max=3, preferred=3),
            style="class:slash-suggestions",
        )
        self.input_area = TextArea(
            multiline=False,
            height=1,
            prompt="alpha> ",
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            accept_handler=self._accept_input,
            style="class:input",
        )
        self.input_area.buffer.on_text_changed += self._input_changed
        self._route_scroll_events_to_history(self.input_area.control)
        self._route_scroll_events_to_history(self.slash_suggestions_area.control)

        self.app = Application(
            layout=Layout(self._root_container(), focused_element=self.input_area),
            key_bindings=self._key_bindings(),
            style=self._style(),
            full_screen=True,
            mouse_support=Condition(lambda: self._mouse_enabled),
            input=input,
            output=output,
        )
        # Wire controller events back into this prompt-toolkit view. The
        # controller owns chat/session state; the UI owns rendering and app
        # shutdown, so these hooks are the boundary between the two.
        self.controller.request_redraw = self.refresh
        self.controller.request_app_exit = self.exit
        self.refresh()

    async def run_async(self) -> int:
        """Run the underlying prompt-toolkit application until it exits."""
        result = await self.app.run_async()
        return int(result or 0)

    def refresh(self) -> None:
        """Refresh widget text and ask prompt-toolkit to redraw the screen."""
        self._set_text(self.pending_area, self.controller.state.render_pending())
        self._set_text(
            self.slash_suggestions_area,
            self._render_slash_suggestions(),
            default_cursor="start",
        )
        if self.app.is_running:
            self.app.invalidate()

    def exit(self, exit_code: int) -> None:
        """Exit callback installed on the controller for prompt-toolkit shutdown."""
        if self.app.is_running:
            self.app.exit(result=exit_code)

    def _accept_input(self, _buffer) -> bool:  # type: ignore[no-untyped-def]
        """TextArea accept_handler hook called by prompt-toolkit on Enter."""
        if self._complete_slash_command():
            return True
        self.controller.submit(self.input_area.text)
        return False

    def _root_container(self) -> HSplit:
        return HSplit(
            [
                Window(
                    FormattedTextControl(self._status_text),
                    height=1,
                    style="class:status",
                ),
                Window(
                    FormattedTextControl(" Conversation"),
                    height=1,
                    style="class:section-title",
                ),
                self.history_container,
                ConditionalContainer(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(" Queued Prompts"),
                                height=1,
                                style="class:section-title",
                            ),
                            self.pending_area,
                        ]
                    ),
                    filter=Condition(self._has_pending_prompts),
                ),
                self.input_area,
                ConditionalContainer(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(" Slash Commands"),
                                height=1,
                                style="class:section-title",
                            ),
                            self.slash_suggestions_area,
                        ]
                    ),
                    filter=Condition(self._should_show_slash_suggestions),
                ),
            ]
        )

    def _has_pending_prompts(self) -> bool:
        return bool(self.controller.state.pending_prompts)

    def _should_show_slash_suggestions(self) -> bool:
        text = self.input_area.text
        return text.startswith("/") and " " not in text

    def _render_slash_suggestions(self) -> str:
        matches = self._matching_slash_commands()
        if not matches:
            if self._should_show_slash_suggestions():
                return "No matching commands."
            return ""

        return "\n".join(
            f"{command.name:<8} {command.description}"
            for command in matches
        )

    def _matching_slash_commands(self):
        text = self.input_area.text
        if not text.startswith("/") or " " in text:
            return []
        return [
            command
            for command in SLASH_COMMANDS
            if command.name.startswith(text)
        ]

    def _complete_slash_command(self) -> bool:
        matches = self._matching_slash_commands()
        if not matches:
            return False

        completion = matches[0].name
        if completion == self.input_area.text:
            return False

        self.input_area.document = Document(completion, cursor_position=len(completion))
        self.refresh()
        return True

    def _input_changed(self, _buffer) -> None:  # type: ignore[no-untyped-def]
        """Buffer change hook used to update slash suggestions while typing."""
        self.refresh()

    def _history_fragments(self) -> list[tuple[str, str]]:
        lines = self._history_display_line_fragments(width=80)
        fragments: list[tuple[str, str]] = []
        for lineno, line in enumerate(lines):
            fragments.extend(line)
            if lineno < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments

    def _history_line_fragments(
        self,
        width: int,
        height: int | None,
    ) -> list[list[tuple[str, str]]]:
        lines = self._history_display_line_fragments(width)
        self._history_total_lines = max(1, len(lines))
        if height is not None:
            self._history_view_height = max(1, height)
        self._sync_history_scroll()
        return lines or [[("", "")]]

    def _history_display_line_fragments(self, width: int) -> list[list[tuple[str, str]]]:
        fragments: list[list[tuple[str, str]]] = []
        for line in self.controller.state.history_lines():
            style = self._history_line_style(line.role)
            for wrapped_line in self._wrap_history_line(line.text, width):
                fragments.append([(style, wrapped_line)])
        return fragments

    @staticmethod
    def _wrap_history_line(line: str, width: int) -> list[str]:
        wrap_width = max(1, width)
        return textwrap.wrap(
            line,
            width=wrap_width,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=False,
            replace_whitespace=False,
        ) or [""]

    @staticmethod
    def _history_line_style(role: str) -> str:
        return {
            "user": "class:user-message",
            "assistant": "class:assistant-message",
            "assistant_note": "class:assistant-note-message",
            "tool_call": "class:tool-call-message",
            "tool_result": "class:tool-result-message",
            "notice": "class:notice-message",
            "error": "class:error-message",
            "spacer": "",
        }[role]

    def _status_text(self) -> str:
        """FormattedTextControl callback that recomputes the status bar text."""
        mouse_mode = "mouse-scroll" if self._mouse_enabled else "copy-select"
        status = self._ui_status or self.controller.state.status
        return (
            f" Alpha Forge | {self.controller.config.model} | "
            f"{status} | {mouse_mode} | F2 toggle"
        )

    def _key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _handle_ctrl_c(_event) -> None:  # type: ignore[no-untyped-def]
            self.controller.request_exit()

        @bindings.add("c-d")
        def _handle_ctrl_d(_event) -> None:  # type: ignore[no-untyped-def]
            if not self.input_area.text:
                self.controller.request_exit()

        @bindings.add("tab")
        def _handle_tab(_event) -> None:  # type: ignore[no-untyped-def]
            self._complete_slash_command()

        @bindings.add("pageup")
        def _handle_pageup(_event) -> None:  # type: ignore[no-untyped-def]
            self._scroll_history_page(-1)

        @bindings.add("pagedown")
        def _handle_pagedown(_event) -> None:  # type: ignore[no-untyped-def]
            self._scroll_history_page(1)

        @bindings.add(Keys.ScrollUp)
        def _handle_scroll_up(_event) -> None:  # type: ignore[no-untyped-def]
            self._scroll_history_lines(-3)

        @bindings.add(Keys.ScrollDown)
        def _handle_scroll_down(_event) -> None:  # type: ignore[no-untyped-def]
            self._scroll_history_lines(3)

        @bindings.add("f2")
        def _handle_f2(_event) -> None:  # type: ignore[no-untyped-def]
            self._toggle_mouse_mode()

        @bindings.add("f3")
        def _handle_f3(_event) -> None:  # type: ignore[no-untyped-def]
            self._copy_history_to_terminal_clipboard()

        return bindings

    def _toggle_mouse_mode(self) -> None:
        self._mouse_enabled = not self._mouse_enabled
        self._ui_status = (
            "mouse scroll enabled"
            if self._mouse_enabled
            else "terminal selection enabled"
        )
        if self.app.is_running:
            self.app.invalidate()

    def _copy_history_to_terminal_clipboard(self) -> None:
        encoded = base64.b64encode(self._history_plain_text().encode()).decode()
        self.app.output.write_raw(f"\x1b]52;c;{encoded}\a")
        self.app.output.flush()
        self._ui_status = "conversation copied"
        if self.app.is_running:
            self.app.invalidate()

    def _history_plain_text(self) -> str:
        return self.controller.state.history_text()

    def _route_scroll_events_to_history(self, control) -> None:  # type: ignore[no-untyped-def]
        """Wrap a prompt-toolkit mouse handler so wheel events scroll history."""
        original_mouse_handler = control.mouse_handler

        def mouse_handler(mouse_event: MouseEvent):
            """Mouse handler shim installed onto prompt-toolkit controls."""
            result = self._handle_mouse_scroll(mouse_event)
            if result is not NotImplemented:
                return result
            return original_mouse_handler(mouse_event)

        control.mouse_handler = mouse_handler

    def _handle_mouse_scroll(self, mouse_event: MouseEvent):
        """Shared prompt-toolkit mouse event hook for history scrolling."""
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._scroll_history_lines(-3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll_history_lines(3)
            return None
        return NotImplemented

    def _scroll_history_page(self, direction: int) -> None:
        page_height = self._history_page_height()
        self._scroll_history_lines(direction * page_height)

    def _scroll_history_lines(self, amount: int) -> None:
        max_scroll = self._history_scroll_max()
        current_scroll = (
            max_scroll
            if self._history_follow_tail
            else self._history_scroll_offset
        )
        next_scroll = current_scroll + amount
        self._history_scroll_offset = min(max(next_scroll, 0), max_scroll)
        self._history_follow_tail = self._history_is_at_bottom()
        if self.app.is_running:
            self.app.invalidate()

    def _get_history_vertical_scroll(self, _window: Window) -> int:
        """Window get_vertical_scroll callback for the history viewport."""
        self._sync_history_scroll()
        return self._history_scroll_offset

    def _sync_history_scroll(self) -> None:
        max_scroll = self._history_scroll_max()
        if self._history_follow_tail:
            self._history_scroll_offset = max_scroll
        else:
            self._history_scroll_offset = min(
                max(self._history_scroll_offset, 0),
                max_scroll,
            )

    def _history_cursor_position(self) -> Point:
        self._sync_history_scroll()
        return Point(
            x=0,
            y=min(self._history_scroll_offset, self._history_total_lines - 1),
        )

    def _history_page_height(self) -> int:
        return max(1, self._history_view_height - 1)

    def _history_scroll_max(self) -> int:
        return max(0, self._history_total_lines - self._history_view_height)

    def _history_is_at_bottom(self) -> bool:
        return self._history_scroll_offset >= self._history_scroll_max()

    @staticmethod
    def _set_text(
        area: TextArea,
        text: str,
        *,
        default_cursor: str = "end",
    ) -> None:
        if area.text == text:
            return

        old_position = area.buffer.cursor_position
        was_at_end = old_position >= len(area.text)
        was_scrolled_up = area.window.vertical_scroll > 0
        if default_cursor == "start":
            cursor_position = 0
        elif was_at_end and not was_scrolled_up:
            cursor_position = len(text)
        else:
            cursor_position = min(old_position, len(text))

        area.document = Document(text, cursor_position=cursor_position)

    @staticmethod
    def _style() -> Style:
        return Style.from_dict(
            {
                "status": "reverse",
                "section-title": "bold",
                "history": "",
                "pending": "",
                "input": "",
                "slash-suggestions": "#bbbbbb",
                "scrollbar.background": "#666666",
                "scrollbar.button": "bg:#bbbbbb",
                "user-message": "bg:#2f4f4f #ffffff",
                "assistant-message": "",
                "assistant-note-message": "italic #5fd7ff",
                "tool-call-message": "#5fafff",
                "tool-result-message": "#5faf87",
                "notice-message": "#888888",
                "error-message": "#ff5f5f",
            }
        )
