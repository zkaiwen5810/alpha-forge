"""Conversation history control, including its viewport and presentation state."""

import textwrap
from collections.abc import Callable
from typing import override

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text.base import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import Event as WidgetEvent

from alpha_forge.application.events import ApplicationEvent, SessionView
from alpha_forge.ui.component import UiComponent
from alpha_forge.ui.history_state import HistoryState


class HistoryControl(UIControl):
    def __init__(self, view: SessionView, redraw: Callable[[], None]) -> None:
        self.state = HistoryState(view)
        self._redraw = redraw
        self._follow_tail = True
        self._scroll_offset = 0
        self._total_lines = 1
        self._view_height = 1
        self.window = Window(
            content=self,
            wrap_lines=False,
            style="class:history",
            right_margins=[ScrollbarMargin(display_arrows=False)],
            get_vertical_scroll=self.vertical_scroll,
            always_hide_cursor=True,
        )

    def handle(self, event: ApplicationEvent) -> bool:
        """Application event entry point, independent of the UIControl interface."""
        return self.state.handle(event)

    def transcript_text(self) -> str:
        return self.state.transcript_text()

    @override
    def is_focusable(self) -> bool:
        """UIControl override queried by prompt-toolkit during focus navigation."""
        return True

    @override
    def create_content(self, width: int, height: int) -> UIContent:
        """UIControl override: Window requests styled lines before viewport drawing."""
        lines = self._line_fragments(width, height)
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=self._cursor_position(),
            show_cursor=False,
        )

    def _line_fragments(
        self,
        width: int,
        height: int | None,
    ) -> list[StyleAndTextTuples]:
        lines = self._display_line_fragments(width)
        self._total_lines = max(1, len(lines))
        if height is not None:
            self._view_height = max(1, height)
        self._sync_scroll()
        return lines or [[("", "")]]

    def _display_line_fragments(
        self,
        width: int,
    ) -> list[StyleAndTextTuples]:
        fragments: list[StyleAndTextTuples] = []
        for line in self.state.history_lines():
            style = self._line_style(line.role)
            text = (
                line.text.rjust(width)
                if line.role == "token_usage" and len(line.text) <= width
                else line.text
            )
            for wrapped_line in self._wrap_line(text, width):
                fragments.append([(style, wrapped_line)])
        return fragments

    @staticmethod
    def _wrap_line(line: str, width: int) -> list[str]:
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
    def _line_style(role: str) -> str:
        return {
            "user": "class:user-message",
            "assistant": "class:assistant-message",
            "assistant_note": "class:assistant-note-message",
            "tool_call": "class:tool-call-message",
            "tool_result": "class:tool-result-message",
            "token_usage": "class:token-usage-message",
            "notice": "class:notice-message",
            "error": "class:error-message",
            "spacer": "",
        }[role]

    @override
    def mouse_handler(self, mouse_event: MouseEvent):
        """UIControl override: None consumes the event; NotImplemented defers to Window."""
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll_lines(-3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll_lines(3)
            return None
        return NotImplemented

    def scroll_page(self, direction: int) -> None:
        page_height = self._page_height()
        self.scroll_lines(direction * page_height)

    def scroll_lines(self, amount: int) -> None:
        max_scroll = self._scroll_max()
        current_scroll = max_scroll if self._follow_tail else self._scroll_offset
        next_scroll = current_scroll + amount
        self._scroll_offset = min(max(next_scroll, 0), max_scroll)
        self._follow_tail = self._is_at_bottom()
        self._redraw()

    def vertical_scroll(self, _window: Window) -> int:
        """Callback passed to Window(get_vertical_scroll=...), not a UIControl override."""
        self._sync_scroll()
        return self._scroll_offset

    def _sync_scroll(self) -> None:
        max_scroll = self._scroll_max()
        if self._follow_tail:
            self._scroll_offset = max_scroll
        else:
            self._scroll_offset = min(
                max(self._scroll_offset, 0),
                max_scroll,
            )

    def _cursor_position(self) -> Point:
        self._sync_scroll()
        return Point(
            x=0,
            y=min(self._scroll_offset, self._total_lines - 1),
        )

    def _page_height(self) -> int:
        return max(1, self._view_height - 1)

    def _scroll_max(self) -> int:
        return max(0, self._total_lines - self._view_height)

    def _is_at_bottom(self) -> bool:
        return self._scroll_offset >= self._scroll_max()


class HistoryArea:
    """Scrollable conversation region, composed as one widget by its parent."""

    def __init__(self, view: SessionView) -> None:
        self.on_change: WidgetEvent[UiComponent] = WidgetEvent(self)
        self.control = HistoryControl(view, self.on_change.fire)
        self._container = HSplit(
            [
                Window(
                    FormattedTextControl(" Conversation"),
                    height=1,
                    style="class:section-title",
                ),
                HSplit([self.control.window]),
            ]
        )
        self.key_bindings = self._key_bindings()

    def __pt_container__(self) -> Container:
        """prompt-toolkit widget protocol: return the root container (not an override)."""
        return self._container

    def handle(self, event: ApplicationEvent) -> None:
        """Application event entry point called by our parent, not prompt-toolkit."""
        if self.control.handle(event):
            self.on_change.fire()

    def transcript_text(self) -> str:
        return self.control.transcript_text()

    def _key_bindings(self) -> KeyBindings:
        """Build bindings explicitly; @bindings.add registers toolkit callbacks."""
        bindings = KeyBindings()

        @bindings.add("pageup")
        def page_up(_event) -> None:
            self.control.scroll_page(-1)

        @bindings.add("pagedown")
        def page_down(_event) -> None:
            self.control.scroll_page(1)

        return bindings
