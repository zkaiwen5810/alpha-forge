"""Editor and suggestions, independent of permission and queue state."""

from collections.abc import Callable

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout import Container, Dimension, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl, UIControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import Event as WidgetEvent
from prompt_toolkit.widgets import TextArea

from alpha_forge.slash_commands import SLASH_COMMANDS
from alpha_forge.ui.component import UiComponent
from alpha_forge.ui.text import set_text


class InputPanel:
    def __init__(
        self,
        submit: Callable[[str], None],
        *,
        model_name: str,
    ) -> None:
        self._submit = submit
        self.on_change: WidgetEvent[UiComponent] = WidgetEvent(self)
        self.suggestions_area = TextArea(
            read_only=True,
            focusable=True,
            focus_on_click=True,
            scrollbar=True,
            wrap_lines=True,
            height=Dimension(min=3, max=3, preferred=3),
            style="class:slash-suggestions",
        )
        self.editor = TextArea(
            multiline=False,
            height=1,
            prompt="alpha> ",
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            accept_handler=self.accept_input,
            style="class:input",
        )
        self.editor.buffer.on_text_changed += self._input_changed
        self._model_footer = Window(
            FormattedTextControl(f"Model: {model_name}"),
            height=1,
            wrap_lines=False,
            style="class:model-footer",
        )
        suggestions_visible = Condition(self.show_suggestions)
        self._container = HSplit(
            [
                self.editor,
                ConditionalContainer(self._model_footer, filter=~suggestions_visible),
                ConditionalContainer(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(" Slash Commands"),
                                height=1,
                                style="class:section-title",
                            ),
                            self.suggestions_area,
                        ]
                    ),
                    filter=suggestions_visible,
                ),
            ]
        )
        self._disable_editor_scrolling()
        self.refresh()

    def __pt_container__(self) -> Container:
        """prompt-toolkit widget protocol: return the root container (not an override)."""
        return self._container

    @property
    def focus_target(self) -> UIControl:
        """Application widget API: the control our parent passes to Layout.focus()."""
        return self.editor.control

    @property
    def has_draft(self) -> bool:
        return bool(self.editor.text)

    def refresh(self) -> None:
        set_text(
            self.suggestions_area,
            self._render_slash_suggestions(),
            default_cursor="start",
        )
        self.on_change.fire()

    def accept_input(self, _buffer) -> bool:  # type: ignore[no-untyped-def]
        """Callback passed to TextArea(accept_handler=...), invoked on acceptance.

        True keeps the completed text; False lets Buffer reset after submission.
        The method name is ours, not a prompt-toolkit override.
        """
        if self.complete_slash_command():
            return True
        self._submit(self.editor.text)
        return False

    def show_suggestions(self) -> bool:
        """Predicate passed to Condition; prompt-toolkit reevaluates visibility."""
        text = self.editor.text
        return text.startswith("/") and " " not in text

    def _render_slash_suggestions(self) -> str:
        matches = self._matching_slash_commands()
        if not matches:
            if self.show_suggestions():
                return "No matching commands."
            return ""

        return "\n".join(
            f"{command.name:<8} {command.description}" for command in matches
        )

    def _matching_slash_commands(self):
        text = self.editor.text
        if not text.startswith("/") or " " in text:
            return []
        return [command for command in SLASH_COMMANDS if command.name.startswith(text)]

    def complete_slash_command(self) -> bool:
        matches = self._matching_slash_commands()
        if not matches:
            return False

        completion = matches[0].name
        if completion == self.editor.text:
            return False

        self.editor.document = Document(completion, cursor_position=len(completion))
        self.refresh()
        return True

    def _input_changed(self, _buffer) -> None:  # type: ignore[no-untyped-def]
        """Subscriber to Buffer.on_text_changed; the argument is the sending Buffer."""
        self.refresh()

    def _disable_editor_scrolling(self) -> None:
        """Consume editor wheel events before Window's default scroll handling."""
        original_mouse_handler = self.editor.control.mouse_handler

        def mouse_handler(mouse_event: MouseEvent):
            if mouse_event.event_type in (
                MouseEventType.SCROLL_UP,
                MouseEventType.SCROLL_DOWN,
            ):
                return None
            return original_mouse_handler(mouse_event)

        # Instance-level handler replacement, not a subclass method override.
        self.editor.control.mouse_handler = mouse_handler
