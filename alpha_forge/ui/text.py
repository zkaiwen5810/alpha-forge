"""Update read-only text while preserving its cursor and scroll behavior."""

from prompt_toolkit.document import Document
from prompt_toolkit.widgets import TextArea


def set_text(
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
