"""Common widget surface; application events and actions are separate capabilities."""

from __future__ import annotations

from typing import Protocol

from prompt_toolkit.layout import Container
from prompt_toolkit.utils import Event as WidgetEvent


class UiComponent(Protocol):
    """Compose a widget and observe presentation changes without accessing children.

    Notifications request a repaint after local updates. They never redistribute
    application events. Event-consuming components additionally expose handle().

    This is our structural Protocol, not a prompt-toolkit base class. Only
    __pt_container__ is discovered by prompt-toolkit itself. on_change, handle(),
    and focus_target are our conventions, wired explicitly by parent components.
    """

    @property
    def on_change(self) -> WidgetEvent[UiComponent]: ...

    def __pt_container__(self) -> Container:
        """prompt-toolkit discovers this method through to_container()."""
        ...
