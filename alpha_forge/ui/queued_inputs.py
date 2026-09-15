"""Queued-input state and its conditional, titled display."""

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import Container, Dimension, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.utils import Event as WidgetEvent
from prompt_toolkit.widgets import TextArea

from alpha_forge.application.events import ApplicationEvent, InputQueued, InputStarted
from alpha_forge.ui.component import UiComponent
from alpha_forge.ui.text import set_text


class QueuedInputsPanel:
    def __init__(self) -> None:
        self.on_change: WidgetEvent[UiComponent] = WidgetEvent(self)
        self._queued_inputs: dict[str, str] = {}
        self.pending_area = TextArea(
            read_only=True,
            focusable=True,
            focus_on_click=True,
            scrollbar=True,
            wrap_lines=True,
            height=Dimension(preferred=4, max=6),
            style="class:pending",
        )
        self._container = ConditionalContainer(
            HSplit(
                [
                    Window(
                        FormattedTextControl(" Queued Inputs"),
                        height=1,
                        style="class:queue-heading",
                    ),
                    self.pending_area,
                ]
            ),
            filter=Condition(self.has_pending_inputs),
        )
        set_text(self.pending_area, self.render_pending())

    def __pt_container__(self) -> Container:
        """prompt-toolkit widget protocol: return the root container (not an override)."""
        return self._container

    @property
    def pending_inputs(self) -> list[str]:
        return list(self._queued_inputs.values())

    def has_pending_inputs(self) -> bool:
        """Predicate passed to Condition for prompt-toolkit visibility evaluation."""
        return bool(self._queued_inputs)

    def handle(self, event: ApplicationEvent) -> None:
        """Application event entry point called by our parent, not prompt-toolkit."""
        if isinstance(event, InputQueued):
            self._queued_inputs[event.item_id] = event.raw
        elif isinstance(event, InputStarted):
            self._queued_inputs.pop(event.item_id, None)
        else:
            return
        set_text(self.pending_area, self.render_pending())
        self.on_change.fire()

    def render_pending(self) -> str:
        """Application formatter called by this panel, not a toolkit render hook."""
        if not self.pending_inputs:
            return "No pending prompts."
        return "\n".join(
            f"{index}. {value}"
            for index, value in enumerate(self.pending_inputs, start=1)
        )
