"""Rendered layout and region placement across terminal states and sizes."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from alpha_forge.application import ApplicationCoordinator
from alpha_forge.application.events import (
    InputQueued,
    SessionView,
    SessionViewChanged,
    ToolPermissionRequested,
)
from alpha_forge.config import Config
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.projectors.ui_history import UiPrompt
from alpha_forge.sessions import Session
from alpha_forge.ui.terminal import TerminalChatUi

SCENARIOS = ("ready", "queue", "suggestions", "permission")
SIZES = ((80, 24), (40, 16))
SNAPSHOTS = Path(__file__).with_name("fixtures") / "ui_layout.json"


def render_snapshot(width, height, scenario):
    return asyncio.run(_render_snapshot(width, height, scenario))


async def _render_snapshot(width, height, scenario):
    output = DummyOutput()
    output.get_size = lambda: Size(rows=height, columns=width)
    session = Session.create(in_memory=True)
    coordinator = ApplicationCoordinator(
        Config("key", model="gpt-test"), provider=Mock(), session=session
    )
    with create_pipe_input() as input:
        ui = TerminalChatUi(coordinator, input=input, output=output)
        try:
            coordinator.event_router.publish(
                SessionViewChanged(
                    SessionView(
                        "session",
                        2,
                        (
                            UiPrompt(
                                1,
                                "prompt",
                                "\n".join(f"history line {i}" for i in range(30)),
                            ),
                        ),
                    )
                )
            )
            if scenario != "ready":
                coordinator.event_router.publish(InputQueued("one", "queued prompt"))
            if scenario in ("suggestions", "permission"):
                ui.bottom_area.input_panel.editor.text = "/he"
            if scenario == "permission":
                coordinator.event_router.publish(
                    ToolPermissionRequested(
                        "request",
                        call_id="call",
                        tool_name="bash",
                        tool_input=FrozenJsonObject({"cmd": "pwd"}),
                    )
                )
            with set_app(ui.app):
                ui.app.renderer.render(ui.app, ui.app.layout)
            screen = ui.app.renderer._last_screen
            positions = screen.visible_windows_to_write_positions
            history_position = positions.get(ui.history_area.control.window)
            status_position = positions[ui.bottom_area._status_window]
            if history_position is not None:
                assert (
                    status_position.ypos
                    == history_position.ypos + history_position.height
                )
            else:
                # At the smallest permission size, only the history heading fits.
                assert (width, height, scenario) == (40, 16, "permission")
                assert status_position.ypos == 1
            assert status_position.height == 1
            panel = ui.bottom_area.input_panel
            footer_visible = panel._model_footer in positions
            assert footer_visible == (scenario in ("ready", "queue"))
            if footer_visible:
                editor_position = positions[panel.editor.window]
                footer_position = positions[panel._model_footer]
                assert (
                    footer_position.ypos
                    == editor_position.ypos + editor_position.height
                )
                assert footer_position.height == 1
            if scenario != "ready":
                queue_position = positions[
                    ui.bottom_area.queued_inputs.pending_area.window
                ]
                assert queue_position.ypos == status_position.ypos + 2  # Queue heading.
            return {
                "rows": [
                    "".join(
                        screen.data_buffer[y][x].char for x in range(width)
                    ).rstrip()
                    for y in range(height)
                ],
                "windows": [
                    [
                        window.style if isinstance(window.style, str) else "<dynamic>",
                        position.xpos,
                        position.ypos,
                        position.width,
                        position.height,
                    ]
                    for window, position in screen.visible_windows_to_write_positions.items()
                ],
            }
        finally:
            await ui.app.cancel_and_wait_for_background_tasks()
            ui._event_subscription.unsubscribe()
            session.close()


class UiLayoutTests(unittest.TestCase):
    def test_rendered_rows_and_window_geometry_are_unchanged(self):
        snapshots = json.loads(SNAPSHOTS.read_text())
        for width, height in SIZES:
            for scenario in SCENARIOS:
                key = f"{width}x{height}/{scenario}"
                with self.subTest(key=key):
                    self.assertEqual(
                        render_snapshot(width, height, scenario), snapshots[key]
                    )
