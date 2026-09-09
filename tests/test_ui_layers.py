"""Composite widgets work without a coordinator or a terminal application."""

import unittest
from unittest.mock import Mock

from prompt_toolkit.layout import Window
from prompt_toolkit.layout.containers import to_container
from prompt_toolkit.layout.layout import walk

from alpha_forge.application.events import (
    InputQueued,
    InputStarted,
    RequestFailed,
    ToolPermissionRequested,
    ToolPermissionResolved,
    ToolResultRecorded,
)
from alpha_forge.hooks import PreToolExecution
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.ui.bottom import BottomArea


def permission_request(request_id="request"):
    return ToolPermissionRequested(
        request_id,
        PreToolExecution(
            call_id="call",
            tool_name="bash",
            tool_input=FrozenJsonObject({"cmd": "pwd"}),
        ),
    )


class BottomAreaTests(unittest.TestCase):
    def setUp(self):
        self.submit = Mock()
        self.resolve = Mock(return_value=True)
        self.exit = Mock()
        self.focus = Mock()
        self.area = BottomArea(
            model_name="gpt-test",
            submit=self.submit,
            resolve_permission=self.resolve,
            request_exit=self.exit,
            request_focus=self.focus,
        )

    def visible_controls(self):
        return [
            container.content
            for container in walk(to_container(self.area), skip_hidden=True)
            if isinstance(container, Window)
        ]

    def test_visibility_matrix_is_owned_by_bottom_area(self):
        for queued in (False, True):
            for draft in ("", "hello", "/he", "/missing", "/resume path"):
                for pending in (False, True):
                    with self.subTest(queued=queued, draft=draft, pending=pending):
                        self.area.handle(ToolPermissionResolved("request", False))
                        self.area.input_panel.editor.text = draft
                        self.area.handle(InputStarted("queued"))
                        if queued:
                            self.area.handle(InputQueued("queued", "waiting"))
                        if pending:
                            self.area.handle(permission_request())
                        controls = self.visible_controls()
                        self.assertIs(controls[0], self.area._status_window.content)
                        self.assertEqual(
                            self.area.input_panel._model_footer.content in controls,
                            not pending and draft not in ("/he", "/missing"),
                        )
                        self.assertEqual(
                            self.area.queued_inputs.pending_area.control in controls,
                            queued,
                        )
                        self.assertEqual(
                            self.area.input_panel.focus_target in controls, not pending
                        )
                        self.assertEqual(
                            self.area.permission_panel.focus_target in controls, pending
                        )
                        self.assertEqual(
                            self.area.input_panel.suggestions_area.control in controls,
                            not pending and draft in ("/he", "/missing"),
                        )

    def test_model_footer_returns_after_suggestions_and_permission(self):
        panel = self.area.input_panel
        footer = panel._model_footer.content
        for draft in ("hello", "/he", "/missing", "/resume path"):
            with self.subTest(draft=draft):
                panel.editor.text = draft
                expected_footer = not panel.show_suggestions()
                self.assertEqual(footer in self.visible_controls(), expected_footer)
                self.area.handle(permission_request())
                self.assertNotIn(footer, self.visible_controls())
                self.assertNotIn(
                    panel.suggestions_area.control, self.visible_controls()
                )
                self.area.handle(ToolPermissionResolved("request", False))
                self.assertEqual(panel.editor.text, draft)
                self.assertEqual(footer in self.visible_controls(), expected_footer)
                self.assertEqual(
                    panel.suggestions_area.control in self.visible_controls(),
                    not expected_footer,
                )

    def test_status_feedback_notifies_parent_and_retains_override(self):
        observed = []
        self.area.on_change += lambda sender: observed.append(sender.status_message)
        self.area.show_status_message("transcript copied")
        self.assertEqual(observed, ["transcript copied"])
        self.area.handle(InputQueued("queued", "waiting"))
        self.assertEqual(self.area.status_message, "transcript copied")
        self.assertEqual(self.area.queued_inputs.pending_inputs, ["waiting"])

    def test_child_change_bubbles_after_local_text_update(self):
        observed = []
        self.area.on_change += lambda sender: observed.append(
            (sender, self.area.input_panel.suggestions_area.text)
        )
        self.area.input_panel.editor.text = "/he"
        self.assertTrue(observed)
        self.assertTrue(all(sender is self.area for sender, _text in observed))
        self.assertIn("/help", observed[-1][1])
        self.assertFalse(self.submit.called)
        self.assertFalse(self.resolve.called)

    def test_status_and_permission_state_are_updated_before_focus_hook(self):
        observations = []
        self.focus.side_effect = lambda target: observations.append(
            (
                target,
                self.area.status_message,
                self.area.permission_panel.pending_request,
            )
        )
        request = permission_request()
        self.area.handle(request)
        self.area.handle(ToolPermissionResolved("request", True))
        self.assertEqual(
            observations,
            [
                (
                    self.area.permission_panel.focus_target,
                    "Approval required: bash",
                    request,
                ),
                (self.area.input_panel.focus_target, "Running approved tool", None),
            ],
        )

    def test_actions_are_forwarded_without_application_dependency(self):
        self.area.input_panel.editor.text = "hello"
        self.assertFalse(self.area.input_panel.accept_input(None))
        self.submit.assert_called_once_with("hello")
        self.area.handle(permission_request())
        self.area.permission_panel.resolve(True)
        self.resolve.assert_called_once_with("request", True)
        # Resolution stays coordinator-owned: clicking does not clear request state.
        self.assertIsNotNone(self.area.permission_panel.pending_request)
        self.area.handle(ToolPermissionResolved("request", True))
        self.assertIsNone(self.area.permission_panel.pending_request)

    def test_failure_and_result_cleanup_keep_existing_focus_behavior(self):
        for event in (
            RequestFailed("failed"),
            ToolResultRecorded("result", "output", "call"),
        ):
            with self.subTest(event=event):
                self.area.handle(permission_request())
                self.focus.reset_mock()
                self.area.handle(event)
                self.assertIsNone(self.area.permission_panel.pending_request)
                self.assertFalse(self.focus.called)
                self.assertIn(
                    self.area.input_panel.focus_target, self.visible_controls()
                )

    def test_stale_resolution_preserves_request_but_still_requests_editor_focus(self):
        self.area.handle(permission_request())
        self.focus.reset_mock()
        self.area.handle(ToolPermissionResolved("stale", False))
        self.assertEqual(
            self.area.permission_panel.pending_request.request_id, "request"
        )
        self.focus.assert_called_once_with(self.area.input_panel.focus_target)
