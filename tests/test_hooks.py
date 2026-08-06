import asyncio
import unittest

from alpha_forge.hooks import (
    Hook,
    HookRegistry,
    PermissionAction,
    PreToolExecution,
    match_lifecycle,
    match_tool_names,
)
from alpha_forge.json_values import FrozenJsonObject
from alpha_forge.providers import ToolCall
from alpha_forge.tools import Tool, ToolExecutor, ToolRegistry


class HookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.invocations: list[str] = []
        self.registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    aliases=("repeat",),
                    handler=lambda arguments: self._invoke(arguments),
                    description="echo",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                )
            ]
        )

    def _invoke(self, arguments):  # type: ignore[no-untyped-def]
        self.invocations.append("handler")
        return str(arguments["value"])

    def test_event_is_discriminated_and_input_is_immutable(self) -> None:
        event = PreToolExecution(
            call_id="call",
            tool_name="echo",
            tool_input=FrozenJsonObject({"value": ["a"]}),
        )

        self.assertEqual(event.lifecycle, "PreToolExecution")
        self.assertEqual(event.tool_input, {"value": ["a"]})
        with self.assertRaises(TypeError):
            event.tool_input["extra"] = True  # type: ignore[index]

    def test_multiple_matching_hooks_run_in_registration_order(self) -> None:
        observed: list[str] = []

        async def first(event: PreToolExecution) -> None:
            observed.append(f"first:{event.tool_name}")

        async def second(_event: PreToolExecution) -> None:
            observed.append("second")

        hooks = HookRegistry()
        hooks.register(Hook(match_lifecycle(PreToolExecution), first))
        hooks.register(Hook(match_tool_names("echo"), second))

        result = asyncio.run(
            ToolExecutor(self.registry, hooks).execute(
                ToolCall("call", "repeat", '{"value":"ok"}')
            )
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(observed, ["first:echo", "second"])
        self.assertEqual(self.invocations, ["handler"])

    def test_nonmatching_hooks_are_skipped(self) -> None:
        observed: list[str] = []

        async def action(_event: PreToolExecution) -> None:
            observed.append("called")

        hooks = HookRegistry([Hook(match_tool_names("other"), action)])
        result = asyncio.run(
            ToolExecutor(self.registry, hooks).execute(
                ToolCall("call", "echo", '{"value":"ok"}')
            )
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(observed, [])

    def test_invalid_input_never_reaches_hooks(self) -> None:
        observed: list[str] = []

        async def action(_event: PreToolExecution) -> None:
            observed.append("called")

        hooks = HookRegistry(
            [Hook(match_lifecycle(PreToolExecution), action)]
        )
        result = asyncio.run(
            ToolExecutor(self.registry, hooks).execute(
                ToolCall("call", "echo", '{"value":1}')
            )
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(observed, [])
        self.assertEqual(self.invocations, [])

    def test_permission_denial_short_circuits_hooks_and_handler(self) -> None:
        observed: list[str] = []

        async def deny(_event: PreToolExecution) -> bool:
            observed.append("permission")
            return False

        async def later(_event: PreToolExecution) -> None:
            observed.append("later")

        hooks = HookRegistry(
            [
                Hook(match_tool_names("echo"), PermissionAction(deny)),
                Hook(match_tool_names("echo"), later),
            ]
        )
        result = asyncio.run(
            ToolExecutor(self.registry, hooks).execute(
                ToolCall("call", "echo", '{"value":"no"}')
            )
        )

        self.assertEqual(result.status, "error")
        self.assertIn("permission denied", result.content)
        self.assertEqual(observed, ["permission"])
        self.assertEqual(self.invocations, [])

    def test_permission_is_requested_for_every_call(self) -> None:
        requests: list[str] = []

        async def allow(event: PreToolExecution) -> bool:
            requests.append(event.call_id)
            return True

        hooks = HookRegistry(
            [Hook(match_tool_names("echo"), PermissionAction(allow))]
        )
        executor = ToolExecutor(self.registry, hooks)

        async def execute_both():
            return await asyncio.gather(
                executor.execute(ToolCall("one", "echo", '{"value":"1"}')),
                executor.execute(ToolCall("two", "echo", '{"value":"2"}')),
            )

        results = asyncio.run(execute_both())

        self.assertEqual([result.status for result in results], ["success", "success"])
        self.assertEqual(requests, ["one", "two"])


if __name__ == "__main__":
    unittest.main()
