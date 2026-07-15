import unittest
from pathlib import Path

from alpha_forge.conversation import (
    AssistantMessage,
    Conversation,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolResultPreview,
    UserMessage,
)


class ConversationTests(unittest.TestCase):
    def test_keeps_multi_turn_history(self) -> None:
        conversation = Conversation(system_prompt="system")
        conversation.add_user("hello")
        conversation.add_assistant("hi")
        conversation.add_user("again")

        self.assertEqual(
            conversation.to_openai_messages(),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
        )

    def test_clear_keeps_system_prompt(self) -> None:
        conversation = Conversation(system_prompt="system")
        conversation.add_user("hello")
        conversation.clear()

        self.assertEqual(
            conversation.to_openai_messages(),
            [{"role": "system", "content": "system"}],
        )

    def test_serializes_assistant_tool_calls_and_results(self) -> None:
        conversation = Conversation()
        conversation.add_assistant(
            None,
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="calculator",
                    arguments='{"expression":"2+2"}',
                ),
            ),
        )
        conversation.add_tool(ToolMessage("4", tool_call_id="call-1"))

        self.assertEqual(
            conversation.to_openai_messages(),
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "4",
                    "tool_call_id": "call-1",
                },
            ],
        )

    def test_message_hierarchy_separates_role_specific_fields(self) -> None:
        system = SystemMessage("system")
        user = UserMessage("hello")
        assistant = AssistantMessage(
            None,
            tool_calls=(ToolCall("call-1", "calculator", "{}"),),
        )
        tool = ToolMessage("4", tool_call_id="call-1")

        self.assertFalse(hasattr(system, "tool_calls"))
        self.assertFalse(hasattr(user, "tool_call_id"))
        self.assertFalse(hasattr(assistant, "tool_call_id"))
        self.assertFalse(hasattr(tool, "tool_calls"))
        self.assertEqual(
            [message.role for message in (system, user, assistant, tool)],
            ["system", "user", "assistant", "tool"],
        )

    def test_multiple_tool_messages_remain_distinct_and_consecutive(self) -> None:
        conversation = Conversation()
        conversation.add_assistant(
            None,
            tool_calls=(
                ToolCall("call-1", "first", "{}"),
                ToolCall("call-2", "second", "{}"),
            ),
        )
        conversation.add_tool(ToolMessage("one", tool_call_id="call-1"))
        conversation.add_tool(ToolMessage("two", tool_call_id="call-2"))

        self.assertIsInstance(conversation.messages[0], AssistantMessage)
        self.assertIsInstance(conversation.messages[1], ToolMessage)
        self.assertIsInstance(conversation.messages[2], ToolMessage)
        self.assertEqual(
            conversation.to_openai_messages()[1:],
            [
                {"role": "tool", "content": "one", "tool_call_id": "call-1"},
                {"role": "tool", "content": "two", "tool_call_id": "call-2"},
            ],
        )

    def test_tool_metadata_stays_internal_to_wire_message(self) -> None:
        message = ToolMessage(
            "preview with /tmp/result.txt",
            tool_call_id="call-1",
            failed=True,
            preview=ToolResultPreview(
                persisted_path=Path("/tmp/result.txt"),
                original_chars=20_000,
                reason="individual_limit",
            ),
        )

        self.assertEqual(
            message.to_openai(),
            {
                "role": "tool",
                "content": "preview with /tmp/result.txt",
                "tool_call_id": "call-1",
            },
        )
