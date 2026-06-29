import unittest

from alpha_forge.conversation import Conversation, ToolCall


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

        self.assertEqual(conversation.to_openai_messages(), [{"role": "system", "content": "system"}])

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
        conversation.add_tool("4", tool_call_id="call-1")

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
