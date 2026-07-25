import unittest

from alpha_forge.model_messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class WireMessageTests(unittest.TestCase):
    def test_serializes_role_specific_messages(self) -> None:
        messages = [
            SystemMessage("system"),
            UserMessage("hello"),
            AssistantMessage(
                None,
                tool_calls=(
                    ToolCall(
                        "call-1",
                        "calculator",
                        '{"expression":"2+2"}',
                    ),
                ),
            ),
            ToolMessage("4", tool_call_id="call-1", failed=True),
        ]

        self.assertEqual(
            [message.to_openai() for message in messages],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
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
                {"role": "tool", "content": "4", "tool_call_id": "call-1"},
            ],
        )

    def test_message_types_only_expose_role_specific_fields(self) -> None:
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

    def test_assistant_serializes_reasoning_extension_and_refusal(self) -> None:
        assistant = AssistantMessage(
            "answer",
            reasoning_content="reason",
            refusal="refusal",
        )

        self.assertEqual(
            assistant.to_openai(),
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reason",
                "refusal": "refusal",
            },
        )
