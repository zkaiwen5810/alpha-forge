import unittest

from alpha_forge.conversation import Conversation


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
