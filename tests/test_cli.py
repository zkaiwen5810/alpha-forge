import unittest
from unittest.mock import patch

from alpha_forge.cli import run_repl
from alpha_forge.config import Config


class CliTests(unittest.TestCase):
    def test_prompt_submits_on_enter(self) -> None:
        created = {}

        class FakeSession:
            def __init__(self, **kwargs) -> None:
                created.update(kwargs)

            def prompt(self, _message: str) -> str:
                return "/exit"

        with patch("alpha_forge.cli.PromptSession", FakeSession):
            with patch("alpha_forge.cli.print_formatted_text"):
                exit_code = run_repl(Config(api_key="sk-test"))

        self.assertEqual(exit_code, 0)
        self.assertFalse(created["multiline"])
