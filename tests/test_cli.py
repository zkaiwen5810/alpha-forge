import contextlib
import io
import unittest
from unittest.mock import patch

from prompt_toolkit.document import Document

from alpha_forge.cli import build_parser, run_repl
from alpha_forge.config import Config
from alpha_forge.slash_commands import SlashCommandCompleter


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

    def test_parser_does_not_expose_system_option(self) -> None:
        help_text = build_parser().format_help()

        self.assertNotIn("--system", help_text)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--system", "custom"])

    def test_slash_completer_matches_after_slash_character(self) -> None:
        completer = SlashCommandCompleter(["/clear", "/model"])

        completions = list(completer.get_completions(Document("/m"), None))

        self.assertEqual([completion.text for completion in completions], ["/model"])

    def test_slash_completer_ignores_normal_text(self) -> None:
        completer = SlashCommandCompleter(["/clear", "/model"])

        completions = list(completer.get_completions(Document("hello"), None))

        self.assertEqual(completions, [])

    def test_model_command_lists_models_and_marks_current_model(self) -> None:
        output = []

        class FakeSession:
            def __init__(self, **_kwargs) -> None:
                pass

            def prompt(self, _message: str) -> str:
                if not hasattr(self, "called"):
                    self.called = True
                    return "/model"
                return "/exit"

        class FakeChatClient:
            def __init__(self, _config: Config) -> None:
                pass

            def list_models(self) -> list[str]:
                return ["gpt-other", "gpt-test"]

        with patch("alpha_forge.cli.PromptSession", FakeSession):
            with patch("alpha_forge.cli.ChatClient", FakeChatClient):
                with patch("alpha_forge.cli.print_formatted_text", output.append):
                    exit_code = run_repl(Config(api_key="sk-test", model="gpt-test"))

        self.assertEqual(exit_code, 0)
        self.assertIn("  gpt-other", output)
        self.assertIn("* gpt-test", output)

    def test_model_command_marks_provider_prefixed_current_model(self) -> None:
        output = []

        class FakeSession:
            def __init__(self, **_kwargs) -> None:
                pass

            def prompt(self, _message: str) -> str:
                if not hasattr(self, "called"):
                    self.called = True
                    return "/model"
                return "/exit"

        class FakeChatClient:
            def __init__(self, _config: Config) -> None:
                pass

            def list_models(self) -> list[str]:
                return ["openai/gpt-4o"]

        with patch("alpha_forge.cli.PromptSession", FakeSession):
            with patch("alpha_forge.cli.ChatClient", FakeChatClient):
                with patch("alpha_forge.cli.print_formatted_text", output.append):
                    exit_code = run_repl(Config(api_key="sk-test", model="gpt-4o"))

        self.assertEqual(exit_code, 0)
        self.assertIn("* openai/gpt-4o", output)
