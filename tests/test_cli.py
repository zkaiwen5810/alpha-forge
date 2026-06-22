import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.document import Document

from alpha_forge.cli import build_parser, main, run_repl
from alpha_forge.config import Config
from alpha_forge.slash_commands import SlashCommandCompleter


@contextlib.contextmanager
def chdir(path: str | os.PathLike[str]):
    """Temporarily change the working directory."""
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


def _no_dotenv():
    """Disable load_dotenv so tests don't accidentally pick up a stray
    .env file from the test runner's environment."""
    return patch("alpha_forge.config.load_dotenv", lambda: None)


def _capture_run_repl():
    """Patch run_repl so tests can inspect the Config main() built."""
    captured: dict = {}

    def fake_run_repl(config: Config, **kwargs) -> int:
        captured["config"] = config
        return 0

    return patch("alpha_forge.cli.run_repl", fake_run_repl), captured


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

    def test_parser_exposes_init_config(self) -> None:
        # New flag for bootstrapping the user config file.
        self.assertIn("--init-config", build_parser().format_help())

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

    # --- Layered-merge end-to-end tests (drive main() and inspect Config) ---

    def _write_user_config(self, xdg_root: Path, body: str) -> Path:
        cfg_dir = xdg_root / "alpha-forge"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / "config.toml"
        path.write_text(body)
        return path

    def test_cli_flag_overrides_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            self._write_user_config(
                xdg,
                '[openai]\n'
                'api_key = "file-key"\n'
                'model = "file-model"\n',
            )
            env = {"XDG_CONFIG_HOME": str(xdg), "OPENAI_API_KEY": "env-key"}
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main(["--model", "cli-model"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].api_key, "file-key")
            self.assertEqual(captured["config"].model, "cli-model")

    def test_user_config_overrides_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            self._write_user_config(
                xdg,
                '[openai]\n'
                'api_key = "file-key"\n'
                'model = "file-model"\n',
            )
            env = {
                "XDG_CONFIG_HOME": str(xdg),
                "OPENAI_API_KEY": "env-key",
                "OPENAI_MODEL": "env-model",
            }
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main([])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].api_key, "file-key")
            self.assertEqual(captured["config"].model, "file-model")

    def test_env_overrides_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            env = {
                "XDG_CONFIG_HOME": str(xdg),
                "OPENAI_API_KEY": "env-key",
                "OPENAI_MODEL": "env-model",
            }
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main([])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].api_key, "env-key")
            self.assertEqual(captured["config"].model, "env-model")

    def test_base_url_cli_flag_overrides_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            self._write_user_config(
                xdg,
                '[openai]\n'
                'api_key = "file-key"\n'
                'base_url = "https://file.example/v1"\n',
            )
            env = {"XDG_CONFIG_HOME": str(xdg)}
            with patch.dict(os.environ, env, clear=True):
                with _no_dotenv():
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main(["--base-url", "https://cli.example/v1"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["config"].base_url, "https://cli.example/v1")

    def test_dotenv_still_loaded(self) -> None:
        # Regression guard: load_dotenv() in cli.main must keep working
        # so existing .env-based workflows are preserved.
        # We mock load_dotenv to inject the env var the way a real
        # .env file would, then verify it flows into load_env_config.
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()

            def fake_load_dotenv() -> None:
                os.environ["OPENAI_API_KEY"] = "from-dotenv"

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with patch("alpha_forge.config.load_dotenv", fake_load_dotenv):
                    repl_patch, captured = _capture_run_repl()
                    with repl_patch:
                        exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["config"].api_key, "from-dotenv")

    def test_init_config_writes_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main(["--init-config"])

            target = xdg / "alpha-forge" / "config.toml"
            self.assertEqual(exit_code, 0)
            self.assertTrue(target.exists())
            contents = target.read_text()
            self.assertIn("[openai]", contents)
            self.assertIn("api_key", contents)
            self.assertIn("# model", contents)
            self.assertIn("# base_url", contents)

    def test_init_config_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            target = xdg / "alpha-forge" / "config.toml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("preexisting = 'do-not-overwrite'\n")

            stderr = io.StringIO()
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with contextlib.redirect_stderr(stderr):
                    exit_code = main(["--init-config"])

            self.assertEqual(exit_code, 1)
            self.assertEqual(target.read_text(), "preexisting = 'do-not-overwrite'\n")
            self.assertIn("already exists", stderr.getvalue())

    def test_missing_api_key_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            stderr = io.StringIO()
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with _no_dotenv():
                    with contextlib.redirect_stderr(stderr):
                        exit_code = main([])

            self.assertEqual(exit_code, 2)
            message = stderr.getvalue()
            self.assertIn("api_key", message)
            # The error must point at the XDG config file path so the
            # user can find it without reading the source.
            self.assertIn("config.toml", message)
