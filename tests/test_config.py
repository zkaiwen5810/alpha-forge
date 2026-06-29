import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_forge.config import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    Config,
    ConfigError,
    ConfigSource,
    InitConfigAction,
    build_config,
    default_user_config_path,
    load_env_config,
    load_user_config,
    resolve_config,
)


class ConfigTests(unittest.TestCase):
    # --- Config.from_layers (rewrites of the original three tests) ---

    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as ctx:
                Config.from_layers(ConfigSource())
        self.assertIn("api_key", str(ctx.exception))
        # The error should hint at the XDG config path so the user
        # can find the file location without reading the source.
        self.assertIn(".config", str(ctx.exception))

    def test_loads_env_settings(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-test",
            "OPENAI_BASE_URL": "http://localhost:4000/v1",
            "OPENAI_TIMEOUT": "45.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_layers(ConfigSource(), load_env_config())

        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.model, "gpt-test")
        self.assertEqual(config.base_url, "http://localhost:4000/v1")
        self.assertEqual(config.timeout, 45.5)

    def test_default_model(self) -> None:
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True
        ):
            config = Config.from_layers(ConfigSource(), load_env_config())

        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertIsNone(config.base_url)
        self.assertEqual(config.timeout, DEFAULT_TIMEOUT)

    # --- load_user_config ---

    def test_load_user_config_missing_file_returns_empty_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = load_user_config(Path(tmp) / "nope.toml")
        self.assertEqual(result, ConfigSource())

    def test_load_user_config_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("")
            result = load_user_config(path)
        self.assertEqual(result, ConfigSource())

    def test_load_user_config_valid_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '[openai]\n'
                'api_key = "file-key"\n'
                'model = "file-model"\n'
                'base_url = "https://file.example/v1"\n'
                'timeout = 42.5\n'
            )
            result = load_user_config(path)
        self.assertEqual(
            result,
            ConfigSource(
                api_key="file-key",
                model="file-model",
                base_url="https://file.example/v1",
                timeout=42.5,
            ),
        )

    def test_load_user_config_partial_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[openai]\nmodel = "file-model"\n')
            result = load_user_config(path)
        self.assertEqual(result, ConfigSource(model="file-model"))

    def test_load_user_config_ignores_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '[openai]\n'
                'foo = "bar"\n'
                '[other]\n'
                'x = 1\n'
            )
            result = load_user_config(path)
        self.assertEqual(result, ConfigSource())

    def test_load_user_config_malformed_toml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("not = valid = toml")
            with self.assertRaises(ConfigError) as ctx:
                load_user_config(path)
        self.assertIn("not valid TOML", str(ctx.exception))

    def test_load_user_config_wrong_type_for_api_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[openai]\napi_key = 123\n")
            with self.assertRaises(ConfigError) as ctx:
                load_user_config(path)
        self.assertIn("must be a string", str(ctx.exception))

    def test_load_user_config_rejects_invalid_timeout(self) -> None:
        for value in ('"slow"', "0", "-1"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.toml"
                    path.write_text(f"[openai]\ntimeout = {value}\n")
                    with self.assertRaises(ConfigError) as ctx:
                        load_user_config(path)
                self.assertIn("timeout", str(ctx.exception))

    def test_load_user_config_top_level_not_table_raises(self) -> None:
        # Note: Python 3.14's tomllib is strict enough that non-table
        # top-level values are rejected as malformed TOML before our
        # isinstance check runs. This test exercises the
        # "tomllib parsed a non-dict" branch via a hand-built mock so
        # the defense-in-depth check stays covered.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[openai]\napi_key = "sk"\n')
            with patch("alpha_forge.config.tomllib.loads", return_value=42):
                with self.assertRaises(ConfigError) as ctx:
                    load_user_config(path)
        self.assertIn("TOML table at the top level", str(ctx.exception))

    def test_load_user_config_section_not_table_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('openai = "oops"\n')
            with self.assertRaises(ConfigError) as ctx:
                load_user_config(path)
        self.assertIn("must be a TOML table", str(ctx.exception))

    # --- default_user_config_path ---

    def test_default_user_config_path_uses_xdg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}):
                result = default_user_config_path()
        self.assertEqual(result, Path(tmp) / "alpha-forge" / "config.toml")

    def test_default_user_config_path_fallback(self) -> None:
        # Strip XDG_CONFIG_HOME but keep HOME so Path.home() works.
        env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
        with patch.dict(os.environ, env, clear=True):
            result = default_user_config_path()
        self.assertEqual(
            result, Path.home() / ".config" / "alpha-forge" / "config.toml"
        )

    # --- resolve_config (priority matrix) ---

    def test_resolve_config_cli_overrides_user_overrides_env(self) -> None:
        cli = ConfigSource(
            model="cli-model",
            base_url="https://cli/v1",
            timeout=10,
        )
        user = ConfigSource(
            api_key="user-key",
            model="user-model",
            base_url="https://user/v1",
            timeout=20,
        )
        env = ConfigSource(
            api_key="env-key",
            model="env-model",
            base_url="https://env/v1",
            timeout=30,
        )
        config = Config.from_layers(cli, user, env)
        self.assertEqual(config.api_key, "user-key")  # user beats env
        self.assertEqual(config.model, "cli-model")  # cli beats user
        self.assertEqual(config.base_url, "https://cli/v1")  # cli beats user
        self.assertEqual(config.timeout, 10)  # cli beats user

    def test_resolve_config_user_overrides_env(self) -> None:
        user = ConfigSource(api_key="sk", model="user-model")
        env = ConfigSource(api_key="sk", model="env-model")
        config = Config.from_layers(ConfigSource(), user, env)
        self.assertEqual(config.model, "user-model")

    def test_resolve_config_env_overrides_default(self) -> None:
        env = ConfigSource(api_key="sk", model="env-model")
        config = Config.from_layers(ConfigSource(), ConfigSource(), env)
        self.assertEqual(config.model, "env-model")

    def test_resolve_config_raises_without_api_key(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            Config.from_layers(ConfigSource(), ConfigSource(), ConfigSource())
        self.assertIn("api_key", str(ctx.exception))

    def test_resolve_config_fills_default_model(self) -> None:
        config = Config.from_layers(ConfigSource(api_key="sk"))
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.timeout, DEFAULT_TIMEOUT)

    # --- load_env_config ---

    def test_load_env_config_treats_empty_string_as_unset(self) -> None:
        with patch.dict(
            os.environ, {"OPENAI_BASE_URL": ""}, clear=True
        ):
            result = load_env_config()
        self.assertIsNone(result.base_url)

    def test_load_env_config_rejects_invalid_timeout(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_TIMEOUT": "not-a-number"},
            clear=True,
        ):
            with self.assertRaises(ConfigError) as ctx:
                load_env_config()
        self.assertIn("OPENAI_TIMEOUT", str(ctx.exception))

    def test_load_env_config_does_not_call_load_dotenv(self) -> None:
        # Regression guard: load_dotenv must be called by build_config
        # orchestration, not from inside the env layer, so the layer
        # stays pure.
        with patch("dotenv.load_dotenv") as mock_load:
            load_env_config()
        mock_load.assert_not_called()

    # --- build_config: the single public entry point used by cli.main ---

    def test_build_config_orchestrates_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            (xdg / "alpha-forge").mkdir(parents=True)
            (xdg / "alpha-forge" / "config.toml").write_text(
                '[openai]\napi_key = "file-key"\n'
                'model = "file-model"\ntimeout = 20\n'
            )
            args = _ns(model="cli-model", timeout=10)
            with patch.dict(
                os.environ,
                {
                    "XDG_CONFIG_HOME": str(xdg),
                    "OPENAI_API_KEY": "env-key",
                    "OPENAI_MODEL": "env-model",
                    "OPENAI_TIMEOUT": "30",
                },
                clear=True,
            ):
                with patch("alpha_forge.config.load_dotenv", lambda: None):
                    config = build_config(args)

        # CLI > user > env precedence.
        self.assertEqual(config.api_key, "file-key")  # user > env
        self.assertEqual(config.model, "cli-model")  # cli > user
        self.assertEqual(config.timeout, 10)  # cli > user

    def test_build_config_calls_load_dotenv(self) -> None:
        args = _ns()
        with patch.dict(
            os.environ,
            {"XDG_CONFIG_HOME": str(tempfile.gettempdir())},
            clear=True,
        ):
            with patch("alpha_forge.config.load_dotenv") as mock_load:
                with self.assertRaises(ConfigError):
                    # Raises because no api_key in any layer.
                    build_config(args)
        mock_load.assert_called_once()

    def test_build_config_init_writes_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            args = _ns(init_config=True)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(InitConfigAction) as ctx:
                            build_config(args)

            self.assertEqual(ctx.exception.exit_code, 0)
            target = xdg / "alpha-forge" / "config.toml"
            self.assertTrue(target.exists())
            self.assertIn("[openai]", target.read_text())

    def test_build_config_init_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            target = xdg / "alpha-forge" / "config.toml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("preexisting = 'do-not-overwrite'\n")
            args = _ns(init_config=True)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(InitConfigAction) as ctx:
                            build_config(args)

            self.assertEqual(ctx.exception.exit_code, 1)
            self.assertEqual(target.read_text(), "preexisting = 'do-not-overwrite'\n")

    def test_build_config_propagates_config_error(self) -> None:
        # No api_key in any layer -> ConfigError propagates out unchanged.
        args = _ns()
        with patch.dict(
            os.environ,
            {"XDG_CONFIG_HOME": str(tempfile.gettempdir())},
            clear=True,
        ):
            with patch("alpha_forge.config.load_dotenv", lambda: None):
                with self.assertRaises(ConfigError) as ctx:
                    build_config(args)
        self.assertIn("api_key", str(ctx.exception))


def _ns(**attrs):
    """Build a minimal argparse Namespace for build_config tests."""
    from argparse import Namespace

    return Namespace(**attrs)
