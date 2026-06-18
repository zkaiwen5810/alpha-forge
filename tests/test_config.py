import os
import unittest
from unittest.mock import patch

from alpha_forge.config import Config, ConfigError, DEFAULT_MODEL


class ConfigTests(unittest.TestCase):
    def test_requires_openai_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                Config.from_env(load_dotenv_file=False)

    def test_loads_openai_compatible_settings(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-test",
            "OPENAI_BASE_URL": "http://localhost:4000/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env(load_dotenv_file=False)

        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.model, "gpt-test")
        self.assertEqual(config.base_url, "http://localhost:4000/v1")

    def test_default_model(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            config = Config.from_env(load_dotenv_file=False)

        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertIsNone(config.base_url)
