"""Layered configuration for the chat CLI.

Configuration is resolved by merging three ``ConfigSource`` layers in
priority order (highest first): CLI flags, the user-level TOML config
file, and environment variables. ``DEFAULT_MODEL`` fills in any
remaining gaps. ``api_key`` is required; missing it raises
``ConfigError`` with a hint pointing at the XDG config path.

The single public entry point used by ``alpha_forge.cli`` is
:func:`build_config`, which handles ``--init-config``, ``.env`` loading,
the user config file, env vars, CLI args, and the layered merge.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from argparse import Namespace


DEFAULT_MODEL = "gpt-4.1-mini"


class ConfigError(Exception):
    """Raised when required configuration is missing or malformed."""


class InitConfigAction(BaseException):
    """Raised by :func:`build_config` after ``--init-config`` has been
    handled. Carries the intended process exit code (0 on success, 1 on
    a file-system error). ``alpha_forge.cli.main`` catches this and
    returns ``exit_code`` instead of starting the REPL."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__(f"init-config completed with exit code {exit_code}")


@dataclass(frozen=True)
class ConfigSource:
    """Per-layer configuration values. Every field is Optional so a
    missing key in this layer falls through to the next-priority layer."""

    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class Config:
    """Resolved configuration consumed by ChatClient."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str | None = None

    @classmethod
    def from_layers(cls, *sources: ConfigSource) -> "Config":
        return resolve_config(*sources)


def default_user_config_path() -> Path:
    """Return the XDG-style user config path. Honors ``$XDG_CONFIG_HOME``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "alpha-forge" / "config.toml"


def build_config(args: "Namespace") -> Config:
    """Resolve all configuration for the CLI in a single call.

    Handles, in order:

    1. ``--init-config``: writes a commented template to the user config
       path and raises :class:`InitConfigAction` so the caller can exit
       without starting the REPL.
    2. ``.env`` loading via :func:`dotenv.load_dotenv`.
    3. The user config file at the XDG path.
    4. ``OPENAI_*`` environment variables.
    5. CLI flags from the argparse ``Namespace``.

    All layers are merged in priority order (CLI > user > env > defaults).
    Returns a fully-resolved :class:`Config`. Raises :class:`ConfigError`
    on any configuration error (missing ``api_key``, malformed user
    config, etc.).
    """
    if getattr(args, "init_config", False):
        raise InitConfigAction(_write_init_config())

    # .env populates os.environ exactly once, before any env reading.
    load_dotenv()

    # Load each layer in priority order (high to low).
    user = load_user_config(default_user_config_path())
    env = load_env_config()
    cli = load_cli_config(args)

    return resolve_config(cli, user, env)


def load_user_config(path: Path) -> ConfigSource:
    """Read the user-level TOML config at ``path``.

    - Missing file: return empty ``ConfigSource`` (file is optional).
    - Permission denied or other ``OSError``: raise ``ConfigError``.
    - Malformed TOML / wrong type: raise ``ConfigError``.
    - Unknown top-level keys: ignored (forward-compat).
    """
    if not path.exists():
        return ConfigSource()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read user config at {path}: {exc}") from exc
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"user config at {path} is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"user config at {path} is not valid TOML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(
            f"user config at {path} must be a TOML table at the top level"
        )
    section = parsed.get("openai")
    if section is None:
        return ConfigSource()
    if not isinstance(section, dict):
        raise ConfigError(
            f"[openai] in {path} must be a TOML table, got {type(section).__name__}"
        )

    def _str(key: str) -> str | None:
        if key not in section:
            return None
        value = section[key]
        if not isinstance(value, str):
            raise ConfigError(
                f"[openai].{key} in {path} must be a string, got {type(value).__name__}"
            )
        return value

    return ConfigSource(
        api_key=_str("api_key"),
        model=_str("model"),
        base_url=_str("base_url"),
    )


def load_env_config() -> ConfigSource:
    """Read ``OPENAI_*`` from ``os.environ``.

    Empty strings become ``None`` so the layer-merge treats them the same
    as unset vars (matches the prior ``os.getenv(...) or None`` behavior).
    """

    def _opt(name: str) -> str | None:
        value = os.environ.get(name)
        return value if value else None

    return ConfigSource(
        api_key=_opt("OPENAI_API_KEY"),
        model=_opt("OPENAI_MODEL"),
        base_url=_opt("OPENAI_BASE_URL"),
    )


def load_cli_config(args: "Namespace") -> ConfigSource:
    """Wrap the argparse ``Namespace``.

    Only flags the user actually passed contribute (the parser uses
    ``argparse.SUPPRESS`` for unset flags, so missing attributes stay
    absent and the layer-merge ignores them).
    """
    return ConfigSource(
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        # api_key intentionally absent — never on the CLI.
    )


def resolve_config(*sources: ConfigSource) -> Config:
    """Merge sources in priority order (highest first).

    Built-in defaults fill any remaining gaps. Raises ``ConfigError`` if
    ``api_key`` is unresolved after all layers.
    """
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    for source in sources:
        if api_key is None:
            api_key = source.api_key
        if model is None:
            model = source.model
        if base_url is None:
            base_url = source.base_url
    if api_key is None:
        raise ConfigError(
            "no api_key found: run `alpha-forge --init-config` to create "
            "a config file, or set OPENAI_API_KEY, or add api_key to "
            f"[openai] in {default_user_config_path()}"
        )
    return Config(
        api_key=api_key,
        model=model if model is not None else DEFAULT_MODEL,
        base_url=base_url,
    )


# --- init-config side effect -------------------------------------------------

_CONFIG_TEMPLATE = """# alpha-forge user config (TOML)
# Path: {path}
# Priority: CLI args > this file > env vars > built-in defaults

[openai]
# api_key = "sk-..."
# model = "gpt-4.1-mini"
# base_url = "https://api.openai.com/v1"
"""


def _write_init_config() -> int:
    """Write a commented template to the XDG user config path.

    Returns ``0`` on success. If the parent directory cannot be created
    or the file already exists, prints an error to stderr and returns
    ``1`` (the caller raises :class:`InitConfigAction` with this code).
    """
    path = default_user_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"alpha-forge: cannot create {path.parent}: {exc}", file=sys.stderr)
        return 1
    if path.exists():
        print(f"alpha-forge: {path} already exists; not overwriting", file=sys.stderr)
        return 1
    path.write_text(_CONFIG_TEMPLATE.format(path=path))
    print(f"alpha-forge: wrote template to {path}")
    return 0
