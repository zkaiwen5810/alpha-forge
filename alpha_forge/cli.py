"""Alpha Forge chat CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys

from alpha_forge.application.coordinator import ApplicationCoordinator
from alpha_forge.config import (
    Config,
    ConfigError,
    InitConfigAction,
    build_config,
    default_user_config_path,
)
from alpha_forge.sessions import DEFAULT_SYSTEM_PROMPT
from alpha_forge.terminal_ui import TerminalChatUi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-forge")
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="Override model for this run",
    )
    parser.add_argument(
        "--base-url",
        default=argparse.SUPPRESS,
        help="Override OPENAI_BASE_URL for this run",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=argparse.SUPPRESS,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help=(f"write a template config to {default_user_config_path()} and exit"),
    )
    return parser


def run_repl(config: Config, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> int:
    """Sync entry point. Wraps :func:`run_repl_async` with ``asyncio.run``."""
    return asyncio.run(run_repl_async(config, system_prompt=system_prompt))


async def run_repl_async(
    config: Config, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> int:
    """Run the full-screen prompt-toolkit chat UI."""
    controller = ApplicationCoordinator(config, system_prompt=system_prompt)
    ui = TerminalChatUi(controller)
    consumer_task = asyncio.create_task(controller.consume())
    try:
        return await ui.run_async()
    finally:
        controller.request_exit()
        await consumer_task


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(args)
    except InitConfigAction as action:
        return action.exit_code
    except ConfigError as exc:
        print(f"alpha-forge: {exc}", file=sys.stderr)
        return 2
    return run_repl(config)


if __name__ == "__main__":
    raise SystemExit(main())
