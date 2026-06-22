"""Basic multi-turn chat CLI."""
from __future__ import annotations

import argparse
import asyncio
import sys

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout

from alpha_forge.chat import ChatClient
from alpha_forge.config import (
    Config,
    ConfigError,
    InitConfigAction,
    build_config,
    default_user_config_path,
)
from alpha_forge.conversation import Conversation, Message
from alpha_forge.slash_commands import SLASH_COMMANDS, SlashCommandCompleter, SlashCommandHandler
from alpha_forge.slash_commands.base import CommandContext


DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."


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
        "--init-config",
        action="store_true",
        help=(
            f"write a template config to {default_user_config_path()} and exit"
        ),
    )
    return parser


def run_repl(config: Config, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> int:
    """Sync entry point. Wraps :func:`run_repl_async` with ``asyncio.run``."""
    return asyncio.run(run_repl_async(config, system_prompt=system_prompt))


async def run_repl_async(
    config: Config, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> int:
    """Async event-loop REPL.

    Two coroutines run concurrently:

    - **Producer** (awaited inline): reads prompts via
      ``session.prompt_async``. Slash commands are dispatched inline
      under :func:`patch_stdout`. Real user messages call
      ``conversation.add_user`` and enqueue a snapshot of
      ``conversation.messages`` to the worker queue.
    - **Consumer** (``asyncio.create_task``): drains the queue, drives
      ``chat.stream`` under :func:`patch_stdout`, and appends the
      joined response to the conversation. A ``None`` sentinel stops
      the consumer.

    The snapshot is taken at the producer boundary, so the API call
    for turn N always sees the exact history the user intended for
    turn N, even if a follow-up message has been typed while turn N
    was streaming.

    Streaming output is shown incrementally: each chunk triggers
    a single write under :func:`patch_stdout` that overwrites the
    response line with the accumulated-so-far response. The
    sequence is ``\\033[B`` (move down one row to the response
    area), ``\\r`` (column 0), then the accumulated text. The line
    grows with each chunk, producing the streaming "typing"
    effect. We deliberately avoid ``\\033[K`` (clear to EOL) and
    ``\\0337``/``\\0338`` (DECSC/DECRC) because some terminals
    display those as literal characters; only ``\\033[B``,
    ``\\033[1A``, and ``\\r`` have proven dependable.
    """
    conversation = Conversation(system_prompt=system_prompt)
    chat = ChatClient(config)
    command_handler = SlashCommandHandler()
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(command.name for command in SLASH_COMMANDS),
        multiline=False,
    )

    print_formatted_text("Alpha Forge chat. Type /help for commands.")

    queue: asyncio.Queue[list[Message] | None] = asyncio.Queue()

    async def produce() -> None:
        while True:
            try:
                user_input = await session.prompt_async("alpha> ")
            except (EOFError, KeyboardInterrupt):
                await queue.put(None)
                return

            text = user_input.strip()
            if not text:
                continue

            if text.startswith("/"):
                with patch_stdout():
                    command_result = command_handler.handle(
                        text,
                        CommandContext(
                            config=config,
                            conversation=conversation,
                            chat=chat,
                            print_text=print_formatted_text,
                        ),
                    )
                if command_result.exit_requested:
                    await queue.put(None)
                    return
                if command_result.handled:
                    continue
                with patch_stdout():
                    print_formatted_text(
                        f"unknown command: {text.split(maxsplit=1)[0]}"
                    )
                continue

            conversation.add_user(user_input)
            await queue.put(list(conversation.messages))

    async def consume() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            try:
                # Stream the response by rewriting the line on each
                # chunk. Each chunk triggers a single write under
                # patch_stdout: \033[B moves down one row to the
                # response area, \r returns to column 0, and then
                # the accumulated-so-far response is written. The
                # line is overwritten on each chunk, producing the
                # streaming "growing response" effect. We avoid
                # \033[K (clear to EOL) and DECSC/DECRC because
                # those are not interpreted reliably across
                # terminals — only \033[1A / \r / \033[B have
                # proven dependable.
                chunks: list[str] = []
                async for piece in chat.stream(item):
                    chunks.append(piece)
                    with patch_stdout():
                        print(
                            f"\033[B\r{''.join(chunks)}",
                            end="",
                            flush=True,
                        )

                conversation.add_assistant("".join(chunks))
            except Exception as exc:
                with patch_stdout():
                    print(f"\033[B\rrequest failed: {exc}")
            finally:
                queue.task_done()

    consumer_task = asyncio.create_task(consume())
    await produce()
    await consumer_task
    return 0


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
