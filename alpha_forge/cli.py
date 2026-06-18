"""Basic multi-turn chat CLI."""

from __future__ import annotations

import argparse

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from alpha_forge.chat import ChatClient
from alpha_forge.config import Config, ConfigError
from alpha_forge.conversation import Conversation


DEFAULT_SYSTEM_PROMPT = "You are Alpha Forge, a concise and helpful assistant."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-forge")
    parser.add_argument("--model", help="Override OPENAI_MODEL for this run")
    parser.add_argument("--base-url", help="Override OPENAI_BASE_URL for this run")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    return parser


def run_repl(config: Config, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> int:
    conversation = Conversation(system_prompt=system_prompt)
    chat = ChatClient(config)
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(["/clear", "/exit", "/help", "/model", "/quit"], ignore_case=True),
        multiline=False,
    )

    print_formatted_text("Alpha Forge chat. Type /help for commands.")
    while True:
        try:
            user_input = session.prompt("alpha> ")
        except (EOFError, KeyboardInterrupt):
            print_formatted_text("")
            return 0

        text = user_input.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/help":
            print_formatted_text("/help /model /clear /exit")
            continue
        if text == "/model":
            print_formatted_text(config.model)
            continue
        if text == "/clear":
            conversation.clear()
            print_formatted_text("conversation cleared")
            continue

        conversation.add_user(user_input)
        try:
            answer = chat.complete(conversation.messages)
        except Exception as exc:
            print_formatted_text(f"request failed: {exc}")
            continue

        conversation.add_assistant(answer)
        print_formatted_text(answer)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
    except ConfigError as exc:
        parser.error(str(exc))

    if args.model or args.base_url:
        config = Config(
            api_key=config.api_key,
            model=args.model or config.model,
            base_url=args.base_url or config.base_url,
        )

    return run_repl(config, system_prompt=args.system)


if __name__ == "__main__":
    raise SystemExit(main())
