"""Streaming terminal chat and headless conversation commands."""

import argparse
import asyncio
import json
import sys
from contextlib import aclosing

from langchain_core.messages import AIMessage

from .runtime import AgentRuntime
from .session import ChatSession, MessageState
from .terminal import ChatApplication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-forge-langchain")
    parser.add_argument("--database", help="SQLite database path")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("new", help="create a conversation and print its ID")
    chat = commands.add_parser("chat", help="open terminal chat (default)")
    chat.add_argument("conversation_id", nargs="?", help="resume a saved conversation")
    send = commands.add_parser("send", help="stream one prompt")
    send.add_argument("conversation_id")
    send.add_argument("input", metavar="PROMPT")
    history = commands.add_parser("history", help="print completed history as JSON")
    history.add_argument("conversation_id")
    return parser


async def arun(args: argparse.Namespace) -> int:
    interactive = args.command in (None, "chat")
    if interactive and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError(
            "chat requires a terminal; use new, send, or history for headless operation"
        )
    async with AgentRuntime(args.database) as runtime:
        if args.command == "new":
            print(await runtime.acreate_conversation())
            return 0
        conversation_id = getattr(args, "conversation_id", None)
        if conversation_id is None:
            conversation_id = await runtime.acreate_conversation()
        info = await runtime.arestore_conversation(conversation_id)
        diagnostic = "Unfinished work excluded; restored the last completed turn."
        if interactive:
            session = ChatSession(
                runtime, conversation_id, await runtime.ahistory(conversation_id)
            )
            if info.unfinished_work_excluded:
                session.notices.append(diagnostic)
            return await ChatApplication(session).arun()
        if info.unfinished_work_excluded:
            print(diagnostic, file=sys.stderr)
        if args.command == "history":
            messages = await runtime.ahistory(conversation_id)
            print(
                json.dumps(
                    [m.model_dump(mode="json") for m in messages], ensure_ascii=False
                )
            )
        else:
            state = MessageState()
            emitted: dict[str | None, str] = {}
            async with aclosing(runtime.asend(conversation_id, args.input)) as stream:
                async for part in stream:
                    state.update(part)
                    for message in state.messages:
                        if not isinstance(message, AIMessage):
                            continue
                        previous = emitted.get(message.id, "")
                        text = message.text
                        if text.startswith(previous):
                            print(text[len(previous) :], end="", flush=True)
                        emitted[message.id] = text
            print()
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(arun(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"alpha-forge-langchain: {exc}", file=sys.stderr)
        return 1
