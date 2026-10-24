"""Streaming terminal chat and headless conversation commands."""

import argparse
import asyncio
import json
import sys
from contextlib import aclosing

from langchain_core.messages import AIMessage, HumanMessage

from .application import AgentApplication
from .message_state import MessageState
from .terminal import ChatApplication
from .terminal_chat import TerminalChatController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-forge-langchain")
    parser.add_argument("--database", help="SQLite database path")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("new", help="create a conversation and print its ID")
    chat = commands.add_parser("chat", help="open terminal chat (default)")
    chat.add_argument("thread_id", nargs="?", help="resume a saved thread")
    send = commands.add_parser("send", help="stream one prompt")
    send.add_argument("thread_id")
    send.add_argument("input", metavar="PROMPT")
    history = commands.add_parser("history", help="print completed history as JSON")
    history.add_argument("thread_id")
    return parser


async def arun(args: argparse.Namespace) -> int:
    interactive = args.command in (None, "chat")
    if interactive and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError(
            "chat requires a terminal; use new, send, or history for headless operation"
        )
    async with AgentApplication(args.database) as application:
        if args.command == "new":
            print(await application.threads.acreate())
            return 0
        thread_id = getattr(args, "thread_id", None)
        if thread_id is None:
            thread_id = await application.threads.acreate()
        info = await application.threads.aget(thread_id)
        diagnostic = "Unfinished work excluded; restored the last completed turn."
        if interactive:
            controller = TerminalChatController(
                application.query,
                thread_id,
                await application.threads.ahistory(thread_id),
                agent_factory=application.agents.acreate,
            )
            if info.unfinished_work_excluded:
                controller.notices.append(diagnostic)
            return await ChatApplication(controller).arun()
        if info.unfinished_work_excluded:
            print(diagnostic, file=sys.stderr)
        if args.command == "history":
            messages = await application.threads.ahistory(thread_id)
            print(
                json.dumps(
                    [m.model_dump(mode="json") for m in messages], ensure_ascii=False
                )
            )
        else:
            state = MessageState()
            emitted: dict[str | None, str] = {}
            if not args.input.strip():
                raise ValueError("input must be a non-empty string")
            agent = await application.agents.acreate()
            async with aclosing(
                application.query.asubmit(
                    thread_id,
                    agent=agent,
                    input={"messages": [HumanMessage(content=args.input)]},
                )
            ) as stream:
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
