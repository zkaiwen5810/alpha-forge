"""One operation per invocation; no configuration resolver or REPL."""

import argparse
import asyncio
import json
import sys

from .runtime import AgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-forge-langchain")
    parser.add_argument("--database", help="SQLite database path")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("new", help="create a conversation and print its ID")
    send = commands.add_parser("send", help="send one prompt")
    send.add_argument("conversation_id")
    send.add_argument("prompt")
    history = commands.add_parser("history", help="print completed history as JSON")
    history.add_argument("conversation_id")
    return parser


async def run(args: argparse.Namespace) -> None:
    async with AgentRuntime(args.database) as runtime:
        if args.command == "new":
            print(await runtime.create_conversation())
            return
        info = await runtime.restore_conversation(args.conversation_id)
        if info.unfinished_work_excluded:
            print("Unfinished work excluded; restored the last completed turn.", file=sys.stderr)
        if args.command == "history":
            messages = await runtime.history(args.conversation_id)
            print(json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False))
        else:
            result = await runtime.send(args.conversation_id, args.prompt)
            print(result.assistant_text)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"alpha-forge-langchain: {exc}", file=sys.stderr)
        return 1
    return 0
