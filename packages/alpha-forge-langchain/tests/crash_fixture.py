"""Leave a durable unpromoted checkpoint by exiting without cleanup."""

import asyncio
import os
import sys
from unittest.mock import patch

from alpha_forge_langchain import AgentApplication
from langchain_core.messages import AIMessage
from test_agent_application import ScriptedModel, aconsume


async def main():
    model = ScriptedModel(
        responses=[AIMessage(content="baseline"), AIMessage(content="unpromoted")]
    )
    async with AgentApplication(sys.argv[1], model=model) as runtime:
        conversation = await runtime.threads.acreate()
        await aconsume(runtime, conversation, "first")
        print(conversation, flush=True)
        with patch.object(
            runtime._persistence._threads,
            "apromote",
            side_effect=lambda *args: os._exit(23),
        ):
            await aconsume(runtime, conversation, "abandoned")


asyncio.run(main())
