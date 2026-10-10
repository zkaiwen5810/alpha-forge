"""Leave a durable unpromoted checkpoint by exiting without cleanup."""

import asyncio
import os
import sys
from unittest.mock import patch

from alpha_forge_langchain import AgentRuntime
from langchain_core.messages import AIMessage
from test_runtime import ScriptedModel


async def main():
    model = ScriptedModel(responses=[AIMessage(content="baseline"), AIMessage(content="unpromoted")])
    async with AgentRuntime(sys.argv[1], model=model) as runtime:
        conversation = await runtime.create_conversation()
        await runtime.send(conversation, "first")
        print(conversation, flush=True)
        with patch.object(runtime._store, "promote", side_effect=lambda *args: os._exit(23)):
            await runtime.send(conversation, "abandoned")


asyncio.run(main())
