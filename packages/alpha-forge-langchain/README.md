# Alpha Forge LangChain

A standalone implementation built with LangChain `create_agent` and a SQLite
LangGraph checkpointer. It has no dependency on the original Alpha Forge runtime.
Requires Python 3.14+ and Linux/macOS file locking.

## Install and run

From the repository root:

```sh
uv sync --project packages/alpha-forge-langchain --locked
uv run --project packages/alpha-forge-langchain alpha-forge-langchain new
```

Copy the printed conversation ID, set `OPENAI_API_KEY` to the LiteLLM proxy key
in your environment, then:

```sh
uv run --project packages/alpha-forge-langchain alpha-forge-langchain send CONVERSATION_ID "What is (2 + 3) * 4? Use the calculator."
uv run --project packages/alpha-forge-langchain alpha-forge-langchain send CONVERSATION_ID "What number did we calculate?"
uv run --project packages/alpha-forge-langchain alpha-forge-langchain history CONVERSATION_ID
```

`new` and `history` do not require credentials. `send` prints the final assistant
text; `history` prints a JSON array of LangChain messages, including tool exchanges.
Errors and restoration diagnostics go to stderr and failures return nonzero.

The default database is
`$XDG_DATA_HOME/alpha-forge-langchain/conversations.sqlite3`, or
`~/.local/share/alpha-forge-langchain/conversations.sqlite3`. Override it with
`--database PATH` **before the subcommand**:

```sh
uv run --project packages/alpha-forge-langchain alpha-forge-langchain --database ./conversations.sqlite3 new
```

## Python API

```python
import asyncio
from alpha_forge_langchain import AgentRuntime

async def main():
    async with AgentRuntime("./conversations.sqlite3") as runtime:
        conversation_id = await runtime.create_conversation()
        result = await runtime.send(conversation_id, "Calculate 13 * 17.")
        print(conversation_id, result.assistant_text)
        await runtime.send(conversation_id, "Repeat the answer.")

    # A new runtime (or another process) can restore the conversation.
    async with AgentRuntime("./conversations.sqlite3") as runtime:
        info = await runtime.restore_conversation(conversation_id)
        print("Unfinished work excluded:", info.unfinished_work_excluded)
        print(await runtime.history(conversation_id))
        print((await runtime.send(conversation_id, "Continue our conversation.")).assistant_text)

asyncio.run(main())
```

`TurnResult.messages` contains only the new turn's user, model, and tool messages.
`history()` returns the full completed history. Both return independent copies.
System instructions are stored once in conversation metadata and supplied on each
model request; they are not duplicated in the message history.

The calculator supports numeric literals and basic arithmetic with bounded
expression size, exponent magnitude, and results. Add tools through the API:

```python
from langchain_core.tools import tool

@tool
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

# Supply tools again when reopening the runtime; Python handlers are not persisted.
runtime = AgentRuntime("./conversations.sqlite3", tools=[uppercase])
```

Tool names must be unique; `calculator` is reserved by the built-in tool. Ordinary
tool exceptions become error tool messages so the model can respond. Inject a
`BaseChatModel` with `model=...` for offline use; injected models stay caller-owned.
Caller-supplied tools run without approval in this headless milestone. Side-effecting
tools and human-in-the-loop workflows are outside this version's intended scope.

## Defaults and restoration

Model settings are defined directly in `alpha_forge_langchain/defaults.py`:
`DeepSeek-V4-Flash`, the LiteLLM endpoint `http://litellm:4000/v1` inside the
development container, 30-second request timeout, and ten
model calls per prompt. SDK retries are disabled. There are no TOML or `.env`
loaders and no model/key/endpoint CLI flags. Only `OPENAI_API_KEY` supplies the
credential; endpoint/model environment settings do not override code defaults.
The application uses OpenAI-compatible requests through the LiteLLM gateway.
For direct OpenAI access, set the code-defined endpoint to
`https://api.openai.com/v1`, select an OpenAI model, and supply an OpenAI API key.
From outside the development container, the local proxy is exposed at
`http://127.0.0.1:14000/v1`. Endpoint changes remain code-defined in this milestone.

Only successfully completed turns advance the conversation's completed checkpoint.
Restore/history operations never call a model or execute tools. After a failed,
cancelled, or interrupted turn, restore selects the last completed turn and reports
that unfinished work was excluded. If the first turn never completed, history is
empty. The next prompt starts a fresh invocation from that completed history.

Intermediate checkpoints remain stored for diagnosis but are not exposed as
conversation history. Even a finished checkpoint is excluded if a crash prevented
its promotion to the completed head. This deliberately avoids replaying abandoned
tool calls; it does not undo tool effects or guarantee exactly-once execution.

Only one runtime may own a database at a time, enforced with an advisory process
lock. Operations within that runtime are serialized. A SQLite failure halts further
operations until the runtime is closed and reopened. Use `async with` to release
database connections, HTTP clients, and the process lock.

This milestone returns completed responses only. It does not include a terminal
REPL, streaming, approvals, filesystem/Bash tools, context compaction, automatic
checkpoint pruning, old transcript import, or queued prompts.

## Tests

```sh
uv run --project packages/alpha-forge-langchain python -m unittest discover -s packages/alpha-forge-langchain/tests -v
```

Tests use fake models, a mocked OpenAI transport, temporary SQLite databases, and
separate CLI processes. No API credentials or network requests are required.
