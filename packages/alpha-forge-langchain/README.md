# Alpha Forge LangChain

A standalone async streaming terminal agent built with LangChain `create_agent`,
the OpenAI SDK, and a SQLite LangGraph checkpointer. It has no dependency on the
original Alpha Forge runtime or UI. Requires Python 3.14+ and Linux/macOS file
locking.

## Install and run

From the repository root:

```sh
uv sync --project packages/alpha-forge-langchain --locked
uv run --project packages/alpha-forge-langchain alpha-forge-langchain
```

For direct OpenAI access, supply `OPENAI_API_KEY` in your environment and set
`MODEL` to your OpenAI model and `BASE_URL` to `https://api.openai.com/v1` in
`alpha_forge_langchain/defaults.py` before sending input. See the existing
code-defined development settings below. Keys are never accepted on the command
line.

No subcommand starts a new conversation. The full-screen app displays its ID,
completed history, streamed assistant text, tool calls/results, and queue status.
The editor stays available while the agent runs; submitted prompts execute FIFO.
To resume a conversation:

```sh
uv run --project packages/alpha-forge-langchain alpha-forge-langchain chat CONVERSATION_ID
```

- Enter submits nonblank input; Up/Down recall editor history.
- Page Up/Page Down scroll the transcript without changing your draft. New output
  follows the end unless you have scrolled back. Mouse scrolling is also available.
- Ctrl+C cancels active work, drops queued inputs, and exits with code 130.
- Ctrl+D or `/exit` performs the same cleanup and exits with code 0.

Streamed messages are marked pending until the completed checkpoint is promoted.
Failed output stays visible and is marked failed; queued prompts then continue
from completed history. A persistence failure stops the app with a nonzero exit.
Queued prompts and editor drafts are not persisted.

## Headless commands

```sh
uv run --project packages/alpha-forge-langchain alpha-forge-langchain new
uv run --project packages/alpha-forge-langchain alpha-forge-langchain send CONVERSATION_ID "What is (2 + 3) * 4? Use the calculator."
uv run --project packages/alpha-forge-langchain alpha-forge-langchain history CONVERSATION_ID
```

`new` prints a conversation ID. `send` streams assistant text to stdout; partial
output can appear before a failure, and successful completion ends with a newline.
`history` prints a JSON array of completed LangChain messages including tool
exchanges. Errors and restoration diagnostics go to stderr. `new` and `history`
do not require credentials. Interactive commands require terminal stdin/stdout.

The default database is
`$XDG_DATA_HOME/alpha-forge-langchain/conversations.sqlite3`, or
`~/.local/share/alpha-forge-langchain/conversations.sqlite3`. Set `--database PATH`
before the subcommand, or use it alone for a new terminal conversation:

```sh
uv run --project packages/alpha-forge-langchain alpha-forge-langchain --database ./conversations.sqlite3 chat CONVERSATION_ID
```

## Python streaming API

```python
import asyncio
from contextlib import aclosing
from langchain_core.messages import AIMessage
from alpha_forge_langchain import AgentRuntime


async def main():
    async with AgentRuntime("./conversations.sqlite3") as runtime:
        conversation_id = await runtime.acreate_conversation()
        print("Conversation:", conversation_id)
        async with aclosing(
            runtime.asend(conversation_id, "Calculate 13 * 17.")
        ) as stream:
            async for part in stream:
                if part["type"] == "messages":
                    chunk, metadata = part["data"]
                    if isinstance(chunk, AIMessage):
                        print(chunk.text, end="", flush=True)
                elif part["type"] == "updates":
                    # Native node updates contain completed messages/tool results.
                    pass
                elif part["type"] == "values":
                    print("\nSaved:", part["data"]["messages"][-1].text)

    async with AgentRuntime("./conversations.sqlite3") as runtime:
        info = await runtime.arestore_conversation(conversation_id)
        print("Unfinished work excluded:", info.unfinished_work_excluded)
        print(await runtime.ahistory(conversation_id))


asyncio.run(main())
```

`asend()` is an async generator, consumed with `async for`, not `await`. It always
uses LangGraph `astream`, with native v2 stream parts (`type`, `ns`, `data`),
LangChain message objects, metadata, and node update dictionaries. Models without
token streaming emit their completed assistant message when it becomes available.
There is no `send()` compatibility alias or completed-response execution path.

Unlike raw graph `values` streaming, this wrapper emits **one terminal `values`
part only after successful checkpoint promotion**. Its `data` is the full completed
graph state, including all conversation messages, and `interrupts` is empty.
`messages`/`updates` parts are provisional. Errors raise exceptions and emit no
terminal values. Always use `aclosing()` when consuming the generator: breaking
before completion cancels unfinished graph work and releases the operation lock.
Consume or close a provisional stream before calling another runtime operation;
the lock is released before terminal values so history can be inspected there.
Stream payloads and `ahistory()` results are independent copies.

System instructions are stored once in conversation metadata and supplied on each
model request; they are not duplicated in the message history. The logical
`conversation_id` is distinct from LangGraph's execution `thread_id`.

Add tools using LangChain conventions:

```python
from langchain_core.tools import tool


@tool
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


runtime = AgentRuntime("./conversations.sqlite3", tools=[uppercase])
```

Tool names must be unique across all built-ins and supplied tools. Ordinary tool exceptions become error tool
messages so the model can respond. Supply tool handlers again on reopening; Python
handlers are not persisted. Tools execute without approval in this version.

## Built-in tools

File, search, and shell tools use Deep Agents' `FilesystemMiddleware` and
`LocalShellBackend`. Web fetching uses LangChain Community's `RequestsGetTool`.
The calculator remains available alongside these tools in terminal and headless
operation.

| Tool | Behavior |
| --- | --- |
| `calculator` | Bounded numeric arithmetic. |
| `ls` | List entries in a supplied directory. |
| `read_file` | Read files, with `offset` and `limit` for paginated text. |
| `write_file` | Create or overwrite a file; create parent directories as needed. |
| `edit_file` | Replace exact text; require one match unless `replace_all=true`. |
| `glob` | Find paths matching a glob pattern. |
| `grep` | Literal-text search, with path/glob filters and files/content/count output. |
| `execute` | Run a shell command directly in the application's environment. |
| `requests_get` | Fetch a URL with HTTP GET and return its response body as text. |

The workspace is the working directory when `AgentRuntime` is constructed.
File paths can be absolute, relative to that workspace, parent paths, or `~`
paths. `grep` and `glob` search the workspace by default. Regex searches can use
`execute` with an installed command such as `rg`.

Each shell call starts a new system shell in the workspace and inherits the
parent environment. `cd` and `export` changes do not carry over to subsequent
calls. The default timeout is 120 seconds, with an optional per-call `timeout`
up to 3,600 seconds. Output is truncated at the backend's 100,000-character
threshold (upstream labels it as bytes); stderr lines are marked `[stderr]`,
and nonzero exit codes are included in the result. Interpret those exit codes
even when the upstream tool message has success status.

Web fetch returns raw HTML, text, or JSON text, follows redirects, and retains
response bodies for HTTP error statuses. Network exceptions become tool errors.
Its runtime-owned HTTP session has a 30-second timeout and closes with the runtime.
No browser rendering, HTML extraction, or authenticated request configuration is
provided.

For example, send these prompts using the `send` command or terminal editor:

```text
Find Python files under src, then search them for the literal text TODO.
Write notes/example.txt containing hello, replace hello with welcome, and read it back.
Run pwd and report the command output and exit code.
Fetch https://example.com and summarize the returned HTML.
```

Tools act directly on host files and commands without sandboxing or approval.
Cancelling a turn or restoring completed history does not undo tool effects.
An in-flight shell command dispatched to the backend's worker can continue after
stream cancellation; upstream timeout handling does not guarantee cleanup of all
child processes. Restoration does not replay abandoned calls. Automatic message
and tool-result offloading is disabled.

OS-sandboxed shell execution is deferred; its integration, compatibility, resource
lifecycle, and test design are saved in
[the sandbox design document](../../docs/langchain-shell-sandbox-design.md).

Inject a LangChain `BaseChatModel` through `model=...` for custom configuration or
offline testing; injected models remain caller-owned. For example, an injected
`ChatOpenAI` can use its normal OpenAI configuration, including `OPENAI_BASE_URL`
for an optional OpenAI-compatible gateway. LiteLLM is a gateway/proxy layer, not
the application's API surface.

## Existing development settings and recovery

This change retains the code-defined settings in `defaults.py`: model
`DeepSeek-V4-Flash`, development gateway endpoint `http://litellm:4000/v1`,
30-second request timeout, and ten model calls per input. SDK retries are disabled.
The built-in model constructor reads only `OPENAI_API_KEY`; model/endpoint env
vars do not override these code settings. There are no TOML or `.env` loaders or
model/key/endpoint CLI flags in this package. When using the development gateway,
supply its credential through `OPENAI_API_KEY`; outside the development container
its local endpoint is `http://127.0.0.1:14000/v1`. For direct OpenAI, use the
OpenAI settings described above or inject your configured `ChatOpenAI`.

Only successfully completed runs advance the conversation's completed checkpoint.
Restore/history never call a model or execute tools. Failed, cancelled, interrupted,
or explicitly closed streams restore the last completed run. If the first run
never completed, history is empty; subsequent input starts a fresh invocation.

Intermediate checkpoints remain for diagnosis. Even a finished checkpoint is
excluded if a crash prevented promotion. Recovery avoids replaying abandoned tool
calls; it does not undo tool effects or guarantee exactly-once execution.

One runtime owns each database, enforced with an advisory process lock. Runtime
operations are serialized. A SQLite failure halts further operations until closing
and reopening. Use `async with` to release database connections, owned HTTP
clients, and the process lock.

Approvals, sandbox execution, context compaction, automatic checkpoint pruning,
old transcript import, and advanced slash commands are outside this version.

## Tests

```sh
uv run --project packages/alpha-forge-langchain python -m unittest discover -s packages/alpha-forge-langchain/tests -v
```

Tests use fake streaming models, mocked OpenAI SSE transport, temporary SQLite
databases, separate CLI processes, and prompt-toolkit pipe input/dummy output.
No API credentials or model network requests are required.
