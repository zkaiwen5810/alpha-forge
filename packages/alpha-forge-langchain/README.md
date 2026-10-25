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
uv run --project packages/alpha-forge-langchain alpha-forge-langchain chat THREAD_ID
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
uv run --project packages/alpha-forge-langchain alpha-forge-langchain send THREAD_ID "What is (2 + 3) * 4? Use the calculator."
uv run --project packages/alpha-forge-langchain alpha-forge-langchain history THREAD_ID
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
uv run --project packages/alpha-forge-langchain alpha-forge-langchain --database ./conversations.sqlite3 chat THREAD_ID
```

## Python streaming API

`AgentApplication` composes three separate components: `query` executes turns,
`agents` builds native LangChain agents, and `threads` manages thread records and
history. The checkpointer is wired privately at application construction.

For direct OpenAI access, set `OPENAI_API_KEY` and choose your model through
`OPENAI_MODEL` in this example:

```python
import asyncio
import os
from contextlib import aclosing

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from alpha_forge_langchain import AgentApplication


async def main():
    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])
    async with AgentApplication("./threads.sqlite3", model=model) as app:
        first = await app.threads.acreate()
        second = await app.threads.acreate()

        async def run_turn(thread_id, text, prompt):
            # Build/select an agent for this turn, not for the thread's lifetime.
            agent = await app.agents.acreate(system_prompt=prompt)
            async with aclosing(app.query.asubmit(
                thread_id,
                agent=agent,
                input={"messages": [HumanMessage(content=text)]},
            )) as stream:
                async for part in stream:
                    if part["type"] == "messages":
                        chunk, metadata = part["data"]
                        if isinstance(chunk, AIMessage):
                            print(thread_id, chunk.text, flush=True)
                    elif part["type"] == "values":
                        print(thread_id, "Saved:", part["data"]["messages"][-1].text)
            return agent

        # The caller owns tasks. Exit them before closing application resources.
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(run_turn(first, "Calculate 13 * 17.", "Be concise."))
            tasks.create_task(run_turn(second, "Explain multiplication.", "Be a tutor."))

        # Next turn can change the prompt, tools, middleware, or model.
        agent = await run_turn(first, "Explain your answer.", "Explain step by step.")
        print(await app.query.aget_state(first, agent=agent))

    # Inspection does not require a model or credentials.
    async with AgentApplication("./threads.sqlite3") as app:
        info = await app.threads.aget(first)
        print("Unfinished work excluded:", info.unfinished_work_excluded)
        print(await app.threads.ahistory(first))


asyncio.run(main())
```

`QueryEngine.asubmit()` is an async generator: consume it with `async for`, not
`await`. Calling it alone starts no work. Pass a coroutine that consumes the
generator to `asyncio.create_task` or a task group. A `TerminalChatController`
consumes FIFO input in its caller-owned task; separate terminal controllers can
share one query engine.

Each submission calls its supplied agent's native `astream` exactly once. Native
v2 message/update parts retain LangChain objects and metadata. Models without
token streaming emit completed messages when available. Supply only new input;
LangGraph restores prior messages and compatible agent state from the committed
checkpoint. `thread_id` is the same stable identifier used by LangGraph.

Different threads progress concurrently. Same-thread submissions serialize from
checkpoint selection through promotion. `aget_state(thread_id, agent=agent)`
uses the agent's native state inspection API at the committed checkpoint, so its
schema projects state without internal routing channels. It does not invoke the
model. `threads.ahistory(thread_id)` reads messages directly from that checkpoint
and needs no agent or credentials. Both return committed data while a turn is
responding. Stream and state results are independent copies. QueryEngine owns no
input queues, worker tasks, event broker, or public checkpointer.

The generator emits **one terminal `values` part after successful checkpoint
promotion**, with full public graph state and empty `interrupts`. Message/update
parts are provisional; ordinary failures emit no terminal values. Always consume
inside `aclosing()`: early closure or task cancellation cleans up unfinished graph
work and releases the thread lock. That lock is released before terminal values,
so another turn or history read can start there. Once durable promotion starts,
its outcome is settled before cancellation propagates; an already committed turn
is not rolled back. There is no detached agent execution after stream closure.

The application agent factory accepts per-turn `model`, `tools`, `system_prompt`,
`middleware`, `state_schema`, `context_schema`, and `response_format` arguments.
Pass per-turn runnable configuration and runtime `context` to `asubmit`. Thread
and checkpoint controls remain engine-owned. Agents must be built against this
application's persistence backend; agents retained from a closed application
must be rebuilt after reopening.

Default system instructions belong to the current agent. To persist a thread
prompt, include `system_prompt_override` in native input along with the new
messages. Omitting it preserves the saved override; setting it to `None` clears
it and restores the current agent's default. The override is applied by native
model middleware before filesystem instructions are appended. Failed turns do
not commit prompt changes. Custom state must obey LangGraph's compatibility rules;
there is no automatic migration for incompatible schemas.

Add tools using LangChain conventions:

```python
from langchain_core.tools import tool


@tool
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


# Inside the application context:
agent = await app.agents.acreate(tools=[uppercase])
```

Tool names must be unique across built-ins and supplied tools. Ordinary tool
exceptions become error tool messages so the model can respond. Supply Python
handlers again on reopening; handlers are not persisted. Tools execute without
approval in this version.

## Transcript and model-facing context

Keep one original transcript and derive a separate view for each model call:

| View | Source | Persistence |
| --- | --- | --- |
| Transcript | Original `AgentState.messages`, including tool arguments and outputs | Native committed checkpoint |
| Model-facing context | Edited copy of `ModelRequest.messages` | Derived for each model call |

`threads.ahistory()`, terminal rendering, and the terminal `values` part use the
original interactions. Context edits affect model input only. The durable
transcript includes completed turns; failed and cancelled work remains excluded
by the existing completed-checkpoint recovery policy.

Every factory-built agent automatically compacts model input using
`MAX_CONTEXT_TOKENS = 64_000` in `alpha_forge_langchain/defaults.py`. Before each
model call, it estimates tokens for messages, system instructions, and tool
definitions, then drops oldest complete turns until the request fits. The current
user turn and all its tool exchanges remain intact. If those alone exceed the
budget, the turn fails with a descriptive error before calling the model. Counting
is approximate, and the budget does not reserve output tokens. New agents read
the current setting, so it can change between turns.

Additional context editing is available through the agent middleware argument,
and runs before default compaction. For example, LangChain's native tool-output
clearing can keep the latest three tool results above a token threshold:

```python
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
)


agent = await app.agents.acreate(
    middleware=[
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=40_000, keep=3)]
        )
    ]
)
```

LangChain's middleware deep-copies request messages before applying its edits and
passes them to the model using `request.override(messages=edited_messages)`.
Original tool outputs remain available in the transcript, including after
reopening the application. Rebuilding an agent with a different policy derives a
new view from those originals; the previous edited view is never saved as history.

Custom asynchronous policies use `awrap_model_call` or an async
`@wrap_model_call` function. Deep-copy `request.messages`, apply policies to that
copy, and await the handler with the overridden request. Preserve tool-call IDs
and matching tool results when trimming or filtering. Inspect the actual model
input through native model-call callbacks; checkpoint inspection returns the
transcript rather than a particular request's context.

Context-only policies must not return updates to `state["messages"]`, mutate its
nested message objects, or add synthetic context messages to transcript/UI
events. Built-in `SummarizationMiddleware` and state-mutating trimming examples
replace persisted messages, so they cannot be used unchanged for this purpose.
Future summarization can inject request-only summary messages and cache policy
metadata in other agent-state fields while preserving the original transcript.
Auxiliary policy model responses must also stay outside transcript/UI events.

This separation needs no additional history table, duplicated state blob, or
query-engine API. It reduces model input size; full transcript/checkpoint storage
continues to grow.

References: [LangChain context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering),
[native context editing](https://docs.langchain.com/oss/python/langchain/middleware/built-in#context-editing).

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

The workspace is the working directory when `AgentApplication` is constructed.
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
Its factory-owned HTTP session has a 30-second timeout and closes with the application.
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

One application owns each database, enforced with an advisory process lock.
Queries serialize only within the same thread. Metadata transactions use a
separate SQLite connection from native checkpoints. The thread table stores
only the thread ID, committed checkpoint pointer, and unfinished marker; graph
state is stored exclusively by LangGraph. Promotion atomically advances the
pointer and clears the marker. A SQLite failure halts further operations until
closing and reopening. Finish caller-owned tasks and streams before leaving the
application context to release database connections, owned clients, and the
process lock.

This private API and thread metadata schema replace `AgentRuntime`. Databases
with the previous `threads.state_type` and `threads.state` payload columns are
upgraded automatically at startup. Thread IDs, committed checkpoint pointers,
unfinished markers, and native checkpoints are preserved. Older `conversations`
records from before the thread-based API are not imported.

Approvals, sandbox execution, automatic checkpoint pruning,
old transcript import, and advanced slash commands are outside this version.

## Tests

```sh
uv run --project packages/alpha-forge-langchain python -m unittest discover -s packages/alpha-forge-langchain/tests -v
```

Tests use fake streaming models, mocked OpenAI SSE transport, temporary SQLite
databases, separate CLI processes, and prompt-toolkit pipe input/dummy output.
No API credentials or model network requests are required.
