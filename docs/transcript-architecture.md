# Query, session, and transcript architecture

Alpha Forge separates stateless query execution from stateful session
persistence. The transcript is the only durable source of completed chat
history. It is a schema-v4 JSONL write-ahead activity ledger: each record is
appended, flushed, and synced before the query is allowed to continue.

## Boundaries

- `chat.py` adapts OpenAI Chat Completions streams into provider-neutral model
  deltas and one authoritative completed response. It receives explicit
  messages and has no session state.
- `query.py` owns the multi-round model/tool loop. `QueryEngine.run()` receives
  a `QueryRequest` and emits typed events from an async generator. All mutable
  prompt state is local to that generator.
- `tool_execution.py` parses and executes complete tool calls. It returns raw
  results and has no persistence or prompt-editing responsibility.
- `prompt_editor.py` defines the pre-request prompt-editor strategy contract.
  Its default policies synthesize failures for missing tail results, apply the
  versioned tool-result budget, and return persistence effects plus exact
  outgoing messages.
- `transcript.py` defines semantic activities, JSONL storage, and structural
  reference validation. It has no query, tool, or UI dependency.
- `session.py` is the exclusive transcript writer. It enforces protocol order,
  selects a branch head, applies durable query events, and performs recovery.
- `model_history.py` projects one root-to-head branch into complete model
  history or canonical query messages containing an unfinished raw tail.
- `ui_history.py` projects the same branch into immutable presentation facts.
- `repl_controller.py` serializes all user inputs, maps query events to session
  commits, and publishes application events.
- `ui_state.py` is a presentation reducer. It consumes immutable session views
  and ephemeral progress events, applying its own last-20-lines tool-result
  policy; it never references `Session` or `Transcript`.

The query engine imports no session, transcript, command, controller, or UI
types. The terminal UI owns its reducer and subscribes to controller events.

## Durable activities

| Event | Meaning |
| --- | --- |
| `session.start` | Session identity and system prompt |
| `session.transition` | `/clear` or `/resume` link to the source session |
| `user.message` | Complete prompt plus parent turn |
| `user.command` | Raw slash-command input and parsed fields |
| `command.result` | Structured visible command outcome |
| `model.output` | One completed response with ordered tool calls |
| `tool.result` | One complete, uncapped raw application result |
| `tool.result_edit` | Versioned prompt-edit decisions and batch completion |
| `turn.failure` | Durable terminal failure |

Transport deltas, progress updates, queue state, and commit notifications are
ephemeral application events.

## Query and commit flow

1. The FIFO controller dequeues an input and appends its user or command
   activity to the then-current session before performing its effects.
2. For a prompt, the session projects canonical messages, including any
   unfinished assistant/raw-tool tail, while the controller creates a
   query-scoped tool runtime. The transcript reader tool is bound to that
   captured session.
3. At the beginning of each iteration, prompt policies construct the exact
   outgoing messages. Missing calls become ordinary failed raw-result events;
   those events and the final edit are committed before the model client can
   be called.
4. `QueryEngine` streams model deltas for presentation and emits a completed
   model-round event.
5. The controller lets the UI retain the authoritative completed response,
   commits the event through `Session.apply_query_event`, then publishes a new
   immutable `SessionView`.
6. If the response requests tools, the query executes calls sequentially.
   Every raw result is committed before the generator advances, then UI state
   displays its last 20 lines.
7. Raw results remain in application-only tool messages until the following
   iteration applies prompt policies. A response without tool calls completes
   the query.

Because an async generator does not advance until the consumer asks for the
next item, the controller/session boundary is the commit acknowledgement. A
persistence failure closes the query before another provider request or tool
can run. The UI retains the completed response or bounded tool preview and
marks it as unsaved. The controller also stops accepting and processing later
FIFO items because their durable parent state is no longer trustworthy; an
orderly exit remains available.

## Tool results and recovery

The WAL stores the complete raw result once plus compact edit metadata. Model
and UI projections reconstruct the exact bounded representation with the
recorded policy version and limits. Oversized previews include a stable
`transcript_ref`; the session-scoped `tool_result_reader` pages the raw result.

Raw results are appended immediately but remain absent from completed model/UI
history until `tool.result_edit` closes the batch. On resume, an unfinished
batch is never executed again. The session only requeues the turn and projects
its recorded raw results. The query's missing-result policy creates and
persists interruption failures for absent calls, then the model-facing policy
creates and commits the batch edit before continuation.

## FIFO and branches

All submitted prompts and slash commands share one in-memory FIFO. Waiting
items are intentionally not durable. Each item is written to the selected
session only when dequeued, so `/clear` and `/resume` can deterministically
choose the session for later inputs. `/exit` and Ctrl-C stop accepting new
inputs, allow earlier items to finish, then exit.

Every `UserMessage` has a `parent_turn_id`. Both durable projectors walk the
selected root-to-head ancestry. Storage remains branch-capable even though the
terminal currently exposes one selected head.

Schema v3 files are intentionally rejected; this project does not provide a
compatibility migration during its pre-public stage.
