# Transcript, context, and query architecture

Alpha Forge uses one linear schema-v1 transcript as the durable authority for
both provider context and UI history. Completed model-facing contents are not
stored in a second history list. Projectors normalize the ledger independently
for each consumer.

## Package boundaries

- `providers` owns provider-neutral completed output and stream values.
  `OpenAIChatAdapter` is the default adapter and is the only layer that
  translates these values to OpenAI Chat Completions dictionaries.
- `transcript` owns the schema-v1 event catalog, strict codec, protocol replay
  validation, and exclusive JSONL writer.
- `projectors` derives provider context, recovery state, and flat UI facts from
  committed transcript records.
- `context` owns immutable provider-context values and ordered context-edit
  policies. It does not execute tools or call a provider.
- `query` owns the stateless multi-request provider/tool loop. It requests
  context and durable commits through an effect/feedback protocol.
- `tools` owns provider-neutral tool specifications, lookup, and execution.
- `sessions` is the application transcript-write boundary. It exposes
  commands and projections, not a mutable message list.
- `application` owns the FIFO and coordinates session, commands, context,
  query, tools, and reactive presentation events.
- `ui_state` reduces durable session views plus ephemeral streaming progress.

The dependency direction is toward small value and protocol modules. The query
engine has no transcript, session, command, coordinator, or UI reference.

## Record envelope

Every JSONL line is one record:

```json
{
  "schema_version": 1,
  "sequence": 0,
  "event_id": "stable-unique-id",
  "recorded_at": "2026-07-27T12:00:00Z",
  "type": "session.opened",
  "payload": {}
}
```

`sequence` starts at zero and is contiguous. Event IDs are unique. Timestamps
are audit metadata only; protocol order and context visibility never depend on
wall-clock time. This is a completely new schema: records with any schema
version other than 1 are rejected and no legacy migration is attempted.

The writer validates a candidate against replay state, writes one compact line,
flushes it with `fsync`, and only then exposes the new revision to in-process
readers. It holds an exclusive file lock. A stale expected revision, invalid
protocol transition, or write failure cannot publish a partial in-memory
state. An incomplete final JSONL fragment can be removed during resume; a
malformed completed line makes the transcript corrupt.

## Durable event catalog

| Type | Atomic payload and responsibility |
| --- | --- |
| `session.opened` | Session ID and optional instructions; exactly one at sequence 0 |
| `session.linked` | `/clear` or `/resume` link to a source session and command event |
| `input.accepted` | Prompt text, or raw command plus parsed name and arguments |
| `command.completed` | One command status and its ordered visible messages |
| `model.output` | One provider response: ordered output items, finish reason, and usage |
| `tool.result` | One raw result for one call, appended immediately after that call |
| `context.edited` | One policy invocation and an atomic list of declarative operations |
| `query.failed` | Terminal prompt failure with a classified stage and message |

Provider transport chunks, request-start markers, tool-start markers, queue
state, view changes, and commit acknowledgements are ephemeral application
events. They are not required to reconstruct model or UI history.

`model.output` remains atomic because one provider response may request
multiple tools. Its output items use provider-oriented names:

- `OutputMessage`, containing `OutputText` and/or `OutputRefusal`
- `ReasoningItem`
- `ToolCall`, identified by `call_id`

Tool calls are not flattened into separate transcript records. Tool results
are flat because calls execute sequentially and each completed result must be
durable before the next side effect begins. `ToolResult.call_id` supplies the
correlation; UI naming does not leak into the provider value model.

There are no turns, branches, turn IDs, or parent-event chains. A prompt opens
one query in ledger order. Model outputs and tool results extend that open
query until a model output without calls or a `query.failed` event closes it.
This is the smallest ordering model needed for the current sequential app.

## Context edits

A `context.edited` event records:

```text
policy = {name, version, parameters}
operations = [operation, ...]
```

The event exists only when at least one operation changes projected context.
The default policy therefore emits no event when every tool result already
fits its limits.

Schema v1 defines two operation types:

- `SetToolResultRepresentation(result_event_id, representation)` selects
  either the original result or deterministic version-1 head/tail preview
  metadata. Raw tool content is never copied into the edit event.
- `SetToolExchangeVisibility(model_output_event_id, visible)` excludes or
  restores an entire completed intermediate tool exchange—its tool-calling
  model output and correlated results—as one protocol-safe unit.

Operations target durable event IDs, so a future compaction policy can hide an
intermediate provider output from a query days earlier without knowing a turn
number or rewriting historical records. Replay folds later operations over the
same target. An edit cannot target an unknown result/output, repeat a slot in
one event, be a no-op, hide an incomplete exchange, or hide the current query
tail.

`SetToolExchangeVisibility` is intentionally only a schema capability today.
No automatic visibility policy is installed. When one is added, its decision
must use measured context occupation (for example token or context-window
pressure), never record creation time or age.

Policies run serially at provider-request preparation. After a policy returns
operations, the coordinator durably appends them and reprojects before
evaluating the next policy. Consequently each policy sees the committed output
of all earlier policies.

## Query effect and feedback flow

One accepted prompt follows this loop:

1. The query yields `PrepareContext`.
2. The coordinator runs context policies, commits non-noop edits, projects the
   resulting transcript, and sends `ContextPrepared` feedback.
3. The query streams one provider request and emits progress for the UI.
4. The query yields `CommitModelOutput`; it cannot advance until the
   coordinator sends `ModelOutputCommitted` with the assigned event ID and
   committed revision.
5. If the output contains tool calls, calls execute sequentially. Each
   `CommitToolResult` similarly requires `ToolResultCommitted` feedback.
6. The loop returns to context preparation and may request the provider again.
   A response without calls completes the query.

The query engine never appends to a local completed-message list and never
applies a context edit itself. The context snapshot received as feedback is
the exact input to one provider request and is discarded before the next
iteration. This prevents transcript projection and a query-local copy from
diverging.

## Recovery and missing tool results

Resume first opens, validates, and projects the transcript. Semantic recovery
does not execute tools or synthesize events merely because the file was
opened. The storage layer may still truncate an incomplete final JSONL
fragment left by a torn write.

If replay finds an open prompt whose latest model output has missing results,
the coordinator schedules query continuation ahead of queued user input. At
the start of that continuation, the query yields one `CommitToolResult` with
status `interrupted` for each absent call in provider order. Recorded results
remain untouched. Only after every missing result is durable does normal
context preparation run.

This boundary is deliberate: a tool side effect might have happened before a
crash even though its result did not reach the WAL. Re-executing it is unsafe,
while mutating the log merely by opening it makes inspection and validation
surprising. Continuation preparation is the first point that both needs a
provider-valid exchange and has explicit authority to write.

## UI projection and responsiveness

Input submission only enqueues and publishes queue state. The single consumer
dequeues in FIFO order, appends `input.accepted`, then handles a command or
query. Every durable append is followed by a fresh immutable `SessionView`.
Provider deltas and running-tool state remain UI-owned ephemeral values.

The UI projector always renders raw transcript content, including exchanges
excluded from model context. The model projector applies representations and
exchange visibility. Thus UI display and provider context share one source of
truth while retaining consumer-specific filtering.
