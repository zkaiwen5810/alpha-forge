# Transcript architecture

The transcript is the only durable source of completed chat history. It is a
schema-v3 JSONL activity ledger. Every record is appended, flushed, and synced
before it is published in memory.

## Boundaries

- `models.py` contains small completed values shared by layers.
- `transcript.py` defines semantic activities, JSONL storage, and structural
  reference validation. It has no streaming or UI dependency.
- `session.py` is the application’s exclusive transcript writer. It enforces
  model/tool protocol order, selects a branch head, and performs recovery.
- `model_history.py` independently projects one root-to-head branch into
  OpenAI messages.
- `ui_history.py` independently projects the same branch into flat durable UI
  facts.
- `ui_state.py` groups durable UI facts for presentation and owns mutable
  in-flight model/tool display state. It never appends transcript records.
- `events.py` provides ordered, typed, synchronous fan-out.
- `streaming.py` defines normalized model deltas and the authoritative completed
  response emitted by the provider adapter.
- `system_events.py` defines application facts such as transcript changes,
  persistence failures, tool progress, status changes, and exit requests.
- `repl_controller.py` publishes model and system events. Its session consumer
  sends completed activities through `Session`; its UI subscriber is the only
  place that mutates `ChatUiState` and requests a redraw.

Neither projector depends on the other, and there is no shared hierarchical
conversation reconstruction.

## Activities

| Event | Producer | Meaning |
| --- | --- | --- |
| `session.start` | Session creation | Session identity and system prompt |
| `session.transition` | `/clear` or `/resume` | Link to the source session and command |
| `user.message` | User | Complete prompt plus parent turn |
| `user.command` | User | Raw slash-command input and parsed fields |
| `command.result` | Command system | Structured visible command outcome |
| `model.output` | Model | One complete response, including ordered tool calls |
| `tool.result` | Tool system | Complete uncapped raw result |
| `tool.result_limit` | Result policy | Exact reproducible model/UI cap decisions |
| `turn.failure` | Controller | Durable terminal failure |

Transport chunks, response-start markers, progress updates, and commit markers
are not transcript activities.

## Model response flow

1. A complete user prompt is appended and then queued.
2. The controller publishes `ModelResponseStarted`.
3. The provider adapter accumulates normalized deltas while yielding each one
   for the UI's mutable preview.
4. Only clean stream exhaustion emits `StreamCompleted`, carrying one immutable,
   authoritative `ModelResponse`.
5. Ordered fan-out lets the UI render that complete ephemeral response before
   the session consumer calls `Session.add_assistant_message`.
6. A successful add publishes `AssistantMessageAdded`; the UI then resyncs
   durable history and clears its preview.

If persistence fails, `AssistantMessageAddFailed` leaves the authoritative
completed response in the active pane and new prompts/session switches are
blocked. The completed response never flows back from UI state into Session.

## Tool flow

Tool calls are complete only when their containing model stream ends. The
model output is persisted before any requested tool executes. Each completed
raw tool result is appended immediately for crash safety but remains hidden
from durable model/UI projections until the complete ordered batch has a
`ToolResultLimit` event. The active pane renders provisional previews while
tools run.

On resume, an incomplete batch is never rerun. Missing results become synthetic
interruption errors, the batch is capped, and the turn is requeued for the
model.

## Branches

Every `UserMessage` has a `parent_turn_id`. A session selects one head, and both
projectors walk its ancestry from root to head. The current terminal does not
offer a branch selector, but storage and projections do not assume that all
user turns form one linear chain.

Schema v2 files are intentionally rejected; this project does not provide a
compatibility migration during its pre-public stage.
