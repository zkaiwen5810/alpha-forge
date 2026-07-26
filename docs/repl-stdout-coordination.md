# REPL stdout coordination with `patch_stdout`

A mental model for streaming tokens into a terminal while a live prompt is active.
Captured while debugging the async REPL refactor in `alpha_forge/cli.py`.

## The terminal is one shared grid

The terminal has **one** cursor and **one** grid of cells. There is no z-order,
no layer for "prompt" and another for "response". When the prompt "redraws",
it issues escape sequences that move the cursor and write characters,
overwriting whatever is in those cells.

The prompt owns the bottom row and is *alive*: it redraws itself on every
keystroke or event. If you write to the terminal from a background task
without coordinating, the prompt's next redraw will paint over your text.

## Two writers, one cursor

| writer            | owns                          | redraws when              |
| ----------------- | ----------------------------- | ------------------------- |
| prompt app        | bottom row(s)                 | any keystroke, any event  |
| consumer (LLM)    | nothing — guest               | per token it writes       |

Whoever moves the cursor last wins. With no coordination, the prompt wins
because it is the one constantly redrawing.

## What goes wrong without coordination

- **Erasure** — consumer and prompt target the same row; prompt's redraw
  paints over the token.
- **Truncation** — consumer is mid-line on row N; prompt redraws and moves
  cursor to row M; consumer's next write lands at an unexpected column and
  the first few characters of each line get clobbered by the prompt's
  escape sequence + `"alpha> "` text.
- **Jumbled characters** — writes interleave with prompt redraws in
  unpredictable ways.

Both `print(...)` directly and `print_formatted_text(...)` directly hit
this problem — they write to stdout without telling the prompt to step
aside.

## `patch_stdout` is the handshake

```
consumer → proxy → "prompt, please step aside"
prompt app suspends its renderer
consumer writes buffered text to terminal
prompt app resumes its renderer, redraws at the bottom
```

The direction is: **the consumer asks the prompt to stop redrawing**, not
the other way around. The prompt owns the screen state, so only it knows
how to suspend/resume cleanly. The proxy translates "I want to write"
into `run_in_terminal(write_and_flush)` on the prompt app.

The handshake happens **once per flush**, not per token. That is why
buffering matters.

## Buffering: one flush, not many

`StdoutProxy` (the object `patch_stdout` swaps in for `sys.stdout`)
buffers writes until a newline appears or the `with` block exits, then
schedules **one** `run_in_terminal(suspend → write → resume)` cycle.

| pattern                                | result                                         |
| -------------------------------------- | ---------------------------------------------- |
| `print(piece, end="", flush=False)` + final `print()` | one clean transition, response appears atomically |
| `print(piece, end="", flush=True)` per token         | N transitions; prompt redraws between every chunk and overwrites the leading chars of each new line |
| `print(piece, end="", flush=True)` per token, bracketed by `\0337` / `\0338` | in theory, cursor is restored to end of previous chunk before each write; in practice, terminal support for DECSC/DECRC is inconsistent enough that some terminals display the escape sequences as literal characters and produce visibly jumbled output |

## The two-line summary

> The prompt is alive. If you write to the terminal without telling it to
> step aside, it will paint over you.

> `patch_stdout` is that "step aside" request. Buffer your writes so the
> step-aside happens once at the end, not per token.

## Why direct stdout streaming failed

The naïve way to stream the response is one flush per token — that
gives the user real-time feedback as tokens arrive. But it fails for
two distinct reasons:

1. **Without any cursor management**, the prompt's render loop
   redraws between flushes and moves the terminal cursor back to
   the prompt row. Each new chunk then lands at the prompt row and
   overwrites `alpha> `. The output is truncated/garbled.
2. **Bracketing each chunk with `\0337` / `\0338`** (DECSC / DECRC)
   to save and restore the cursor between flushes *should* fix
   issue 1, but support for these sequences is inconsistent across
   terminal emulators. Some interpret them correctly; others display
   them as literal characters, producing visibly jumbled output.

Before the full-screen UI, the safer fallback was to buffer streamed
chunks in memory and render the response as a single block once the
stream finished. The network call still used the streaming endpoint, so
the producer could keep accepting prompts while the consumer was
mid-response.

## The pattern used in this codebase now

The CLI now avoids this class of bug by running a single full-screen
`prompt_toolkit.application.Application` in `alpha_forge/terminal_ui.py`.
Conversation history, queued inputs, status text, and the prompt input
are all prompt-toolkit controls in one layout. Background streaming
updates mutate in-memory UI state and call `Application.invalidate()`;
they no longer write directly to stdout or move the terminal cursor.

That gives prompt-toolkit one renderer that owns the screen while the app
is running, instead of two writers competing for the same terminal grid.

## Legacy buffered-output pattern

```python
# Drain the stream into an in-memory buffer.
chunks: list[str] = []
async for piece in chat.stream(item):
    chunks.append(piece)
response = "".join(chunks)

with patch_stdout():
    # Write the joined response in one shot. patch_stdout suspends
    # the prompt, writes the buffered text above the live "alpha> "
    # input line, and resumes the prompt.
    print(response)
```

Slash-command output and error messages follow the same "buffer one
flush" rule — they don't need cursor save/restore because they're
written in one shot, not incrementally.
