# 14 — Debugging the Stack

## Diagnose the layer that owns the missing progress

| Symptom | Likely boundary | What to inspect |
| --- | --- | --- |
| Coroutine was never awaited | Construction → execution | Was an object created but never awaited or scheduled? |
| Task exception was never retrieved | Execution → owner | Who awaits the Task or inspects its failure? |
| Every timer and request stalls together | Callback → loop | Is synchronous work blocking the loop thread? |
| One Task waits forever | Task → completion object | What code is supposed to complete its Future, set its Event, or release its lock? |
| Future attached to a different loop | Runtime object → loop | Was a loop-bound object reused across separate runners? |
| Cancellation seems ineffective | Request → delivery | Is work blocking, swallowing cancellation, or running in a worker thread? |
| Growing memory under load | Production → consumption | Are Tasks or queue items created faster than they finish? |

Enable `asyncio.run(main(), debug=True)` during development. Asyncio's debug mode can report slow callbacks and additional misuse; warnings can reveal unawaited coroutines. Task names and stack inspection help identify waiting work. [Asyncio debugging](https://docs.python.org/3.14/library/asyncio-dev.html#debug-mode)

## Inspect a known waiting Task

```python
import asyncio


async def worker(started, release):
    started.set_result(None)
    await release
    return "finished"


async def main():
    loop = asyncio.get_running_loop()
    started = loop.create_future()
    release = loop.create_future()
    task = asyncio.create_task(worker(started, release), name="waiting-worker")
    try:
        await started
        print(task.get_name(), "done:", task.done())
        assert task in asyncio.all_tasks()
        task.print_stack()
    finally:
        if not release.done():
            release.set_result(None)
        assert await task == "finished"


asyncio.run(main(), debug=True)
```

The exact stack output depends on the Python version and filename. The useful fact is that `waiting-worker` is pending at its wait, and this script owns the Future that releases it. Task stack output is a diagnostic view, not necessarily a complete rendering of every nested await frame.

## Trace an incident in both directions

Start at the suspended application code and ask what outcome it needs. Follow that dependency down to the completion producer. Then trace the return path: producer callback, Future completion, wake-up scheduling, Task resumption. If the producer fired but nothing resumed, inspect loop ownership and whether the loop can run its queued callbacks.

## Final understanding check

For `answer = await operation()`, explain these five questions:

1. **What exists after `operation()` is called?** If it is a coroutine function, a new coroutine object.
2. **Who drives it?** Usually the current Task through direct await delegation.
3. **What actually suspends?** An await chain whose underlying iterator yields a runtime-supported signal.
4. **Who arranges another step?** The Task and the completion object's callbacks through the loop.
5. **What reaches `answer`?** The completed operation's return value; an exception instead raises at the await.

The stack is understandable when each state transition has an owner. Syntax describes composition, protocol objects preserve execution, Tasks drive it, Futures connect outcomes to callbacks, and the loop connects callbacks to time and external events.

[Reading order](README.md) · [Stack reference](async-coroutine-principles.md)
