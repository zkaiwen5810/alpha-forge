# 07 — Tasks and Scheduling

## A coroutine driver with an outcome handle

An `asyncio.Task` wraps a coroutine, drives it, and retains its result, exception, or cancelled status. It is a kind of Future, but callers cannot use `set_result()` or `set_exception()` to supply its outcome: the wrapped execution determines it.

The coroutine owns its execution state. The Task owns the scheduling lifecycle, including its current wait and cancellation requests. Use `asyncio.create_task(coro)` on a running loop to give a fresh coroutine independent scheduling. [Task API](https://docs.python.org/3.14/library/asyncio-task.html#task-object)

```python
import asyncio


async def child():
    await asyncio.sleep(0)
    return asyncio.current_task()


async def main():
    parent_task = asyncio.current_task()
    assert await child() is parent_task

    child_task = asyncio.create_task(child())
    assert await child_task is child_task
    assert await child_task is child_task  # Stored outcome, no rerun.
    print("direct await shares a Task; create_task creates another")


asyncio.run(main())
```

Both calls to `child()` produce new coroutine objects. The first extends the parent's execution chain. The second gets its own Task. Never give the same coroutine object to two Tasks.

## The three-party handshake

| Sender → receiver | Message |
| --- | --- |
| Task → loop | Schedule an execution callback |
| Task → coroutine | Resume with `.send(None)` or inject an exception |
| Coroutine chain → Task | Yield a waiting signal, finish, or raise |
| Task → pending Future | Register a wake-up callback |
| Future → loop | Schedule completion callbacks |
| Loop → Task | Run the wake-up callback |

On a normal Future completion, the Task resumes with `.send(None)` and the Future's await iterator retrieves the stored result. On failure it can inject the exception with `.throw()`. A returned coroutine value completes the Task through the protocol's `StopIteration` signal.

CPython recognizes correctly yielded asyncio Futures and a bare `None` yield, which requests another scheduled step. Unsupported signals fail. A pending Future from another loop is rejected. The private handshake is distinct from the public language-level awaitable contract. [Task implementation](https://github.com/python/cpython/blob/3.14/Lib/asyncio/tasks.py)

## Execution timing

With default scheduling, a new Task starts when the loop runs its callback. Optional eager execution can start it during creation. In either mode, awaiting an immediately completed operation need not give another Task a turn. `asyncio.sleep(0)` explicitly suspends to allow other work.

Keep a reference to created Tasks and await their outcomes. A completed Task can have multiple awaiters; its coroutine still runs only once.

## Check your understanding

**Does every coroutine in an await chain need a separate Task?**

No. One Task can drive many directly nested coroutines. Separate Tasks are needed for independently scheduled execution.

[Reading order](README.md)
