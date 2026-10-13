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

## A runnable model of the handshake

This standalone model connects the language-level await protocol from chapter 04 to the callback scheduling and completion mechanisms from chapters 05–06. These are teaching classes, not asyncio implementations. Their Future and Task use their own waiting protocol; do not mix them with asyncio objects.

```python
from collections import deque


class Loop:
    def __init__(self):
        self.ready = deque()

    def call_soon(self, callback, *args):
        self.ready.append((callback, args))

    def run(self):
        while self.ready:
            callback, args = self.ready.popleft()
            callback(*args)


class Future:
    def __init__(self, loop):
        self.loop = loop
        self.done = False
        self.value = None
        self.callbacks = []

    def add_done_callback(self, callback):
        if self.done:
            self.loop.call_soon(callback, self)
        else:
            self.callbacks.append(callback)

    def set_result(self, value):
        if self.done:
            raise RuntimeError("Future already completed")

        self.value = value
        self.done = True
        for callback in self.callbacks:
            self.loop.call_soon(callback, self)
        self.callbacks.clear()

    def __await__(self):
        if not self.done:
            yield self  # Travels outward through the coroutine to the Task.
        return self.value


class Task:
    def __init__(self, coroutine, loop):
        self.coroutine = coroutine
        loop.call_soon(self.step)

    def step(self):
        try:
            future = self.coroutine.send(None)
        except StopIteration as finished:
            print("Task finished:", finished.value)
        else:
            future.add_done_callback(self.wakeup)

    def wakeup(self, completed_future):
        self.step()


async def work(future):
    print("Coroutine started")
    value = await future
    print("Coroutine resumed:", value)
    return value * 2


loop = Loop()
future = Future(loop)
task = Task(work(future), loop)


def finish_operation():
    print("Completing Future")
    future.set_result(21)
    print("set_result returned")


loop.call_soon(finish_operation)
loop.run()
```

Output:

```text
Coroutine started
Completing Future
set_result returned
Coroutine resumed: 21
Task finished: 42
```

There are two places where callbacks wait: the Future's callback list holds notifications waiting for completion; the loop's ready queue holds callbacks eligible to execute.

| Moment | Loop's ready queue | Future's callback list |
| --- | --- | --- |
| Before `run()` | `task.step`, `finish_operation` | Empty |
| After `task.step` suspends the coroutine | `finish_operation` | `task.wakeup` |
| After `set_result(21)` | `task.wakeup` | Empty |
| After `task.wakeup` finishes | Empty | Empty |

The loop does not inspect the coroutine to determine whether it can continue. The Future's `set_result()` queues the wake-up callback. It returns before that callback executes, as the output shows.

`self.wakeup` is a **bound method**: it remembers both the method's function and the Task instance to pass as `self`. The Future stores this callable, and the loop later calls it with the completed Future as its argument. Calling `task.wakeup(future)` is equivalent to calling `Task.wakeup(task, future)`.

The wake-up callback is an ordinary method call, not a suspended execution. The previous `step()` call has returned. The Task still holds the coroutine object, which preserves the suspended execution state. A fresh call to `step()` invokes `.send(None)` on that same object. Inside `Future.__await__()`, execution continues after `yield self` and returns the stored value, so `await future` evaluates to `21`.

```text
Future stores bound wake-up callback
    → callback remembers Task
    → Task holds coroutine
    → coroutine holds suspended execution state
```

This model omits exception propagation, cancellation, Task outcome storage, waiting-signal validation, timers, and OS waiting. Its loop exits when the ready queue is empty. A real event loop can wait for external events that make more callbacks ready; chapter 08 traces that I/O path.

## Execution timing

With default scheduling, a new Task starts when the loop runs its callback. Optional eager execution can start it during creation. In either mode, awaiting an immediately completed operation need not give another Task a turn. `asyncio.sleep(0)` explicitly suspends to allow other work.

Keep a reference to created Tasks and await their outcomes. A completed Task can have multiple awaiters; its coroutine still runs only once.

## Check your understanding

**Does every coroutine in an await chain need a separate Task?**

No. One Task can drive many directly nested coroutines. Separate Tasks are needed for independently scheduled execution.

[Reading order](README.md)
