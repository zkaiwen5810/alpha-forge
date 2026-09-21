# Python Async Stack — Reference Map

For the complete beginner course, start with the [reading guide](README.md). This page summarizes ownership and contracts across the stack.

## Owners and exposed contracts

A **protocol** is an agreement about methods, inputs, outputs, and behavior. A **runtime** supplies the machinery that drives execution using those agreements.

| Layer | Building block | Exposed contract | Owns |
| --- | --- | --- | --- |
| Python language | Generator | `next`, `send`, `throw`, `close`; yields and final return | Resumable execution state |
| Python language | Native coroutine | `send`, `throw`, `close`; usable with `await` | Locals, execution position, await chain |
| Python language | Custom awaitable | `__await__()` returns an iterator | Delegation to an underlying operation |
| Python language | Async iterator | `__aiter__`, awaitable `__anext__`, `StopAsyncIteration` | Repeated asynchronous item production |
| Python language | Async context manager | Awaitable `__aenter__` and `__aexit__` | Entry and exit behavior for a scope |
| `asyncio` | Future | Await, outcome setters, completion callbacks, state inspection | One eventual result, exception, or cancelled state |
| `asyncio` | Task | Await, cancellation, outcome inspection | Driving a coroutine and tracking its wait |
| `asyncio` or compatible implementation | Event loop | Callback scheduling, timers, factories, I/O APIs | When callbacks run and how external events enter Python |
| `asyncio` | Runner | `asyncio.run()` / `asyncio.Runner` | Top-level loop lifecycle and shutdown |
| `asyncio` | TaskGroup | Async scope and child creation | Child lifetime and failure propagation |
| `asyncio` | Locks, Events, semaphores, queues | Cooperative acquisition, notification, item transfer | Coordination among Tasks |
| Operating system | I/O and execution facilities | Readiness/completion notifications, threads, processes | External operations and worker execution |

The event loop is part of asyncio's architecture, not a separate language feature. The public methods allow compatible implementations. A valid Python awaitable must also yield signals understood by its chosen runtime.

## The resume–yield contract

Calling a coroutine function creates a new coroutine object without running its body. A driver starts it with `.send(None)`. It then observes one of three outcomes:

| Outcome | What the driver sees |
| --- | --- |
| Suspension | The send call returns a yielded object |
| Successful completion | `StopIteration` carries the coroutine's return value |
| Failure | An exception escapes the send call |

An `await` delegates execution to the awaited object. A yield at the bottom of the chain propagates to the driver, preserving the chain's execution states. An await that completes without yielding does not suspend. These mechanisms exist independently of asyncio. [Language contracts](https://docs.python.org/3.14/reference/datamodel.html#coroutines)

## The scheduling and completion contract

```text
Loop runs Task callback
    → Task resumes coroutine
        → coroutine awaits operation
            → operation awaits pending Future
                → Future yields through the chain
    ← Task registers Future wake-up callback and returns
Loop runs other work or waits for external events
    → producer callback completes Future
        → Future schedules wake-up through loop
Loop runs wake-up callback
    → Task resumes coroutine chain
        → await produces a result or raises
```

The Future identifies a dependency; it does not perform I/O itself. An adapter, timer, or other producer completes it. The loop runs callbacks serially on its thread, and those callbacks must give control back before other work can run. Completion makes a waiting Task eligible to resume; it does not interrupt executing code.

CPython uses a private Task/Future handshake for pending waits. Its queue structures and coordination flags are implementation details. Applications use public await, callback, and loop APIs. [Future implementation](https://github.com/python/cpython/blob/3.14/Lib/asyncio/futures.py)

## Composition and ownership rules

- Directly awaiting a coroutine extends the current Task's execution chain.
- Creating a Task gives a fresh coroutine independent scheduling; it does not create another thread.
- Distinct coroutine objects have distinct execution states but can reference shared data.
- A coroutine object represents one execution. A completed Task retains an outcome that can be awaited repeatedly.
- Cancellation requests exception delivery; await completion to observe cleanup.
- A TaskGroup owns related child work. Async iteration and async context management do not by themselves create child Tasks.
- Cooperative scheduling allows shared-state races across suspension points. Locks and bounded queues address different coordination needs.
- Blocking operations need an appropriate boundary, such as a worker thread for blocking I/O. Cancelling a wait does not forcibly stop worker execution.

## Where to look next

Use the [numbered reading path](README.md#reading-order) for runnable demonstrations and explanations of each contract. The chapters distinguish public behavior from CPython implementation details and include links to the corresponding official references.
