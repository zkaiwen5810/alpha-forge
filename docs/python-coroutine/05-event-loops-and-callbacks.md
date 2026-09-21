# 05 — Event Loops and Callbacks

## A scheduler for ordinary callables

A **callback** is a callable registered for later invocation. An **event loop** runs ready callbacks and waits for external events when there is no ready work. `asyncio` supplies loop implementations and an API that alternative implementations can support.

You can understand callback scheduling without any coroutine:

```python
import asyncio


loop = asyncio.new_event_loop()
try:
    loop.call_soon(print, "ready callback")
    loop.call_later(0.01, print, "timer callback")
    loop.call_later(0.02, loop.stop)
    loop.run_forever()
finally:
    loop.close()
```

Registering `print` does not invoke it immediately. `run_forever()` starts processing work; `stop()` requests that it stop. This low-level setup makes the loop's responsibility visible.

## Public scheduling contract

| Method | Contract |
| --- | --- |
| `call_soon(callback, *args)` | Queue a callback; registration order is preserved |
| `call_later(delay, callback, *args)` | Schedule using the loop's monotonic clock |
| `call_soon_threadsafe(callback, *args)` | Submit safely from another thread |
| `create_future()` / `create_task(coro)` | Construct loop-associated runtime objects |

The factory methods create two asyncio objects: a Future holds an eventual outcome, and a Task drives a coroutine. The loop provides their scheduling environment.

Scheduled callbacks can be cancelled through returned handles. Callback contexts use `contextvars`, Python's mechanism for context-local values such as a request identifier; a supplied context can override the captured context. Callbacks on one loop thread execute serially. They must return promptly for other callbacks to run. Timers are scheduling requests, not hard real-time guarantees. [Event-loop contract](https://docs.python.org/3.14/library/asyncio-eventloop.html#scheduling-callbacks)

## Implementation versus contract

CPython's default loop machinery uses ready callbacks, scheduled timers, and an operating-system I/O backend. Conceptually:

```text
repeat:
    determine how long it is safe to wait
    collect external events
    make due timers and event callbacks ready
    run a batch of ready callbacks
```

Exact queue structures and batching are implementation details. The default implementation deliberately limits a ready batch so callbacks added by callbacks wait for another iteration. [CPython loop implementation](https://github.com/python/cpython/blob/3.14/Lib/asyncio/base_events.py)

## Starting an application

For coroutine-based programs, `asyncio.run(main())` creates and manages a loop, drives the main coroutine, and performs shutdown, including asynchronous-generator finalization and default-executor shutdown. An executor is a service for running work in worker threads or processes. `asyncio.Runner` is a context manager for making multiple top-level calls within one managed loop. Neither entry mechanism can run inside another running loop on the same thread. In an already async environment, await operations using that environment's loop. [Runner contract](https://docs.python.org/3.14/library/asyncio-runner.html#running-an-asyncio-program)

## Check your understanding

**Can a timer callback interrupt another callback that is computing?**

No. It becomes eligible to run; the current callback must first give control back.

[Reading order](README.md)
