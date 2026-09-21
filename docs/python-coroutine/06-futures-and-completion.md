# 06 — Futures and Completion

## A bridge between callbacks and awaiting code

An `asyncio.Future` stores one eventual outcome. It begins pending, then finishes with a result, an exception, or cancellation. It does not execute a coroutine or start an operation by itself.

| Interface | Used by | Purpose |
| --- | --- | --- |
| `set_result(value)` / `set_exception(error)` | Operation producer | Complete a pending Future |
| `await future` | Consumer | Wait, then receive its outcome |
| `result()` | Consumer | Retrieve now; raises `InvalidStateError` if pending |
| `add_done_callback(callback)` | Runtime or adapter | Arrange completion notification |
| `done()` / `cancelled()` | Either side | Inspect terminal state |

Create Futures with `loop.create_future()`. They belong to that loop and are not thread-safe. Completion callbacks are scheduled with the loop rather than invoked inline, including callbacks added after completion. A Future can be awaited repeatedly. [Future contract](https://docs.python.org/3.14/library/asyncio-future.html#future-object)

```python
import asyncio


async def main():
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def finish():
        if not future.done():
            future.set_result(42)

    timer = loop.call_later(0.01, finish)
    try:
        value = await future
        assert value == 42
        assert await future == 42
        print(value)
    finally:
        timer.cancel()


asyncio.run(main())
```

The timer starts the completion path. Constructing `future` alone would never make it finish. The `done()` guard also prevents the callback from completing a Future that was cancelled while waiting.

## The await machinery inside a Future

Conceptually, a Future's await iterator yields the Future itself if it is pending, then retrieves its outcome on resumption. CPython also uses a private coordination flag and validates that resumption did not happen prematurely. Those internals are not a custom-awaitable API to copy. [Future implementation](https://github.com/python/cpython/blob/3.14/Lib/asyncio/futures.py)

The important distinction is between the **yielded Future**, which tells the driver what to wait for, and the **stored result**, which the completed await returns. If the Future is already complete, its await iterator can return without yielding.

## Check your understanding

**Does `future.result()` wait for completion?**

No. It retrieves an available outcome or raises. Use `await future` when suspension is needed.

[Reading order](README.md)
