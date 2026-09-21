# 13 — Thread and Process Boundaries

## Async wrappers cannot make arbitrary calls nonblocking

On the event-loop thread, synchronous file access, a blocking library call, or a long computation prevents other callbacks from running until it returns. Move suitable blocking operations to another execution facility and await their outcomes.

`asyncio.to_thread(function, *args)` returns a coroutine that runs the function in a worker thread when driven. It also propagates the current `contextvars` context. A thread is useful for blocking I/O. Cancelling the await does not forcibly stop a function already executing in that thread. [Thread integration](https://docs.python.org/3.14/library/asyncio-task.html#running-in-threads)

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory


async def main(path):
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    assert text == "hello"
    print(text)


with TemporaryDirectory() as directory:
    path = Path(directory) / "message.txt"
    path.write_text("hello", encoding="utf-8")
    asyncio.run(main(path))
```

The setup writes a file before the event loop starts. The read then happens in a worker, and its completion lets the main Task resume. This tiny example demonstrates the boundary, not a performance improvement for tiny files.

## CPU work and executor contracts

For substantial pure-Python CPU work on a conventional GIL-enabled CPython build, processes can execute work on separate cores. A `ProcessPoolExecutor` requires transferable inputs and results; functions and arguments generally must be picklable. Process startup and serialization have costs. Protect process-launching script entry with `if __name__ == "__main__":`.

Thread-based CPU parallelism depends on the interpreter build and whether native code releases the GIL; async syntax itself makes no such promise. [Executor contracts](https://docs.python.org/3.14/library/concurrent.futures.html)

## Crossing back into a loop

| Boundary | Appropriate API |
| --- | --- |
| Another thread → loop callback | `loop.call_soon_threadsafe(callback, ...)` |
| Another thread → coroutine execution | `asyncio.run_coroutine_threadsafe(coro, loop)` |
| Async code → executor work | `loop.run_in_executor(executor, function, ...)` |

The Future returned by `run_coroutine_threadsafe` belongs to `concurrent.futures`, whose `result()` can block the calling thread. An `asyncio.Future.result()` never waits. These Future types are different contracts despite the shared name; `asyncio.wrap_future()` bridges an appropriate concurrent Future into asyncio. Never block the loop thread waiting for work that needs that same loop. [Cross-thread scheduling](https://docs.python.org/3.14/library/asyncio-dev.html#concurrency-and-multithreading)

## Check your understanding

**If you cancel a Task awaiting `to_thread`, has its worker function necessarily stopped?**

No. Cancellation stops waiting cooperatively; it cannot forcibly interrupt a running thread function. Worker lifetime and shutdown still matter.

[Reading order](README.md)
