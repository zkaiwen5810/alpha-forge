# 09 — Async Iteration and Resource Scopes

## Language protocols for repeated waits and scoped work

Python provides two ways to compose awaitable operations around familiar control flow. Neither automatically creates Tasks.

| Syntax | Required interface | Meaning |
| --- | --- | --- |
| `async for item in source` | `source.__aiter__()` returns an async iterator; its `__anext__()` returns an awaitable | Await each next item until `StopAsyncIteration` |
| `async with resource` | `__aenter__()` and `__aexit__(exc_type, exc, traceback)` return awaitables | Await entry and exit, including exceptional exit |

An async context manager is an object implementing the second contract. As with ordinary context managers, a truthy exit result can suppress an exception. An `async for` body processes one item at a time unless you explicitly schedule concurrent work. [Async statement contracts](https://docs.python.org/3.14/reference/compound_stmts.html#coroutines)

## Async generators supply an iterator conveniently

An `async def` containing `yield` creates an **async generator**. Calling it returns an async iterator, not a coroutine to pass to `create_task`. Its yielded values are items for its consumer. Its internal awaits may suspend the consumer's execution chain.

```python
import asyncio


async def numbers():
    for value in range(3):
        await asyncio.sleep(0)
        yield value


class Session:
    async def __aenter__(self):
        await asyncio.sleep(0)
        print("opened")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await asyncio.sleep(0)
        print("closed")
        return False


async def main():
    async with Session():
        async for value in numbers():
            print(value)


asyncio.run(main())
```

Output is `opened`, then `0`, `1`, `2`, then `closed`. One main Task drives the entry, iteration, and exit operations. The artificial zero-delay waits represent places where real resource code might wait for I/O.

## Resource ownership

Scope entry must succeed before scope exit is guaranteed by `async with`. If entry fails partway through acquiring a resource, entry code must handle that partial acquisition itself. Exit code can await cleanup; asynchronous syntax does not make cleanup immune to interruption.

If an async generator owns resources and iteration stops early, explicitly close it, for example with `contextlib.aclosing`, so its cleanup runs at a defined point. Exhaustion and abandoning an iterator are different events. [Deterministic async-generator cleanup](https://docs.python.org/3.14/library/contextlib.html#contextlib.aclosing)

## Check your understanding

**Does `async for` run all iterations concurrently?**

No. It awaits one next-item operation and executes its body before requesting the next item.

[Reading order](README.md)
