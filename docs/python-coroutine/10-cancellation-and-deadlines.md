# 10 — Cancellation and Deadlines

## Cancellation travels through execution as an exception

`task.cancel()` requests that `asyncio.CancelledError` be delivered to the coroutine. It does not forcibly interrupt synchronous code. A Task waiting on another Future or Task normally propagates cancellation to that wait. Cleanup runs as the exception travels through the await chain.

Use `try/finally` to release resources. If you catch `CancelledError`, normally re-raise it after cleanup. It inherits from `BaseException`, so `except Exception` does not catch it. The Task becomes cancelled when that exception escapes; merely requesting cancellation is not the same as observing a cancelled outcome. [Cancellation contract](https://docs.python.org/3.14/library/asyncio-task.html#task-cancellation)

```python
import asyncio


async def worker(started):
    try:
        started.set_result(None)
        await asyncio.sleep(3600)
    finally:
        print("worker cleaned up")


async def main():
    started = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(worker(started))
    await started  # Ensure the worker has entered its try block.
    task.cancel()
    try:
        await task  # Observe completion, not just the request.
    except asyncio.CancelledError:
        print("cancellation completed")
    assert task.cancelled()


asyncio.run(main())
```

The readiness Future gives this example a concrete synchronization point. Cancelling a Task before its coroutine ever starts does not execute cleanup in a body it never entered.

## Deadlines use the same mechanism

`asyncio.timeout(seconds)` is an async context manager that uses cancellation to interrupt an overdue operation and translates its own cancellation into `TimeoutError` outside the scope. A deadline is not a hard execution kill: blocking code can delay delivery, and cleanup takes time.

```python
import asyncio


async def main():
    try:
        async with asyncio.timeout(0.01):
            await asyncio.sleep(3600)
    except TimeoutError:
        print("deadline expired")


asyncio.run(main())
```

`asyncio.wait_for()` instead wraps a particular awaitable with a timeout and waits for its cancellation to complete. `asyncio.shield()` can prevent a caller's cancellation from cancelling an inner Task, but the caller still receives cancellation and someone must still own the inner work. [Timeout and shielding APIs](https://docs.python.org/3.14/library/asyncio-task.html#timeouts)

## Check your understanding

**After calling `cancel()`, can you assume a worker has released its resource?**

No. Await its completion. The request and the worker's cleanup happen at different points.

[Reading order](README.md)
