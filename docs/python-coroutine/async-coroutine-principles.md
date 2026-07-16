# Async Coroutines in Python — Core Principles

## What is a Coroutine?

A **coroutine** is a special type of function that can **suspend its execution** at a point and later **resume** from where it left off, while allowing other tasks to run in the meantime.

In Python, coroutines are defined with `async def` and use `await` to yield control.

## The Key Idea: Cooperative Multitasking

Unlike threads (preemptive multitasking), coroutines use **cooperative multitasking** — the coroutine itself decides when to pause (`await`) and let others run. This avoids race conditions and locking complexity in single-threaded code.

## How It Works Under the Hood

1. **Event Loop** — The central scheduler that orchestrates all coroutines. It maintains a queue of tasks.

2. **Await Points** — When a coroutine hits `await`, it **suspends** and returns control to the event loop. The event loop can then run another ready coroutine.

3. **Future / Awaitable** — `await` expects an **awaitable** object (coroutine, Future, Task). When awaited, the event loop checks: is the result ready? If yes, continue immediately; if no, suspend and register a callback to wake up when done.

### The Flow

```python
async def fetch_data():
    print("Start fetch")
    result = await some_io_operation()  # << suspend here
    print("Got result:", result)
    return result

# The event loop runs this:
# 1. Execute "Start fetch"
# 2. Hit await → suspend, let other tasks run
# 3. When I/O completes → resume from await
# 4. Print result, return
```

## Critical Distinctions

| Concept | Definition |
|---------|------------|
| **Coroutine** | An `async def` function; calling it returns a coroutine object, **not** the result. |
| **Coroutine object** | Must be **awaited** or scheduled as a Task; otherwise it's just an unused object. |
| **Task** | Wraps a coroutine into an independent unit of work scheduled on the event loop. |
| **Future** | A low-level awaitable representing a result that will be available later. |

## Why Not Just Threads?

- **Single-threaded** → no GIL contention, no thread-safety nightmares for shared data.
- **Lightweight** → coroutines have tiny memory overhead vs. OS threads.
- **Explicit** → you see exactly where suspension points are (`await`), making control flow clear.

## Minimal Example

```python
import asyncio

async def hello():
    print("Hello")
    await asyncio.sleep(1)   # suspend here, let others run
    print("World")

async def main():
    await asyncio.gather(hello(), hello())

asyncio.run(main())
```

Output:
```
Hello
Hello
(1 second pause)
World
World
```

Both `hello()` tasks run concurrently — the second begins while the first is sleeping.

## Summary

**Python coroutines** are a cooperative concurrency primitive that:
- Suspend/resume execution at `await` points.
- Run on a single-threaded event loop.
- Enable high-concurrency I/O without threads.
- Are explicit and lightweight.