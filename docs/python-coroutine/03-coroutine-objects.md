# 03 — Coroutine Objects

## Function, object, and result are different things

An `async def` function without `yield` is a **native coroutine function**. Calling it creates a coroutine object; execution has not started. Each call creates a separate execution state, even when both objects use the same arguments.

The object holds its local state, execution position, and await chain. It exposes `.send()`, `.throw()`, and `.close()` for a driver to manage execution. Returning from the coroutine completes that execution and reports the return value through `StopIteration` to the driver. Native coroutine objects are awaitable, but are not ordinary iterators accepted by `next()`. [Coroutine object contract](https://docs.python.org/3.14/reference/datamodel.html#coroutine-objects)

```python
async def double(value):
    print("executing:", value)
    return value * 2


first = double(21)
second = double(21)
assert first is not second

for coroutine in (first, second):
    try:
        coroutine.send(None)
    except StopIteration as finished:
        assert finished.value == 42
```

The two print calls happen inside the loop, during `.send(None)`, not when the objects are constructed. Each completes in one step because nothing in the body suspends.

## One execution per object

A completed coroutine cannot be restarted. Calling `double(21)` again creates another computation; awaiting a finished coroutine object again raises `RuntimeError`. Separate coroutine objects can still reference shared mutable objects, so separate execution state does not imply isolated data.

A newly created coroutine must eventually be driven, awaited, or deliberately closed. Accidentally constructing one and dropping it produces a “coroutine was never awaited” warning. Applications normally give execution ownership to an async runtime instead of calling these protocol methods themselves.

## Check your understanding

**What does `result = double(21)` store?**

A coroutine object, not `42`. The name `result` does not change the object's type or start execution.

[Reading order](README.md)
