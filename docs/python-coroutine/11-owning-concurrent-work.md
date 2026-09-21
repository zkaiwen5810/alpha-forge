# 11 — Owning Concurrent Work

## Every child needs an owner

Creating a Task gives a coroutine independent scheduling. It also creates obligations: retain it, decide when it should stop, wait for cleanup, and observe its outcome. Ordinary `create_task()` calls do not automatically build a parent–child lifetime tree.

**Structured concurrency** means the lifetime of concurrent child work fits inside an owning scope. `asyncio.TaskGroup` provides such a scope: its exit waits for its children. If a child fails with an ordinary exception, the group cancels remaining children, waits for them, and reports failures in an exception group. [TaskGroup contract](https://docs.python.org/3.14/library/asyncio-task.html#task-groups)

```python
import asyncio


async def calculate(value):
    await asyncio.sleep(0.01)
    return value * 2


async def main():
    async with asyncio.TaskGroup() as group:
        first = group.create_task(calculate(10))
        second = group.create_task(calculate(20))

    assert first.result() == 20
    assert second.result() == 40
    print(first.result(), second.result())


asyncio.run(main())
```

The statement after the scope has a useful guarantee: both children have finished successfully, or an exception already interrupted the flow. A child cancellation alone is not treated as an ordinary child failure that cancels its siblings.

## Handling grouped failures

An `ExceptionGroup` can contain multiple exceptions. `except*` selects matching exceptions within a group, preserving the rest for propagation. [Exception-group handling](https://docs.python.org/3.14/reference/compound_stmts.html#except-star)

```python
import asyncio


async def fail():
    raise ValueError("invalid item")


async def main():
    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(fail())
            group.create_task(asyncio.sleep(3600))
    except* ValueError as errors:
        print("handled failures:", len(errors.exceptions))


asyncio.run(main())
```

## How `gather` differs

`asyncio.gather()` collects results in input order and schedules coroutine arguments. With its default exception behavior, the first failure propagates to its awaiter, but other children are not automatically cancelled. Cancelling the pending gather operation does propagate cancellation to unfinished children. `return_exceptions=True` collects exceptions as result entries, which the caller must inspect. [Gather implementation and behavior](https://github.com/python/cpython/blob/3.14/Lib/asyncio/tasks.py)

Use a TaskGroup when related work should finish together and a failure should stop siblings. Use gather when collecting outcomes with its specific failure policy is intentional. Neither creates parallel CPU execution on a single loop thread.

## Check your understanding

**If you catch one child's failure after a default `gather`, have all siblings necessarily stopped?**

No. The default failure path does not cancel them. Their remaining lifetime still needs ownership.

[Reading order](README.md)
