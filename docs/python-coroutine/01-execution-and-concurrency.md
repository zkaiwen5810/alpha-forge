# 01 — Execution and Concurrency

## The problem async execution solves

An ordinary function call runs on the calling thread until it returns or raises. If it waits synchronously for a network response, that thread cannot use the waiting time to execute another function. This is **blocking**.

**Suspension** preserves a computation's state and gives control back to its driver. The driver can run something else and resume the computation when progress becomes possible. Preserving state does not require assigning a separate thread to each suspended computation.

| Term | Meaning | Example |
| --- | --- | --- |
| Concurrency | Multiple operations are in progress during overlapping periods | Two downloads both waiting for bytes |
| Parallelism | Execution happens simultaneously | Calculations on two CPU cores |
| Blocking | A thread cannot proceed past a synchronous operation | Calling `time.sleep()` |
| Suspension | A computation stops temporarily while its driver regains control | A resumable function yielding a waiting request |

Imagine two operations, each needing a little computation, a long external wait, and a little more computation:

```text
Sequential:  A compute — A wait — A finish — B compute — B wait — B finish
Concurrent:  A compute — A wait ....................... A finish
                         B compute — B wait — B finish
```

Concurrency can overlap those waits. It does not shorten the amount of computation or guarantee that either operation finishes sooner in isolation.

## Cooperative execution

A cooperative scheduler gets control when running work gives it back. On one event-loop thread, another scheduled operation cannot interrupt a long-running callback. A callback is an ordinary callable registered to run when the scheduler chooses it.

This means an apparently small synchronous call can stall every operation sharing that loop. Asynchronous syntax cannot turn blocking code into nonblocking code by itself. This is why asyncio's development guide separates blocking work from event-loop execution. [Blocking-code guidance](https://docs.python.org/3.14/library/asyncio-dev.html#running-blocking-code)

## Check your understanding

**If two requests each spend most of their time waiting, do you need two CPU cores to overlap their progress?**

No. One thread can start one operation, suspend it while it waits, and start the other. Parallel CPU execution is a separate capability.

[Reading order](README.md)
