# 12 — Coordination and Backpressure

## Single-threaded execution still has interleavings

Suppose two Tasks both read `counter = 0`, suspend, then each writes `1`. Two increments have produced only one increase. The read and write were separated by a suspension, allowing another Task to observe the old value.

Asyncio offers coordination objects whose waits cooperate with the event loop:

| Primitive | Contract | Typical use |
| --- | --- | --- |
| `Lock` | Only one Task holds it at a time | Protect a shared-state operation spanning awaits |
| `Event` | Wait until a shared flag is set | Signal readiness to multiple waiters |
| `Semaphore` | Limit simultaneous holders | Bound concurrent access to a service |
| `Queue` | Transfer items between producers and consumers | Decouple production from processing |

These are asyncio coordination tools, not thread synchronization tools. An available lock or queue item may be acquired without suspension. [Synchronization contracts](https://docs.python.org/3.14/library/asyncio-sync.html)

## Bound the work waiting in memory

**Backpressure** means slowing producers when consumers cannot keep up. A bounded queue suspends `put()` when full. A fixed number of workers bounds active processing; the queue's capacity bounds buffered items.

```python
import asyncio


async def worker(queue, results):
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            await asyncio.sleep(0)
            results.append(item * 2)
        finally:
            queue.task_done()


async def main():
    queue = asyncio.Queue(maxsize=2)
    results = []
    async with asyncio.TaskGroup() as group:
        for _ in range(2):
            group.create_task(worker(queue, results))
        for item in range(6):
            await queue.put(item)
        for _ in range(2):
            await queue.put(None)  # One stop item for each worker.
        await queue.join()

    assert sorted(results) == [0, 2, 4, 6, 8, 10]
    print(sorted(results))


asyncio.run(main())
```

`get()` removes an item; `task_done()` acknowledges processing. `join()` waits for all enqueued items to be acknowledged, including the stop items here. It does not itself wait for worker Tasks to exit; the TaskGroup does that. The stop value `None` is reserved by this example's application protocol. [Queue contract](https://docs.python.org/3.14/library/asyncio-queue.html)

A semaphore around millions of pre-created Tasks limits active access but still leaves millions of Task objects in memory. Bound task creation or use a worker queue when the input can be large. In this example there are always exactly two workers.

## Check your understanding

**Does a bounded queue alone limit how many producer Tasks exist?**

No. It limits buffered items. The application must also control how many producers and workers it creates.

[Reading order](README.md)
