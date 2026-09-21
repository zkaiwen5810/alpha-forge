# Learn the Python Async Stack

This guide teaches how Python's resumable execution protocols connect to `asyncio`, callbacks, and operating-system I/O. It assumes you know ordinary functions, classes, loops, and exceptions. No async experience is required.

There are **14 topics**. The first four explain language building blocks without an event loop. The next four build the runtime and trace its coordination. The final six explain how to use that machinery in complete programs. This is a foundation for understanding the stack, not a catalog of every networking API.

## Reading order

| Step | Topic | Question you will be able to answer |
| --- | --- | --- |
| 01 | [Execution and concurrency](01-execution-and-concurrency.md) | What is the difference between blocking, suspension, concurrency, and parallelism? |
| 02 | [Generators and resumable execution](02-generators-and-resumption.md) | How can a function yield control and later continue? |
| 03 | [Coroutine objects](03-coroutine-objects.md) | What does calling `async def` create, and who drives it? |
| 04 | [The awaitable contract](04-awaitable-contract.md) | How does `await` delegate execution and propagate a yield? |
| 05 | [Event loops and callbacks](05-event-loops-and-callbacks.md) | What does a scheduler do, and what interface does it expose? |
| 06 | [Futures and completion](06-futures-and-completion.md) | How does a callback deliver an outcome to awaiting code? |
| 07 | [Tasks and scheduling](07-tasks-and-scheduling.md) | How does a Task connect coroutine execution to a loop? |
| 08 | [I/O through the whole stack](08-io-through-the-stack.md) | What happens between a network wait and coroutine resumption? |
| 09 | [Async iteration and resource scopes](09-async-iteration-and-scopes.md) | What contracts power `async for` and `async with`? |
| 10 | [Cancellation and deadlines](10-cancellation-and-deadlines.md) | How does waiting code stop and clean up? |
| 11 | [Owning concurrent work](11-owning-concurrent-work.md) | Who waits for child Tasks and handles their failures? |
| 12 | [Coordination and backpressure](12-coordination-and-backpressure.md) | How do Tasks share state and bound pending work? |
| 13 | [Thread and process boundaries](13-thread-and-process-boundaries.md) | Where does blocking or CPU-intensive code belong? |
| 14 | [Debugging the stack](14-debugging-the-stack.md) | Which layer should you inspect when progress stops? |

Read the numbered files in order. Each explains its own topic using concepts already introduced, with a short understanding check and answer. The [stack reference](async-coroutine-principles.md) is a compact recap after the course.

## Running the examples

Use Python **3.11 or newer**. Each `python` code block is a standalone script unless explicitly described otherwise. Copy a block into a file and run `python3 example.py`. Examples need only the standard library; the I/O example uses a local socket pair, not an internet service.

The teaching model uses default, non-eager Task scheduling. Eager execution is an optional mode where a Task can start immediately during creation; it is called out where timing matters. Examples avoid depending on exact wall-clock durations.

Protocol demonstrations manually resume newly created coroutines. Application examples let `asyncio` own execution. Never manually resume a coroutine that a Task is already driving.

## What belongs to which layer?

| Owner | Building blocks | Contract boundary |
| --- | --- | --- |
| Python language/runtime | Generators, coroutine objects, `await`, async iteration and context management | Methods and syntax for suspending, resuming, and composing execution |
| `asyncio` library | Futures, Tasks, runners, synchronization, streams | Completion, cancellation, work ownership, and asynchronous operations |
| Event-loop implementation, included in or compatible with `asyncio` | Callback scheduling, timers, I/O integration | Public loop methods used by Tasks, Futures, and I/O adapters |
| Operating system | Sockets, readiness/completion notifications, threads, processes | External events and execution facilities used by the loop and workers |

Python's protocols do not mandate `asyncio`. Other runtimes can drive coroutines, but their waiting objects must match their runtime's scheduling rules. An asyncio Future is not a universal cross-runtime waiting object.

Public contracts and CPython implementation details are labeled separately throughout. Official references appear beside the relevant explanation; implementation links are pinned to the CPython 3.14 branch.
