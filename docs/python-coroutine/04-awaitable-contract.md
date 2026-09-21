# 04 — The Awaitable Contract

## What `await` accepts

An **awaitable** is an object that can be used with `await`. Native coroutines qualify. A custom class can provide `__await__()`, which must return an iterator. The interpreter delegates execution through that iterator: yielded values travel outward to the driver, and the iterator's return value becomes the await expression's result. This explains why `await` need not suspend: the iterator might finish without yielding. [Await expressions and awaitables](https://peps.python.org/pep-0492/#await-expression)

## Watch control leave and re-enter a coroutine

```python
class Pause:
    def __await__(self):
        received = yield "pause requested"
        print("received from driver:", received)
        return 42


async def child():
    return await Pause()


async def parent():
    value = await child()
    return value * 2


coroutine = parent()
signal = coroutine.send(None)
print("driver received:", signal)

try:
    coroutine.send("continue")
except StopIteration as finished:
    print("final result:", finished.value)
```

Output:

```text
driver received: pause requested
received from driver: continue
final result: 84
```

During the first send, execution goes inward:

```text
driver → parent → child → Pause.__await__()
```

At `yield`, control and the yielded string travel outward:

```text
Pause → child suspends → parent suspends → driver
```

The driver's `.send(None)` call has now returned. That is the concrete meaning of “yielding control.” The coroutine chain remains suspended in memory. The next send resumes the innermost yield; return values then flow through the awaiting functions.

## Three values, three roles

| Value | Direction | Meaning |
| --- | --- | --- |
| `"pause requested"` | Await iterator → driver | A yielded signal |
| `"continue"` | Driver → await iterator | A resumption input |
| `42` | Await iterator → awaiting coroutine | The completed await's result |

The string signal is a contract invented for this manual driver. A runtime must recognize the yielded signals to know what to do. Merely implementing `__await__()` does not guarantee compatibility with every runtime.

Directly awaiting `child()` delegates within the same execution chain. It does not independently schedule `child`. Putting `yield` directly in an `async def` instead defines an async generator, a different object type; a custom `__await__` method uses ordinary `def` as above. [Native coroutines and generators](https://peps.python.org/pep-0492/#new-coroutine-declaration-syntax)

## Check your understanding

**Does the outer coroutine have to yield the child coroutine object to its driver?**

No. It delegates execution to the child. In this example the value reaching the driver is the string yielded at the bottom of the chain.

[Reading order](README.md)
