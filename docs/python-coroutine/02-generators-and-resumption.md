# 02 — Generators and Resumable Execution

## Python provides the suspension mechanism

A function defined with ordinary `def` and containing `yield` is a **generator function**. Calling it creates a generator object without executing its body. The object preserves local state between resumptions.

Its driver uses `next(generator)` or `generator.send(value)`. At `yield expression`, execution suspends and the expression's value returns to the driver. On resumption, the value sent by the driver becomes the value of the suspended `yield` expression. A generator's final `return value` is reported to the driver as `StopIteration(value)`. [Generator protocol](https://docs.python.org/3.14/reference/expressions.html#generator-iterator-methods)

```python
def exchange():
    print("started")
    reply = yield "waiting"
    print("received:", reply)
    return 42


generator = exchange()
print(next(generator))  # Prints started, then waiting.

try:
    generator.send("continue")
except StopIteration as finished:
    print("returned:", finished.value)
```

Output:

```text
started
waiting
received: continue
returned: 42
```

The first call to `next()` returns because the generator yielded. No thread switch, timer, or scheduler is involved. Our script explicitly decides when to resume it.

## Exposed contract

| Operation | Effect |
| --- | --- |
| `next(g)` / `g.send(None)` | Start or resume the generator |
| `g.send(value)` | Resume and supply a value to its suspended yield |
| `g.throw(exception)` | Raise an exception at the suspension point |
| `g.close()` | Request closure by injecting `GeneratorExit` |

The first send must supply `None`, because no yield expression is waiting to receive another value yet. `yield from child_iterator` delegates iteration, including suspension and resumption, to a child iterator. These are language mechanisms; they do not choose scheduling policies.

An ordinary generator is not automatically an awaitable. It is useful here because it makes the resume–yield exchange visible with minimal machinery.

## Check your understanding

**Does `yield` itself arrange for the generator to resume later?**

No. It returns control to the driver while preserving state. The driver must arrange the next call.

[Reading order](README.md)
