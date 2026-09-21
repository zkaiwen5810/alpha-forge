# 08 — I/O Through the Whole Stack

## From a socket wait to resumed Python code

The operating system manages sockets and can report when I/O can progress or has completed. The event loop adapts these notifications into callbacks. An I/O adapter uses those callbacks to complete Futures. Tasks waiting on the Futures can then resume.

Two common backends differ in their OS contract:

| Backend model | Notification | Adapter's responsibility |
| --- | --- | --- |
| Readiness | A socket may be readable or writable | Attempt nonblocking I/O, handling the possibility of waiting again |
| Completion | A submitted operation has completed | Retrieve its outcome and notify the consumer |

Python supplies selector-based and Windows proactor-based loop implementations. Their supported low-level operations differ, while both serve the asyncio scheduling model. The OS does not know about Python coroutine objects. [Loop implementations](https://docs.python.org/3.14/library/asyncio-eventloop.html#event-loop-implementations)

## A local, runnable experiment

This uses two connected sockets in the same process. No remote server is required. A short timer initiates the write so the receive has time to wait.

```python
import asyncio
import socket


async def send_later(loop, sender):
    await asyncio.sleep(0.01)
    await loop.sock_sendall(sender, b"hello")


async def main():
    loop = asyncio.get_running_loop()
    receiver, sender = socket.socketpair()
    receiver.setblocking(False)
    sender.setblocking(False)
    sending = asyncio.create_task(send_later(loop, sender))
    try:
        data = b""
        while len(data) < 5:
            chunk = await loop.sock_recv(receiver, 5 - len(data))
            if not chunk:
                raise EOFError("sender closed before the message completed")
            data += chunk
        assert data == b"hello"
        await sending
        print(data.decode())
    finally:
        sending.cancel()
        try:
            await sending
        except asyncio.CancelledError:
            pass
        receiver.close()
        sender.close()


asyncio.run(main())
```

The `finally` block stops and joins the sender before closing the sockets. Here, “join” means waiting for the child to finish; `cancel()` requests interruption and `CancelledError` acknowledges it.

## Trace the receive on a readiness-based loop

1. The main Task enters `sock_recv` through `await`.
2. If bytes are already available, the receive can finish immediately. Otherwise, the adapter registers read interest and awaits an internal Future.
3. That pending Future is yielded to the main Task. The Task registers its wake-up callback and returns to the loop.
4. The sender's timer completes its wait. The sender Task runs and writes bytes.
5. The OS reports read readiness. The loop runs the receive adapter's callback.
6. The adapter reads bytes and completes the receive Future.
7. Future completion queues the main Task's wake-up. When run, that callback resumes the await chain, delivering the bytes.

The exact low-level path varies by backend. A successful I/O operation may require several readiness notifications, and a receive can return fewer bytes than requested. The example accumulates chunks until its agreed five-byte message is complete. Real protocols need an explicit framing rule, such as a length prefix or delimiter. [Socket API](https://docs.python.org/3.14/library/asyncio-eventloop.html#working-with-socket-objects-directly)

## Higher-level wrappers

Asyncio also exposes **transports and protocols** as a callback-oriented I/O layer. A transport controls the connection and buffering through operations such as `write()` and `close()`. A protocol receives notifications such as `connection_made()`, `data_received()`, and `connection_lost()`. Protocol callbacks are ordinary synchronous methods; the loop does not automatically await their return values. [Transport/protocol contracts](https://docs.python.org/3.14/library/asyncio-protocol.html)

Streams wrap this plumbing with a reader, a buffered writer, and flow control. `reader.read()` can wait for incoming data. `writer.write()` queues data synchronously; `await writer.drain()` cooperates with write-buffer flow control. Drain does not prove the remote application consumed the message. Low-level socket operations and stream/transport APIs are different abstraction choices; a program does not have to pass through every one for each operation. [Stream contracts](https://docs.python.org/3.14/library/asyncio-stream.html)

```text
Application await
    → stream or socket adapter
    → pending Future → Task suspends

OS event → loop callback → adapter completes Future
    → scheduled Task wake-up → application continues
```

## Check your understanding

**Does a network event directly call `.send()` on your coroutine?**

No. The event reaches an adapter callback, which completes a Future; the scheduled Task wake-up drives the coroutine.

[Reading order](README.md)
