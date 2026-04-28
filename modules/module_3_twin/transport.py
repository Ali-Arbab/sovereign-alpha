"""ZMQ PUB/SUB transport over MessagePack -- Python side of the UE5 bridge.

Wire format: `<topic-utf8>|<msgpack-payload-bytes>`. Single delimiter so SUB
sockets can use ZMQ's prefix-based topic filter natively.

`pack_message` / `unpack_message` are pure functions (testable without sockets).
`Publisher` / `Subscriber` wrap a single shared zmq Context for the process.
"""

from __future__ import annotations

import contextlib
from typing import Any

import msgpack
import zmq
from pydantic import BaseModel

TOPIC_DELIM: bytes = b"|"


def pack_message(topic: str, msg: BaseModel) -> bytes:
    """Serialize `msg` and prepend `topic|`. Raises ValueError if topic empty
    or contains the delimiter byte."""
    if not topic:
        raise ValueError("topic must be non-empty")
    if TOPIC_DELIM in topic.encode("utf-8"):
        raise ValueError(f"topic must not contain {TOPIC_DELIM!r}")
    payload = msgpack.packb(msg.model_dump(), use_bin_type=True)
    return topic.encode("utf-8") + TOPIC_DELIM + payload


def unpack_message(buf: bytes) -> tuple[str, dict[str, Any]]:
    """Split a wire frame into `(topic, payload_dict)`."""
    idx = buf.find(TOPIC_DELIM)
    if idx < 0:
        raise ValueError("frame missing topic delimiter")
    topic = buf[:idx].decode("utf-8")
    payload = msgpack.unpackb(buf[idx + 1 :], raw=False)
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    return topic, payload


class Publisher:
    """ZMQ PUB binding to `endpoint` on a shared Context.

    Use `tcp://*:5555` for network publishing, `inproc://name` for in-process
    fan-out (useful in tests), or `ipc:///path` for cross-process on Unix.
    Windows does not support ipc:// -- use tcp:// or inproc:// there.
    """

    def __init__(self, endpoint: str, *, context: zmq.Context | None = None) -> None:
        self._owns_context = context is None
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(endpoint)
        self.endpoint = endpoint

    def publish(self, topic: str, msg: BaseModel) -> None:
        self.socket.send(pack_message(topic, msg))

    def close(self) -> None:
        self.socket.close(linger=0)
        if self._owns_context:
            # Don't terminate the shared default context -- only the dedicated one
            with contextlib.suppress(zmq.ZMQError):
                self.context.term()

    def __enter__(self) -> Publisher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Subscriber:
    """ZMQ SUB connecting to `endpoint`. Pass `topics=None` for all topics."""

    def __init__(
        self,
        endpoint: str,
        topics: list[str] | None = None,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        self._owns_context = context is None
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(endpoint)
        if topics is None:
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for t in topics:
                if not t:
                    raise ValueError("topic filters must be non-empty")
                self.socket.setsockopt(zmq.SUBSCRIBE, t.encode("utf-8"))
        self.endpoint = endpoint

    def recv(self, *, timeout_ms: int | None = None) -> tuple[str, dict[str, Any]] | None:
        """Block (or wait up to `timeout_ms`) for one message. Returns None on
        timeout, never blocks indefinitely when timeout is set."""
        if timeout_ms is not None and not self.socket.poll(timeout_ms):
            return None
        return unpack_message(self.socket.recv())

    def close(self) -> None:
        self.socket.close(linger=0)
        if self._owns_context:
            with contextlib.suppress(zmq.ZMQError):
                self.context.term()

    def __enter__(self) -> Subscriber:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
