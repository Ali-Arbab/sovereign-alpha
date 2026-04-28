# ADR-0003: ZeroMQ + MessagePack for the simulator → UE5 bridge

**Status:** Accepted
**Date:** 2026-04-28

## Context
Module III renders the Module II simulator state as a real-time procedural city in Unreal Engine 5. At simulation timescale (1× to 1000×), the UE5 client receives thousands of state updates per second across five topics (`prices`, `trades`, `sentiment`, `portfolio_state`, `regime_events`). The transport must support: (a) low per-message overhead, (b) topic-based filtering on the subscriber side, (c) multiple concurrent subscribers without back-pressuring the simulator.

## Decision
**ZeroMQ PUB/SUB + MessagePack** for the bridge. The wire format is `<topic-utf8>|<msgpack-payload>` so SUB sockets use ZMQ's prefix-based topic filter natively. `pack_message` / `unpack_message` are pure functions for testability; `Publisher` / `Subscriber` wrap a shared `zmq.Context`.

## Alternatives considered
- **JSON-over-HTTP / REST.** Familiar but high serialization overhead and request-response semantics are wrong for a streaming firehose. Rejected.
- **gRPC + Protobuf.** Industrial-grade but adds significant build/deploy weight and a second IDL on top of Pydantic schemas. Overkill for a single-node, single-renderer bridge.
- **Shared memory (mmap) with a small ring buffer.** Lowest latency, but UE5 cannot easily consume Python-process shared memory cross-language without a custom plugin. Punt unless we hit a transport bottleneck.

## Consequences
- UE5 needs a MessagePack deserializer (msgpack-c or msgpack-cxx). Both are header-only and trivial to drop into a UE5 plugin.
- Topic taxonomy is fixed at the schema level (`messages.py`); changes require a `schema_version` bump.
- ZMQ inproc / ipc / tcp transports cover (in order): unit tests, single-machine multi-process, and remote client topologies, with no API surface change.
