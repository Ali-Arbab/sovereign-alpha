# ADR-0004: Pydantic at boundaries, Pandera in-flight

**Status:** Accepted
**Date:** 2026-04-28

## Context
The Sovereign Alpha pipeline crosses the Python ↔ Parquet boundary repeatedly (Module I writes the Alpha Ledger; Module II reads it; the bridge serializes a subset to MessagePack; etc.). Two distinct validation needs arise: (a) per-record validation at serialization boundaries (write a row, read a row) and (b) whole-frame validation across in-flight Polars DataFrames between Module II stages.

## Decision
- **Pydantic** for single-row records at serialization boundaries: `AlphaLedgerRecord`, `OHLCVBar`, the five Module III message types, `RunManifest`, `PersonaSpec`. Frozen, `extra="forbid"`, schema_version field.
- **Pandera (polars backend)** for in-flight DataFrame contracts: `AlphaLedgerFrame`, `OHLCVFrame`. Column-level type and bound checks on the whole frame.

## Alternatives considered
- **Pydantic-only.** Iterating row-by-row to validate a 1.8M-row frame is ~1000× slower than vectorized pandera checks. Rejected on perf grounds.
- **Pandera-only.** Pandera is built for DataFrames; using it for single-record serialization boundaries is awkward and loses Pydantic's `model_dump` / `model_validate_json` ergonomics.
- **Custom Polars expressions for both.** Possible but reinvents validation libraries that already exist.

## Consequences
- Two schema definitions per data type (one Pydantic, one Pandera) that must be kept in sync. Tests assert that synthetic generators produce frames the Pandera contract accepts — the synthetic generator is itself the round-trip validator.
- Cross-column invariants (e.g. `low <= open <= high`) require either a Pydantic `model_validator` (single-row) or a custom Polars expression (whole-frame). Both are wired in `validate_ohlcv_frame`.
