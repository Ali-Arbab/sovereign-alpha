# ADR-0002: `as_of_join` is the only sanctioned temporal merge

**Status:** Accepted
**Date:** 2026-04-28

## Context
Master directive §4.2 / §6.4 elevate the **temporal firewall** to a build-breaking invariant. The single highest-value research property is that state at simulation time *t* depends on no information drawn from `timestamp > t`. The naive failure mode is silent: an `inner_join` on rounded timestamps can match a bar at *t* with a ledger row at *t + ε*, leaking future state into past observations.

## Decision
**Every temporal merge in Module II MUST use a backwards `as_of_join` on monotonic `epoch_ns`.** Naive `inner_join` on timestamp columns is build-broken.

The `as_of_fuse()` wrapper in `modules/module_2_quant/fusion.py`:
1. Verifies both inputs are sorted ascending on the join key on every call (not assumed).
2. Calls Polars' `join_asof(strategy="backward")`.
3. Is gated by the synthetic future-leak corpus and a build-breaking test (`test_temporal_firewall.py`).

## Alternatives considered
- **`inner_join` with `timestamp_floor` discipline.** Requires every caller to remember to floor; one mistake produces silent leakage. Rejected as too easy to break.
- **Full event-loop simulation (per-bar Python iteration).** Correct but ~50× slower. Used as the reference for `MonotonicCursor` but not as the production join.
- **Stream-processing backtester (e.g. Vectorbt event mode).** Correct semantics but couples the firewall to a specific framework's lifecycle.

## Consequences
- Strategy authors must sort their inputs before fusion. Sortedness is checked on every call.
- The leak-test corpus + meta-test (which itself verifies the corpus *is* poisoned) is the canonical regression suite for the firewall.
- Cross-column geometry checks (`OHLCVFrame.low <= open <= high`) are layered separately because pandera-polars doesn't expose `@dataframe_check` at the version pin.
