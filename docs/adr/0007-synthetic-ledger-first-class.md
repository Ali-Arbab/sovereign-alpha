# ADR-0007: Synthetic Alpha Ledger is a first-class artifact, not a placeholder

**Status:** Accepted
**Date:** 2026-04-28

## Context
Master directive §0.5 (the pre-hardware bootstrap phase) says: "the synthetic Alpha Ledger is the engine that unblocks everything downstream." Without it, Module II cannot be developed — every test, every backtest, every friction-modeling refinement waits on real Module I inference, which waits on the Sovereign hardware. The temptation is to call this a "placeholder" or "stub" — code that gets ripped out when real data lands.

## Decision
**The synthetic Alpha Ledger generator is a first-class component**, with the same engineering rigor as any production module: schema-validated, byte-deterministic on `(seed, range)`, Hive-partitioned Parquet output, comprehensive tests. Every record is tagged `persona_id="bootstrap_synthetic_v1"` and `model_id="synthetic"` so the data is **structurally distinguishable** from real research output and cannot accidentally contaminate it.

## Alternatives considered
- **Skip Module II until hardware lands.** Wastes the entire bootstrap-phase mandate. Rejected.
- **Hand-rolled toy data per test.** Tests still pass but no end-to-end pipeline ever exercises Module II as a whole.
- **Hard-code bootstrap synthetic data in tests.** Less reusable; can't be used to drive UE5 bridge development or `bootstrap-test`.

## Consequences
- Module II is built and tested at full pipeline scope before real Module I inference exists.
- The `bootstrap-test` script exercises everything — synthetic ledger → fusion → backtest → bridge — in <1s.
- When real inference lands (post-hardware), the swap is mechanical: replace the generator's output with real `AlphaLedgerRecord` rows from the inference pipeline. Module II downstream code does not change.
- The bootstrap-phase tagging prevents a synthetic record from ever masquerading as a research artifact in the published Alpha Ledger.
