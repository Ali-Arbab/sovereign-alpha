# ADR-0005: Content-addressed caching at every pipeline stage

**Status:** Accepted
**Date:** 2026-04-28

## Context
Per master directive §6.3, the iterative research loop must turn "change a persona prompt" into a sub-minute Stage II/III recompilation. The compute budget for re-running unchanged stages is unjustifiable — tokenizing 1.8M documents takes ~hours; re-tokenizing them on every run because something downstream changed is wasteful by 3-4 orders of magnitude.

## Decision
**Every pipeline stage is content-addressed.** A stage's cache key is `hash(inputs + version)`; a hit returns the previous output, a miss recomputes and writes.

| Stage | Key |
|---|---|
| Raw corpus → tokenized chunks | `hash(text + tokenizer_version)` |
| Tokenized chunks → inference output | `hash(chunks + model_id + persona_id + seed)` |
| Inference output → Alpha Ledger | `hash(inference_outputs + schema_version)` |

The five-tuple `(corpus_hash, persona_hash, model_hash, seed, lockfile_hash)` collapses to a single `manifest_id()` (ADR implementation: `shared/manifests/run_manifest.py`).

## Alternatives considered
- **Time-based cache invalidation.** Easy but wrong: a re-run with identical inputs would recompute simply because `mtime` is stale.
- **Manual cache invalidation flags.** Error-prone; one missed flag silently produces stale outputs.
- **No caching, full recompute every run.** Wastes hours to days per iteration; defeats the prompt-as-alpha thesis.

## Consequences
- Cache directories grow; periodic GC needed (out of scope for v0).
- A stage version bump (e.g. tokenizer upgrade) cleanly invalidates only its own cache, not downstream stages whose inputs are unchanged in content.
- Two runs with byte-identical manifests must produce byte-identical outputs (tested in `test_run_manifest.py`).
