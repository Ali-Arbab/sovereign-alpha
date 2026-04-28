# Architecture Decision Records

Per master directive §0.5.1.H. Each ADR captures a single significant decision: **what** we chose, **why we rejected the alternatives**, and **what we now have to live with**.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-polars-over-pandas.md) | Polars over Pandas for Module II | Accepted |
| [0002](0002-as-of-join-only.md) | `as_of_join` is the only sanctioned temporal merge | Accepted |
| [0005](0005-content-addressed-caching.md) | Content-addressed caching at every pipeline stage | Accepted |
| [0008](0008-toml-personas.md) | Persona definitions in TOML, parsed via stdlib `tomllib` | Accepted |

## Template

```
# ADR-NNNN: Short imperative title

**Status:** Proposed / Accepted / Superseded by ADR-XXXX
**Date:** YYYY-MM-DD

## Context
What is the problem? What forces are at play?

## Decision
What did we choose? State it as an imperative.

## Alternatives considered
What else did we look at, and why was each rejected?

## Consequences
What new constraints, costs, or capabilities does this introduce?
```

A new ADR is added by creating the next-numbered file and appending to the index above.
