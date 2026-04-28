# Architecture — C4 Model

Per master directive §0.5.1.H. Three C4 levels (Context, Containers, Components). The Code level is omitted because the codebase, Pydantic schemas, and ADRs already document at that altitude.

---

## Level 1 — System Context

What is Sovereign Alpha and who/what does it talk to?

```mermaid
flowchart LR
    OP[Operator / PI<br/>Architect + Researcher]
    SOV[(Sovereign Alpha<br/>2015-2025 historical sim)]

    subgraph EXT[External data sources]
        EDGAR[SEC EDGAR<br/>10-K / 10-Q / 8-K]
        FOMC[FOMC<br/>statements / minutes / SEP]
        BLS[BLS<br/>CPI / NFP / JOLTS]
        GDELT[GDELT<br/>open news corpus]
        OHLCV[OHLCV providers<br/>Polygon / Databento / Alpaca]
    end

    HW[Sovereign Lab Hardware<br/>RTX 5090 / 9950X3D<br/>128GB / 4TB Gen5 NVMe]

    OP -- "iterates persona prompts" --> SOV
    EDGAR -- "filings" --> SOV
    FOMC -- "monetary policy" --> SOV
    BLS -- "macro releases" --> SOV
    GDELT -- "news firehose" --> SOV
    OHLCV -- "M1/S1 bars" --> SOV
    HW -. runs .-> SOV

    SOV -- "friction-adjusted Sharpe<br/>+ per-trade attribution" --> OP
    SOV -- "navigable 3D city<br/>(4K + VR)" --> OP
```

**Boundary:** everything inside Sovereign Alpha runs on the single Sovereign Lab box (ADR-0006). External data sources are read-only, batch-ingested into the data lake; nothing leaves the box.

---

## Level 2 — Containers

The three modules + the shared infrastructure that binds them.

```mermaid
flowchart TB
    subgraph MOD1[Module I — Async Semantic Extraction]
        ING[Ingestion adapters<br/>edgar / fomc / bls / gdelt]
        TOK[Tokenization pipeline<br/>clean -> chunk -> cached encode]
        INF[LLM inference<br/>DeepSeek-R1 32B q6_k via vLLM/llama.cpp]
        PER[Persona library<br/>6 versioned analyst archetypes]
    end

    subgraph MOD2[Module II — Quant Engine]
        FUSE[Data fusion<br/>as_of_fuse + MonotonicCursor]
        STRAT[Strategy DSL<br/>Signal + TargetPctRule]
        FRIC[Friction layer<br/>slippage / commission / partial fills]
        BT[Backtest runner<br/>monotonic, attribution-bearing]
        METRICS[Metrics + Statistics<br/>Sharpe / Sortino / PSR / DSR / bootstrap]
        VAL[Out-of-sample CV<br/>walk-forward + purged k-fold]
    end

    subgraph MOD3[Module III — UE5 Digital Twin]
        BUS[ZMQ + MessagePack bus<br/>5 topics]
        UE5[UE5 client<br/>Nanite + Lumen + Niagara]
    end

    subgraph SHARED[shared/]
        SCH[Schemas<br/>Pydantic + Pandera contracts]
        MANI[Run manifests<br/>content-addressed reproducibility]
    end

    subgraph LAKE[Data lake — Hive-partitioned Parquet]
        CORPUS[Raw corpus<br/>~1.8M docs]
        LEDGER[Alpha Ledger<br/>per persona]
        BARS[OHLCV bars<br/>M1 / S1]
        RUNS[Backtest runs<br/>+ manifests]
    end

    ING --> CORPUS
    CORPUS --> TOK --> INF
    PER --> INF
    INF --> LEDGER

    LEDGER --> FUSE
    BARS --> FUSE
    FUSE --> BT
    STRAT --> BT
    FRIC --> BT
    BT --> METRICS
    BT --> VAL
    BT --> RUNS

    BT --> BUS --> UE5

    SCH -. validates .-> LEDGER
    SCH -. validates .-> BARS
    SCH -. validates .-> BUS
    MANI -. tags .-> RUNS
    MANI -. tags .-> LEDGER

    classDef bootstrap fill:#fff4e6,stroke:#d97706
    class INF bootstrap
    class UE5 bootstrap
```

**Bootstrap-phase substitutes (highlighted orange):** until the Sovereign Lab hardware lands, `INF` is replaced by the synthetic Alpha Ledger generator and `UE5` is replaced by a Python-side mock subscriber. Both run inside the same containers; only the implementations swap. See `docs/HARDWARE_ARRIVAL_DAY.md`.

---

## Level 3 — Module II Components

Zoom into the quant engine.

```mermaid
flowchart LR
    subgraph IN[Inputs]
        AL[(Alpha Ledger<br/>partitioned Parquet)]
        OH[(OHLCV bars<br/>partitioned Parquet)]
    end

    subgraph FUS[fusion.py]
        EXP[explode_ledger_entities]
        AOF[as_of_fuse<br/>backwards join_asof<br/>+ sortedness check]
    end

    subgraph CUR[cursor.py]
        MC[MonotonicCursor<br/>strict t-only filtering]
    end

    subgraph FW[Temporal firewall]
        LC[leak_corpus.py<br/>poisoned future rows]
        TF[test_temporal_firewall.py<br/>build-breaking]
    end

    subgraph STR[strategy.py]
        SIG[Signal<br/>composable boolean Polars Expr]
        RULE[TargetPctRule]
    end

    subgraph FRI[friction.py]
        FM[FrictionModel<br/>slippage / commission / partial fills]
    end

    subgraph BT[backtest.py]
        RB[run_backtest<br/>single-pass monotonic<br/>doc_hash attribution]
    end

    subgraph EVAL[Evaluation]
        MM[metrics.py<br/>Sharpe / Sortino / drawdown / capture]
        ST[statistics.py<br/>PSR / DSR / bootstrap]
        VV[validation.py<br/>walk-forward / purged k-fold]
    end

    OUT[(BacktestResult<br/>equity_curve + trades)]

    AL --> EXP --> AOF
    OH --> AOF
    AOF --> RB
    MC -. enforces .-> RB
    LC -. exercises .-> AOF
    LC -. exercises .-> MC
    TF -. gates .-> AOF
    TF -. gates .-> MC

    SIG --> RULE --> RB
    FM --> RB

    RB --> OUT
    OUT --> MM
    OUT --> ST
    OUT --> VV
```

The temporal firewall (ADR-0002) cuts across the diagram: every arrow that crosses a time-axis boundary either passes through `as_of_fuse` or `MonotonicCursor`, and is guarded by the build-breaking leak test.

---

## How to update these diagrams

When a new top-level container or major component lands:
1. Update the relevant Mermaid diagram above.
2. If the change reflects a decision worth recording, add the corresponding ADR under `docs/adr/`.
3. Keep the bootstrap-phase visual marker (orange `bootstrap` class) on any container whose current implementation is a pre-hardware substitute.
