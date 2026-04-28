"""Synthetic future-leak corpus builder -- the temporal-firewall test fixture.

Master directive section 4.2 mandates a future-leak corpus injected during CI:
"any run that touches it fails the build." This module is the producer side;
the detection side lives in `tests/test_temporal_firewall.py`.

Concept: build a small Alpha Ledger that contains a mix of (a) legitimate
historical rows and (b) POISONED rows whose `epoch_ns` is in the future
relative to a fixed cursor time. Any code path that surfaces a poisoned row
in its output is leaking the future and fails the test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl

POISON_TAG = "poisoned"


@dataclass(frozen=True)
class LeakCorpus:
    """A test fixture pairing legitimate and poisoned rows around a cursor time."""

    bars: pl.DataFrame
    ledger: pl.DataFrame
    cursor_time_ns: int
    poisoned_doc_hashes: frozenset[str]


def _legit_hash(idx: int) -> str:
    return f"sha256:{hashlib.sha256(f'legit-{idx}'.encode()).hexdigest()}"


def _poisoned_hash(idx: int) -> str:
    return f"sha256:{hashlib.sha256(f'{POISON_TAG}-{idx}'.encode()).hexdigest()}"


def build_leak_corpus(
    *,
    n_legit: int = 50,
    n_poisoned: int = 10,
    n_bars: int = 20,
    cursor_offset: int = 5,
) -> LeakCorpus:
    """Build a fixture with `n_legit` past rows and `n_poisoned` future rows.

    The cursor time is set so it lies between the last legit row and the first
    poisoned row. Bars span from before the cursor through the future zone, so
    a leaking implementation that consults future ledger state would surface
    poisoned doc_hashes in the bar-ledger fusion output.
    """
    if n_legit <= 0 or n_poisoned <= 0 or n_bars <= 0:
        raise ValueError("counts must be positive")
    if cursor_offset < 1:
        raise ValueError("cursor_offset must be >= 1")

    cursor_time_ns = n_legit + cursor_offset

    legit_hashes = [_legit_hash(i) for i in range(n_legit)]
    poisoned_hashes = [_poisoned_hash(i) for i in range(n_poisoned)]

    ledger = pl.DataFrame(
        {
            "doc_hash": [*legit_hashes, *poisoned_hashes],
            "epoch_ns": [
                *range(1, n_legit + 1),
                *range(cursor_time_ns + 1, cursor_time_ns + 1 + n_poisoned),
            ],
            "entity": ["AAPL"] * (n_legit + n_poisoned),
            "macro_sentiment": [
                *([0.1] * n_legit),
                *([999.0] * n_poisoned),
            ],
        }
    ).sort("epoch_ns")

    # CRITICAL: bars all live in the legit zone (epoch_ns <= cursor_time_ns).
    # The firewall invariant under test is: at simulation time t, no ledger row
    # with epoch_ns > t may be visible. as_of_join walks backwards from each
    # bar's time, so when bars are bounded by the legit zone, any poisoned row
    # surfacing in the fused output is unambiguously a breach.
    bar_step = max(cursor_time_ns // n_bars, 1)
    bars = pl.DataFrame(
        {
            "ticker": ["AAPL"] * n_bars,
            "epoch_ns": [min((i + 1) * bar_step, cursor_time_ns) for i in range(n_bars)],
            "close": [100.0 + i for i in range(n_bars)],
        }
    ).sort("epoch_ns")

    return LeakCorpus(
        bars=bars,
        ledger=ledger,
        cursor_time_ns=cursor_time_ns,
        poisoned_doc_hashes=frozenset(poisoned_hashes),
    )


def is_leaked(
    df: pl.DataFrame, poisoned: frozenset[str], *, doc_col: str = "doc_hash"
) -> bool:
    """True if any poisoned doc_hash appears in `df`."""
    if doc_col not in df.columns:
        return False
    return df.filter(pl.col(doc_col).is_in(list(poisoned))).height > 0
