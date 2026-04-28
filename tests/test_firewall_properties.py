"""Property-based tests for the temporal firewall.

Per master directive section 0.5.1.E: "Property-based tests on
temporal-firewall preservation (Hypothesis library)."

The example-based tests in test_temporal_firewall.py exercise the firewall
on a fixed leak corpus. The properties here exercise it on Hypothesis-
generated random schedules of bars and ledger rows, so any seed that
produces a leak fails the build with a minimized counter-example.

Three invariants:
1. Cursor: `advance_to(t)` returns ONLY rows with `epoch_ns <= t`.
2. Cursor: advancing strictly forward, snapshots are subset-monotonic.
3. as_of_fuse: every attached ledger row has `epoch_ns <= bar.epoch_ns`.
"""

from __future__ import annotations

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from modules.module_2_quant.cursor import MonotonicCursor
from modules.module_2_quant.fusion import as_of_fuse

EPOCH_INTS = st.integers(min_value=0, max_value=10**12)


def _sorted_unique_epochs(min_size: int, max_size: int) -> st.SearchStrategy[list[int]]:
    return st.lists(EPOCH_INTS, min_size=min_size, max_size=max_size, unique=True).map(
        sorted
    )


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(epochs=_sorted_unique_epochs(1, 50), cutoff=EPOCH_INTS)
def test_cursor_advance_returns_only_past_rows(epochs: list[int], cutoff: int) -> None:
    df = pl.DataFrame({"epoch_ns": epochs, "v": list(range(len(epochs)))})
    cursor = MonotonicCursor(df)
    snap = cursor.advance_to(cutoff)
    if snap.height == 0:
        return
    assert snap["epoch_ns"].max() <= cutoff, (
        "TEMPORAL FIREWALL BREACH: cursor surfaced epoch_ns > cutoff"
    )


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    epochs=_sorted_unique_epochs(2, 50),
    cutoffs=st.lists(EPOCH_INTS, min_size=2, max_size=10).map(sorted),
)
def test_cursor_forward_advances_are_monotonic_in_visible_set(
    epochs: list[int], cutoffs: list[int]
) -> None:
    """Advancing forward must never SHRINK the set of visible rows."""
    df = pl.DataFrame({"epoch_ns": epochs, "v": list(range(len(epochs)))})
    cursor = MonotonicCursor(df)
    prev_height = -1
    for t in cutoffs:
        snap = cursor.advance_to(t)
        assert snap.height >= prev_height, (
            "TEMPORAL FIREWALL BREACH: forward advance shrank the visible set"
        )
        prev_height = snap.height


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    bar_epochs=_sorted_unique_epochs(1, 30),
    ledger_epochs=_sorted_unique_epochs(1, 30),
)
def test_as_of_fuse_attached_ledger_epoch_never_exceeds_bar_epoch(
    bar_epochs: list[int], ledger_epochs: list[int]
) -> None:
    """For every fused row, the matched ledger row's marker must be <= the
    bar's epoch. We add a ledger-side marker column (`ledger_epoch_marker`)
    so we can recover which ledger row was attached after the join."""
    n_bars = len(bar_epochs)
    n_ledger = len(ledger_epochs)
    bars = pl.DataFrame(
        {
            "ticker": ["AAPL"] * n_bars,
            "epoch_ns": bar_epochs,
            "close": [100.0 + i for i in range(n_bars)],
        }
    )
    ledger = pl.DataFrame(
        {
            "entity": ["AAPL"] * n_ledger,
            "epoch_ns": ledger_epochs,
            "ledger_epoch_marker": ledger_epochs,
        }
    )
    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
    for bar_epoch, marker in zip(
        fused["epoch_ns"].to_list(),
        fused["ledger_epoch_marker"].to_list(),
        strict=True,
    ):
        if marker is None:
            continue
        assert marker <= bar_epoch, (
            f"TEMPORAL FIREWALL BREACH: ledger row at {marker} attached to "
            f"bar at {bar_epoch} (ledger > bar)"
        )


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    bar_epochs=_sorted_unique_epochs(1, 20),
    ledger_epochs=_sorted_unique_epochs(1, 20),
)
def test_as_of_fuse_height_equals_bars_height(
    bar_epochs: list[int], ledger_epochs: list[int]
) -> None:
    """as_of_fuse is a left-style merge -- every bar stays. Even when no
    ledger row matches, the bar should appear with NULLs filled in."""
    bars = pl.DataFrame(
        {
            "ticker": ["AAPL"] * len(bar_epochs),
            "epoch_ns": bar_epochs,
            "close": [100.0 + i for i in range(len(bar_epochs))],
        }
    )
    ledger = pl.DataFrame(
        {
            "entity": ["AAPL"] * len(ledger_epochs),
            "epoch_ns": ledger_epochs,
            "macro_sentiment": [0.0 for _ in ledger_epochs],
        }
    )
    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
    assert fused.height == bars.height, (
        "as_of_fuse dropped or duplicated a bar -- not a left-style merge"
    )
