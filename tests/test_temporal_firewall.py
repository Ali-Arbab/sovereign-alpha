"""The temporal firewall -- directive section 4.2's build-breaking leak test.

Any commit that allows future Alpha Ledger rows to surface in past-state
Module II output fails this test. Failing this test fails the build. This
is the single most important test in the entire repo.
"""

from __future__ import annotations

import polars as pl
import pytest

from modules.module_2_quant.cursor import MonotonicCursor
from modules.module_2_quant.fusion import as_of_fuse
from tests.leak_corpus import build_leak_corpus, is_leaked


def test_as_of_fuse_does_not_leak_future_ledger_rows() -> None:
    corpus = build_leak_corpus()
    fused = as_of_fuse(
        corpus.bars,
        corpus.ledger,
        on="epoch_ns",
        by_left="ticker",
        by_right="entity",
    )
    assert not is_leaked(fused, corpus.poisoned_doc_hashes), (
        "TEMPORAL FIREWALL BREACH: as_of_fuse surfaced a poisoned (future) "
        "doc_hash. Directive section 4.2 invariant violated."
    )
    sentinels = fused["macro_sentiment"].to_list()
    assert all(s is None or abs(s - 0.1) < 1e-9 for s in sentinels), (
        "TEMPORAL FIREWALL BREACH: poison sentinel (999.0) surfaced in fused output."
    )
    # Anti-no-op: ensure the fixture actually exercises the join path. Without
    # this, the assertions above could pass trivially if no ledger row matched
    # any bar.
    non_null = [s for s in sentinels if s is not None]
    assert len(non_null) > 0, (
        "Leak-test no-op: as_of_fuse attached zero ledger rows. The firewall "
        "would pass trivially. Fix the fixture before trusting this test."
    )


def test_cursor_advance_to_does_not_leak_future() -> None:
    corpus = build_leak_corpus()
    cursor = MonotonicCursor(corpus.ledger, on="epoch_ns")
    snap = cursor.advance_to(corpus.cursor_time_ns)
    assert not is_leaked(snap, corpus.poisoned_doc_hashes), (
        "TEMPORAL FIREWALL BREACH: cursor.advance_to(cursor_time) returned a "
        "poisoned future row. Directive section 6.4 invariant violated."
    )
    assert snap["epoch_ns"].max() <= corpus.cursor_time_ns


def test_cursor_rejects_backwards_movement() -> None:
    corpus = build_leak_corpus()
    cursor = MonotonicCursor(corpus.ledger)
    cursor.advance_to(corpus.cursor_time_ns)
    with pytest.raises(ValueError, match="backwards"):
        cursor.advance_to(corpus.cursor_time_ns - 1)


def test_unsorted_input_to_as_of_fuse_is_rejected() -> None:
    bars = pl.DataFrame(
        {"ticker": ["AAPL", "AAPL"], "epoch_ns": [10, 20], "close": [100.0, 101.0]}
    )
    bad_ledger = pl.DataFrame(
        {
            "entity": ["AAPL", "AAPL"],
            "epoch_ns": [20, 5],
            "doc_hash": ["sha256:a", "sha256:b"],
            "macro_sentiment": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        as_of_fuse(bars, bad_ledger, by_left="ticker", by_right="entity")


def test_proof_of_construction_corpus_is_actually_poisoned() -> None:
    """Meta-test: the leak corpus itself contains poisoned rows that, if NOT
    filtered, would leak. This guards against the firewall test becoming a
    silent no-op if the leak-corpus generator ever stops injecting poison."""
    corpus = build_leak_corpus()
    assert len(corpus.poisoned_doc_hashes) > 0
    assert is_leaked(corpus.ledger, corpus.poisoned_doc_hashes), (
        "Leak corpus is not actually poisoned -- the firewall test would be "
        "a silent no-op. Fix `build_leak_corpus` before doing anything else."
    )
