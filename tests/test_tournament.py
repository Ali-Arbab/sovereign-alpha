"""Tests for the persona-tournament harness."""

from __future__ import annotations

import math
from datetime import date

import pytest

from modules.module_2_quant.tournament import (
    TournamentWindow,
    WindowResult,
    make_synthetic_runner,
    rolling_windows,
    run_tournament,
)


def _trivial_window() -> TournamentWindow:
    return TournamentWindow(start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))


def _wr(window: TournamentWindow, persona_id: str, sharpe: float) -> WindowResult:
    return WindowResult(
        window=window,
        persona_id=persona_id,
        sharpe=sharpe,
        sortino=sharpe,
        max_drawdown=-0.1,
        final_equity=100_000.0,
        n_trades=10,
    )


# --- rolling_windows ----------------------------------------------------


def test_rolling_windows_basic() -> None:
    ws = rolling_windows(date(2020, 1, 1), date(2020, 12, 31), window_days=180, step_days=30)
    assert all(isinstance(w, TournamentWindow) for w in ws)
    # First window: Jan 1 -> Jun 28 (180 days inclusive)
    assert ws[0].start_date == date(2020, 1, 1)
    # Each successive window starts step_days later
    assert (ws[1].start_date - ws[0].start_date).days == 30


def test_rolling_windows_last_window_within_range() -> None:
    ws = rolling_windows(date(2020, 1, 1), date(2020, 12, 31), window_days=180)
    assert ws[-1].end_date <= date(2020, 12, 31)


def test_rolling_windows_invalid_args() -> None:
    with pytest.raises(ValueError, match="start"):
        rolling_windows(date(2020, 6, 1), date(2020, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        rolling_windows(date(2020, 1, 1), date(2020, 12, 31), window_days=0)
    with pytest.raises(ValueError, match="positive"):
        rolling_windows(date(2020, 1, 1), date(2020, 12, 31), step_days=0)


def test_rolling_windows_inverts_when_window_too_long() -> None:
    """A window longer than the [start, end] span yields zero windows."""
    ws = rolling_windows(date(2020, 1, 1), date(2020, 1, 10), window_days=30)
    assert ws == []


# --- TournamentWindow ---------------------------------------------------


def test_tournament_window_rejects_inverted_dates() -> None:
    with pytest.raises(ValueError, match="start_date"):
        TournamentWindow(start_date=date(2024, 6, 1), end_date=date(2024, 1, 1))


def test_tournament_window_label_is_human_readable() -> None:
    w = _trivial_window()
    assert w.label() == "2024-01-01_2024-06-30"


# --- run_tournament -----------------------------------------------------


def test_run_tournament_ranks_personas_by_sharpe() -> None:
    """Stub runner returns deterministic Sharpes; ranking must reflect them."""
    window = _trivial_window()
    sharpes = {"good_v1": 1.5, "mid_v1": 0.5, "bad_v1": -0.5}

    def stub(persona_id: str, w: TournamentWindow) -> WindowResult:
        return _wr(w, persona_id, sharpes[persona_id])

    report = run_tournament(
        persona_ids=list(sharpes.keys()),
        windows=[window],
        runner=stub,
        metric="sharpe",
        elimination_fraction=0.5,  # eliminate bottom half
    )
    assert report.rankings[window.label()] == ["good_v1", "mid_v1", "bad_v1"]
    # Top half (rounded up) survives -> ceil(3*0.5)=2 keep -> good_v1, mid_v1.
    assert set(report.survivors) == {"good_v1", "mid_v1"}


def test_run_tournament_survivors_are_only_those_who_survive_every_window() -> None:
    w1 = TournamentWindow(date(2024, 1, 1), date(2024, 3, 31))
    w2 = TournamentWindow(date(2024, 4, 1), date(2024, 6, 30))

    def stub(persona_id: str, w: TournamentWindow) -> WindowResult:
        # persona_a is best in w1, worst in w2; persona_b is consistent middle;
        # persona_c is worst in w1, best in w2.
        sharpe = {
            ("persona_a", w1.label()): 2.0,
            ("persona_a", w2.label()): -1.0,
            ("persona_b", w1.label()): 1.0,
            ("persona_b", w2.label()): 1.0,
            ("persona_c", w1.label()): -1.0,
            ("persona_c", w2.label()): 2.0,
        }[(persona_id, w.label())]
        return _wr(w, persona_id, sharpe)

    report = run_tournament(
        persona_ids=["persona_a", "persona_b", "persona_c"],
        windows=[w1, w2],
        runner=stub,
        metric="sharpe",
        elimination_fraction=0.34,  # cut the bottom 34% -> 1 of 3 each window
    )
    # In each window only one persona is eliminated. Persona_b is never the
    # absolute worst, so it survives everything. a and c each lose once.
    assert "persona_b" in report.survivors
    assert "persona_a" not in report.survivors
    assert "persona_c" not in report.survivors


def test_run_tournament_handles_nan_metric() -> None:
    """A NaN Sharpe should sort to the bottom, not crash."""
    window = _trivial_window()

    def stub(persona_id: str, w: TournamentWindow) -> WindowResult:
        sharpe = math.nan if persona_id == "broken_v1" else 1.0
        return _wr(w, persona_id, sharpe)

    report = run_tournament(
        persona_ids=["good_v1", "broken_v1"],
        windows=[window],
        runner=stub,
        metric="sharpe",
        elimination_fraction=0.5,
    )
    # broken_v1 should rank last
    assert report.rankings[window.label()][-1] == "broken_v1"


def test_run_tournament_results_for_returns_per_persona_subset() -> None:
    window = _trivial_window()

    def stub(persona_id: str, w: TournamentWindow) -> WindowResult:
        return _wr(w, persona_id, 1.0)

    report = run_tournament(
        persona_ids=["a_v1", "b_v1"],
        windows=[window],
        runner=stub,
        metric="sharpe",
    )
    assert len(report.results_for("a_v1")) == 1
    assert len(report.results_for("nope")) == 0


def test_run_tournament_ranks_by_max_drawdown() -> None:
    """max_drawdown is stored negative; descending sort naturally puts the
    less-negative (safer) drawdown first."""
    window = _trivial_window()

    def stub(persona_id: str, w: TournamentWindow) -> WindowResult:
        dd = {"safe_v1": -0.05, "risky_v1": -0.40}[persona_id]
        return WindowResult(
            window=w,
            persona_id=persona_id,
            sharpe=1.0,
            sortino=1.0,
            max_drawdown=dd,
            final_equity=100_000.0,
            n_trades=5,
        )

    report = run_tournament(
        persona_ids=["safe_v1", "risky_v1"],
        windows=[window],
        runner=stub,
        metric="max_drawdown",
    )
    assert report.rankings[window.label()][0] == "safe_v1"


def test_run_tournament_invalid_args() -> None:
    window = _trivial_window()
    runner = lambda p, w: _wr(w, p, 1.0)  # noqa: E731
    with pytest.raises(ValueError, match="persona_ids"):
        run_tournament(persona_ids=[], windows=[window], runner=runner)
    with pytest.raises(ValueError, match="windows"):
        run_tournament(persona_ids=["a_v1"], windows=[], runner=runner)
    with pytest.raises(ValueError, match="elimination_fraction"):
        run_tournament(
            persona_ids=["a_v1"],
            windows=[window],
            runner=runner,
            elimination_fraction=0.0,
        )
    with pytest.raises(ValueError, match="elimination_fraction"):
        run_tournament(
            persona_ids=["a_v1"],
            windows=[window],
            runner=runner,
            elimination_fraction=1.0,
        )
    with pytest.raises(ValueError, match="metric"):
        run_tournament(
            persona_ids=["a_v1"],
            windows=[window],
            runner=runner,
            metric="not_a_metric",
        )


# --- synthetic runner end-to-end ----------------------------------------


def test_make_synthetic_runner_e2e_produces_window_results() -> None:
    """The synthetic runner drives an actual backtest -- this is a real
    end-to-end smoke that the tournament harness composes correctly."""
    runner = make_synthetic_runner(bars_per_day=10, docs_per_day_mean=15.0)
    window = TournamentWindow(start_date=date(2024, 1, 2), end_date=date(2024, 1, 12))
    result = runner("persona_alpha_v1", window)
    assert isinstance(result, WindowResult)
    assert result.persona_id == "persona_alpha_v1"
    assert result.window == window
    # Even on a short window with few trades, equity must be a finite positive number
    assert math.isfinite(result.final_equity)
    assert result.final_equity > 0


def test_synthetic_runner_distinct_personas_yield_distinct_ledgers() -> None:
    """Different persona_ids must seed distinct sentiment distributions, so
    metrics differ between personas. Otherwise the tournament is degenerate."""
    runner = make_synthetic_runner(bars_per_day=10, docs_per_day_mean=15.0)
    window = TournamentWindow(start_date=date(2024, 1, 2), end_date=date(2024, 1, 12))
    a = runner("persona_alpha_v1", window)
    b = runner("persona_beta_v1", window)
    # At least one metric must differ -- with different ledger seeds, the
    # backtests will diverge.
    differs = (
        a.final_equity != b.final_equity
        or a.n_trades != b.n_trades
        or a.sharpe != b.sharpe
    )
    assert differs, "synthetic runner produced identical results for two personas"
