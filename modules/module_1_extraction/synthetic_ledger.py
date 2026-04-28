"""Synthetic Alpha Ledger generator -- bootstrap phase only.

Produces statistically plausible AlphaLedgerRecord rows that drive Module II
development end-to-end without waiting on Module I LLM inference (master
directive section 0.5). Every row is tagged `persona_id="bootstrap_synthetic_v1"`
and `model_id="synthetic"` so it cannot be confused with real research output.

Distributions (deliberately simple -- this is plumbing, not research):
- Per-ticker sentiment: Ornstein-Uhlenbeck-style mean-reverting walk in [-1, 1].
- Sector sentiment: macro sentiment plus Gaussian jitter, clipped.
- Confidence score: Beta(8, 2) -- skewed toward high confidence.
- Confidence interval: symmetric uniform half-width around confidence_score.
- Regime shift flag: Bernoulli(0.005) -- rare.
- Horizon days: uniform over (7, 30, 90, 180).
- Document timestamps: random hour/minute within UTC day.

Determinism: seeded numpy.random.default_rng. Same seed plus same range plus
same ticker dict -> byte-identical output across runs.

Output: Hive-style partitioned Parquet at `out_dir/year=YYYY/month=MM/part-0.parquet`.
Within each partition, rows are sorted ascending on `epoch_ns` so a downstream
`as_of_join` cursor sees a monotonic timeline.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.random import Generator

from modules.module_1_extraction.bootstrap_universe import DEFAULT_BOOTSTRAP_UNIVERSE
from shared.schemas.alpha_ledger import SCHEMA_VERSION

PERSONA_ID: Final[str] = "bootstrap_synthetic_v1"
MODEL_ID: Final[str] = "synthetic"
HORIZON_CHOICES: Final[tuple[int, ...]] = (7, 30, 90, 180)
REGIME_SHIFT_PROB: Final[float] = 0.005

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "doc_hash",
    "timestamp",
    "epoch_ns",
    "entities",
    "sector_tags",
    "macro_sentiment",
    "sector_sentiment",
    "confidence_interval",
    "confidence_score",
    "regime_shift_flag",
    "horizon_days",
    "reasoning_trace",
    "persona_id",
    "model_id",
    "schema_version",
)


def _doc_hash(seed: int, day_idx: int, doc_idx: int) -> str:
    payload = f"bootstrap-synthetic-{seed}-{day_idx}-{doc_idx}".encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _ou_walk(rng: Generator, n: int, theta: float = 0.05, sigma: float = 0.15) -> np.ndarray:
    """Mean-reverting walk in [-1, 1] starting at 0."""
    if n <= 0:
        return np.zeros(0)
    noise = sigma * rng.standard_normal(n)
    x = np.zeros(n)
    decay = 1.0 - theta
    for i in range(1, n):
        x[i] = decay * x[i - 1] + noise[i]
    return np.clip(x, -1.0, 1.0)


def _epoch_ns(ts: datetime) -> int:
    """Integer nanoseconds since Unix epoch -- avoids float precision loss."""
    delta = ts - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def generate_synthetic_ledger(
    out_dir: Path,
    start_date: date,
    end_date: date,
    *,
    tickers: dict[str, str] | None = None,
    seed: int = 0,
    docs_per_day_mean: float = 50.0,
) -> list[Path]:
    """Generate synthetic Alpha Ledger rows over [start_date, end_date] inclusive.

    Writes Parquet files at `{out_dir}/year=YYYY/month=MM/part-0.parquet`.
    Returns the list of written file paths in ascending (year, month) order.

    Raises ValueError on inverted dates, non-positive `docs_per_day_mean`,
    or empty `tickers` mapping.
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if docs_per_day_mean <= 0:
        raise ValueError("docs_per_day_mean must be positive")
    if tickers is None:
        tickers = DEFAULT_BOOTSTRAP_UNIVERSE
    if not tickers:
        raise ValueError("tickers must be a non-empty mapping")

    rng = np.random.default_rng(seed)
    ticker_list = list(tickers.keys())
    n_tickers = len(ticker_list)
    n_days = (end_date - start_date).days + 1

    trajectories = np.stack([_ou_walk(rng, n_days) for _ in range(n_tickers)], axis=0)

    rows: list[dict] = []
    for day_idx in range(n_days):
        current = start_date + timedelta(days=day_idx)
        n_docs = int(rng.poisson(docs_per_day_mean))
        for doc_idx in range(n_docs):
            ti = int(rng.integers(0, n_tickers))
            entity = ticker_list[ti]
            sector = tickers[entity]

            macro_s = float(np.clip(trajectories[ti, day_idx] + rng.normal(0, 0.05), -1.0, 1.0))
            sector_s = float(np.clip(macro_s + rng.normal(0, 0.10), -1.0, 1.0))

            conf = float(rng.beta(8.0, 2.0))
            half_width = float(rng.uniform(0.02, 0.10))
            ci_low = float(max(0.0, conf - half_width))
            ci_high = float(min(1.0, conf + half_width))

            regime = bool(rng.random() < REGIME_SHIFT_PROB)
            horizon = int(rng.choice(HORIZON_CHOICES))

            hour = int(rng.integers(0, 24))
            minute = int(rng.integers(0, 60))
            ts = datetime.combine(current, time(hour=hour, minute=minute, tzinfo=UTC))

            rows.append(
                {
                    "doc_hash": _doc_hash(seed, day_idx, doc_idx),
                    "timestamp": ts.isoformat().replace("+00:00", "Z"),
                    "epoch_ns": _epoch_ns(ts),
                    "entities": [entity],
                    "sector_tags": [sector],
                    "macro_sentiment": macro_s,
                    "sector_sentiment": sector_s,
                    "confidence_interval": [ci_low, ci_high],
                    "confidence_score": conf,
                    "regime_shift_flag": regime,
                    "horizon_days": horizon,
                    "reasoning_trace": (
                        f"[synthetic] {entity} {sector} day={day_idx} doc={doc_idx}"
                    ),
                    "persona_id": PERSONA_ID,
                    "model_id": MODEL_ID,
                    "schema_version": SCHEMA_VERSION,
                    "_year": current.year,
                    "_month": current.month,
                }
            )

    out_dir = Path(out_dir)
    if not rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        return []

    df = pl.DataFrame(rows).select([*_COLUMN_ORDER, "_year", "_month"]).sort("epoch_ns")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sub in df.partition_by(["_year", "_month"], maintain_order=True):
        year = int(sub["_year"][0])
        month = int(sub["_month"][0])
        part_dir = out_dir / f"year={year}" / f"month={month:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "part-0.parquet"
        sub.drop(["_year", "_month"]).write_parquet(out_path)
        written.append(out_path)

    return sorted(written)
