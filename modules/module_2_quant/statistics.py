"""Statistical rigor for backtest-comparison claims.

Per master directive section 0.5.1.G. Strategy comparison claims become
research-grade only after they survive selection-bias correction and
out-of-sample validation. The functions here implement:

- Probabilistic Sharpe Ratio (PSR), López de Prado (2012):
  the probability that the *true* Sharpe exceeds a benchmark, given a
  finite sample with optional skew / kurtosis adjustment.
- Deflated Sharpe Ratio (DSR), Bailey & López de Prado (2014):
  PSR adjusted for the selection bias from running N trials. Treats
  the persona-tuning loop in directive section 7 as the trial count.
- Bootstrap confidence intervals on arbitrary metric functions.
- Stationary block bootstrap (Politis & Romano, 1994) for time-series-
  aware resampling.

Why these and not just raw Sharpe: directive section 0.5.1.G calls for
"multiple-hypothesis correction framework (since persona-tuning is a
search procedure prone to overfitting)." Raw Sharpe over-states alpha
when many strategies were tried; DSR and PSR are the corrections.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

EULER_MASCHERONI = 0.5772156649015328606
ArrayLike = np.ndarray | list[float]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n: int,
    *,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR -- probability that the true Sharpe exceeds `benchmark_sr`.

    All Sharpe arguments are ALREADY ANNUALIZED (or all ALREADY de-annualized
    -- units must match). `kurtosis` is the full kurtosis (not excess);
    a normal distribution has kurtosis = 3.0.

    From López de Prado (2012). Returns NaN for n < 2 or degenerate variance.
    """
    if n < 2:
        return float("nan")
    denom_squared = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    if denom_squared <= 0.0:
        return float("nan")
    z = (observed_sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom_squared)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, *, sr_variance: float = 1.0) -> float:
    """Expected maximum of N i.i.d. Sharpe estimates with given variance.

    From the standard extreme-value approximation used in DSR derivation:
        E[max SR] ~ sqrt(var) * ((1-y) * Phi^-1(1 - 1/N) + y * Phi^-1(1 - 1/(N*e)))
    where y is the Euler-Mascheroni constant.
    """
    if n_trials <= 1:
        return 0.0
    if sr_variance < 0.0:
        raise ValueError("sr_variance must be non-negative")
    sigma = math.sqrt(sr_variance)
    # ppf((1 - 1/N)) and ppf((1 - 1/(N*e))) -- both finite for N >= 2.
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)
    z1 = float(norm.ppf(p1))
    z2 = float(norm.ppf(p2))
    return sigma * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    n: int,
    *,
    n_trials: int,
    sr_variance: float = 1.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR -- PSR with the benchmark set to E[max SR] over N trials.

    Use `n_trials` = the number of strategies / personas / parameter sets
    that were evaluated to produce `observed_sr`. `sr_variance` is the
    variance of the SR estimator across those trials (use 1.0 if unknown
    -- gives a conservative deflation).
    """
    benchmark = expected_max_sharpe(n_trials, sr_variance=sr_variance)
    return probabilistic_sharpe_ratio(
        observed_sr, n, benchmark_sr=benchmark, skew=skew, kurtosis=kurtosis
    )


def bonferroni_correction(
    p_values: ArrayLike, *, alpha: float = 0.05
) -> tuple[np.ndarray, float]:
    """Family-wise error rate correction. Most conservative.

    Returns `(significant_mask, corrected_alpha)`. Each test passes iff
    `p_value < alpha / N`. With N tests and uncorrected alpha=0.05,
    the per-test threshold is 0.05/N.
    """
    p = _as_numpy(p_values)
    if p.size == 0:
        return np.zeros(0, dtype=bool), alpha
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    corrected = alpha / p.size
    return p < corrected, corrected


def holm_bonferroni(p_values: ArrayLike, *, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down correction. Less conservative than vanilla Bonferroni
    for the same FWER bound.

    Sort p-values ascending; the rank-k (1-indexed) threshold is
    `alpha / (N - k + 1)`. Reject all p-values up to the first one
    that fails its threshold; everything after is non-significant by
    the step-down rule.
    """
    p = _as_numpy(p_values)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = alpha / (n - np.arange(n))
    significant_sorted = sorted_p < thresholds
    # Once a sorted p-value fails, everything after is non-significant
    failing = ~significant_sorted
    if failing.any():
        first_fail = int(np.argmax(failing))
        significant_sorted[first_fail:] = False
    out = np.zeros(n, dtype=bool)
    out[order] = significant_sorted
    return out


def benjamini_hochberg(p_values: ArrayLike, *, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg false discovery rate control.

    Sort p-values ascending. Find the largest k such that
    `p_(k) <= (k / N) * alpha`. Reject all p-values up to and including
    rank k. FDR-controlled, less conservative than FWER-controlled
    methods (Bonferroni / Holm) at the cost of admitting some
    expected false discoveries.
    """
    p = _as_numpy(p_values)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    significant_sorted = sorted_p <= thresholds
    if not significant_sorted.any():
        return np.zeros(n, dtype=bool)
    k_max = int(np.where(significant_sorted)[0].max())
    final = np.zeros(n, dtype=bool)
    final[: k_max + 1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = final
    return out


def bootstrap_metric(
    arr: ArrayLike,
    metric_fn: Callable[[np.ndarray], float],
    *,
    n_iter: int = 1_000,
    alpha: float = 0.05,
    block_size: int | None = None,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for `metric_fn(arr)`.

    Returns `(lower, point_estimate, upper)` at confidence `1 - alpha`.
    When `block_size` is given, performs stationary block bootstrap
    (Politis & Romano, 1994); otherwise plain i.i.d. resampling.

    `block_size`: pick something like sqrt(n) for serially-correlated
    return series. None for i.i.d. data (e.g. residuals after a model fit).
    """
    a = _as_numpy(arr)
    if a.ndim != 1:
        raise ValueError("arr must be 1-D")
    if a.size < 2:
        raise ValueError("arr must have at least 2 observations")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if n_iter < 100:
        raise ValueError("n_iter should be at least 100 for a meaningful CI")
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)
    n = a.size
    point = float(metric_fn(a))
    estimates = np.empty(n_iter, dtype=float)

    if block_size is None:
        for i in range(n_iter):
            idx = rng.integers(0, n, size=n)
            estimates[i] = metric_fn(a[idx])
    else:
        # Stationary block bootstrap: at each position, advance with prob
        # 1/block_size to start a new block, else continue the current block.
        for i in range(n_iter):
            idx = np.empty(n, dtype=np.int64)
            cur = int(rng.integers(0, n))
            for j in range(n):
                idx[j] = cur
                cur = (
                    int(rng.integers(0, n))
                    if rng.random() < 1.0 / block_size
                    else (cur + 1) % n
                )
            estimates[i] = metric_fn(a[idx])

    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1.0 - alpha / 2))
    return lo, point, hi
