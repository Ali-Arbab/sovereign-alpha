"""Out-of-sample validation -- walk-forward and purged k-fold CV.

Per master directive section 0.5.1.G. Standard k-fold cross-validation
leaks information when applied to time series: a sample's label may use
data drawn from periods that appear in another fold's training window.
The two splitters here respect time order and provide explicit purge /
embargo buffers to drop samples whose label horizon overlaps the test
fold.

Reference: López de Prado, "Advances in Financial Machine Learning,"
chapter 7. The implementations here are the index-only mask form;
horizon-aware purging is a follow-up commit once Module II strategies
declare their per-row label horizons.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CVFold:
    """One fold of a CV split, expressed as boolean masks over a 0..n range."""

    train: np.ndarray  # bool mask, length n, True for train samples
    test: np.ndarray  # bool mask, length n, True for test samples

    def __post_init__(self) -> None:
        if self.train.shape != self.test.shape:
            raise ValueError("train and test masks must be the same shape")
        if self.train.ndim != 1:
            raise ValueError("masks must be 1-D")
        # Train and test must be disjoint
        if (self.train & self.test).any():
            raise ValueError("train and test masks overlap")

    @property
    def n_train(self) -> int:
        return int(self.train.sum())

    @property
    def n_test(self) -> int:
        return int(self.test.sum())


def walk_forward_splits(
    n: int,
    *,
    initial_window: int,
    test_window: int,
    step: int | None = None,
) -> Iterator[CVFold]:
    """Walk-forward expanding-window CV.

    For each fold:
      - Train indices = [0, train_end)
      - Test indices  = [train_end, train_end + test_window)
      - train_end advances by `step` (default = test_window so test folds
        are non-overlapping).

    The first fold has train_end = initial_window; the iteration ends when
    train_end + test_window would exceed n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if initial_window <= 0 or test_window <= 0:
        raise ValueError("initial_window and test_window must be positive")
    if initial_window + test_window > n:
        raise ValueError(
            f"initial_window + test_window ({initial_window + test_window}) "
            f"exceeds n ({n}); no folds possible"
        )
    s = step if step is not None else test_window
    if s <= 0:
        raise ValueError("step must be positive")

    train_end = initial_window
    while train_end + test_window <= n:
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:train_end] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[train_end : train_end + test_window] = True
        yield CVFold(train=train_mask, test=test_mask)
        train_end += s


def purged_kfold_splits(
    n: int,
    *,
    n_splits: int = 5,
    purge_length: int = 0,
    embargo_length: int = 0,
) -> Iterator[CVFold]:
    """Purged k-fold CV with embargo.

    Test folds are contiguous time slices of the 0..n range. Training
    samples in the `purge_length` window BEFORE the test fold and the
    `embargo_length` window AFTER are excluded to prevent label-horizon
    leakage and serial-correlation leakage respectively.

    Use `purge_length` >= max label horizon (in observations) for your
    strategy; use `embargo_length` >= the AR(1) decorrelation length
    of your residuals.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n < n_splits:
        raise ValueError(f"n ({n}) must be >= n_splits ({n_splits})")
    if purge_length < 0 or embargo_length < 0:
        raise ValueError("purge_length and embargo_length must be non-negative")

    fold_size = n // n_splits
    for k in range(n_splits):
        test_start = k * fold_size
        # Last fold absorbs any remainder so we cover the full range
        test_end = n if k == n_splits - 1 else (k + 1) * fold_size

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_start:test_end] = False
        # Purge training samples just before the test fold
        if purge_length > 0:
            purge_start = max(0, test_start - purge_length)
            train_mask[purge_start:test_start] = False
        # Embargo training samples just after the test fold
        if embargo_length > 0:
            embargo_end = min(n, test_end + embargo_length)
            train_mask[test_end:embargo_end] = False

        yield CVFold(train=train_mask, test=test_mask)
