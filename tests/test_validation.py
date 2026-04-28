"""Tests for Module II out-of-sample validation -- walk-forward + purged k-fold."""

from __future__ import annotations

import numpy as np
import pytest

from modules.module_2_quant.validation import (
    CVFold,
    purged_kfold_splits,
    walk_forward_splits,
)


def test_walk_forward_basic_shape() -> None:
    folds = list(walk_forward_splits(n=100, initial_window=20, test_window=10))
    # train_end starts at 20; advances by 10; (100 - 20) // 10 = 8 folds.
    assert len(folds) == 8
    for f in folds:
        assert f.train.shape == (100,)
        assert f.test.shape == (100,)


def test_walk_forward_train_strictly_before_test() -> None:
    for f in walk_forward_splits(n=50, initial_window=10, test_window=5):
        last_train = np.where(f.train)[0].max() if f.n_train else -1
        first_test = np.where(f.test)[0].min()
        assert last_train < first_test, "train extends into or past test (leak)"


def test_walk_forward_train_grows_monotonically() -> None:
    folds = list(walk_forward_splits(n=100, initial_window=10, test_window=5))
    sizes = [f.n_train for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] == 10


def test_walk_forward_overlapping_step_smaller_than_test() -> None:
    folds = list(walk_forward_splits(n=50, initial_window=10, test_window=10, step=5))
    # Test windows may overlap when step < test_window; folds still well-formed.
    assert len(folds) >= 2
    for f in folds:
        assert f.n_test == 10


def test_walk_forward_invalid_args() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        list(walk_forward_splits(n=0, initial_window=10, test_window=5))
    with pytest.raises(ValueError, match="must be positive"):
        list(walk_forward_splits(n=10, initial_window=0, test_window=5))
    with pytest.raises(ValueError, match="exceeds n"):
        list(walk_forward_splits(n=10, initial_window=10, test_window=5))
    with pytest.raises(ValueError, match="step must be positive"):
        list(walk_forward_splits(n=50, initial_window=10, test_window=5, step=0))


def test_purged_kfold_returns_n_splits_folds() -> None:
    folds = list(purged_kfold_splits(n=100, n_splits=5))
    assert len(folds) == 5


def test_purged_kfold_test_folds_partition_the_range() -> None:
    """With no purge / embargo, test masks are disjoint and cover [0, n)."""
    folds = list(purged_kfold_splits(n=100, n_splits=5))
    union = np.zeros(100, dtype=bool)
    for f in folds:
        # Test sets must be pairwise disjoint
        assert not (union & f.test).any()
        union |= f.test
    assert union.all(), "test folds do not cover the full range"


def test_purged_kfold_train_excludes_purge_window() -> None:
    folds = list(purged_kfold_splits(n=100, n_splits=5, purge_length=3))
    for k, f in enumerate(folds):
        if k == 0:
            continue  # nothing to purge before the first fold
        test_start = np.where(f.test)[0].min()
        # Indices [test_start - 3, test_start) must be excluded from train
        assert not f.train[test_start - 3 : test_start].any(), (
            f"fold {k} did not purge {3} pre-test samples"
        )


def test_purged_kfold_train_excludes_embargo_window() -> None:
    folds = list(purged_kfold_splits(n=100, n_splits=5, embargo_length=4))
    for k, f in enumerate(folds[:-1]):  # last fold has no after-region
        test_end = np.where(f.test)[0].max() + 1
        assert not f.train[test_end : test_end + 4].any(), (
            f"fold {k} did not embargo {4} post-test samples"
        )


def test_purged_kfold_last_fold_absorbs_remainder() -> None:
    """When n is not divisible by n_splits, the last fold should pick up the
    remainder so test masks still cover [0, n)."""
    folds = list(purged_kfold_splits(n=103, n_splits=5))
    union = np.zeros(103, dtype=bool)
    for f in folds:
        union |= f.test
    assert union.all()
    assert folds[-1].n_test == 103 - 4 * (103 // 5)  # 103 - 4*20 = 23


def test_purged_kfold_invalid_args() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        list(purged_kfold_splits(n=100, n_splits=1))
    with pytest.raises(ValueError, match=r"n .* must be >= n_splits"):
        list(purged_kfold_splits(n=3, n_splits=5))
    with pytest.raises(ValueError, match="non-negative"):
        list(purged_kfold_splits(n=100, n_splits=5, purge_length=-1))


def test_cvfold_rejects_overlapping_masks() -> None:
    train = np.array([True, True, False])
    test = np.array([False, True, True])
    with pytest.raises(ValueError, match="overlap"):
        CVFold(train=train, test=test)


def test_cvfold_rejects_mismatched_shapes() -> None:
    train = np.array([True, False])
    test = np.array([False, True, False])
    with pytest.raises(ValueError, match="same shape"):
        CVFold(train=train, test=test)


def test_cvfold_rejects_non_1d() -> None:
    train = np.zeros((2, 2), dtype=bool)
    test = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="1-D"):
        CVFold(train=train, test=test)
