"""Within-ROI combination modes: mean, mean_flip, pca_flip."""
from __future__ import annotations

import numpy as np
import pytest

from source_localization.source_space.roi_combine import (
    DEFAULT_COMBINE_MODE, VALID_COMBINE_MODES, combine_sources, resolve_combine_mode)
from source_localization.source_space.realizations import (
    _parcel_rows, build_roi_operator)


def test_default_is_the_historical_behaviour():
    assert DEFAULT_COMBINE_MODE == "mean"
    X = np.random.default_rng(0).standard_normal((5, 40))
    np.testing.assert_allclose(combine_sources(X), X.mean(axis=0))
    np.testing.assert_allclose(combine_sources(X, "mean"), X.mean(axis=0))


def test_config_resolution_and_validation():
    assert resolve_combine_mode(None) == "mean"
    assert resolve_combine_mode({}) == "mean"
    assert resolve_combine_mode({"roi_extraction": {}}) == "mean"
    assert resolve_combine_mode({"roi_extraction": {"combine": "pca_flip"}}) == "pca_flip"
    with pytest.raises(ValueError, match="Unknown roi_extraction.combine"):
        resolve_combine_mode({"roi_extraction": {"combine": "median"}})


def test_flip_recovers_signal_that_plain_averaging_cancels():
    """Members carrying one source with opposite signs: the mean cancels, the flips do not."""
    rng = np.random.default_rng(1)
    common = rng.standard_normal(500)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    X = signs[:, None] * common[None, :] + 0.05 * rng.standard_normal((5, 500))

    power = {m: np.var(combine_sources(X, m)) for m in VALID_COMBINE_MODES}
    assert power["mean"] < 0.2 * np.var(common)          # cancelled
    assert power["mean_flip"] > 0.8 * np.var(common)     # recovered
    assert power["pca_flip"] > 0.8 * np.var(common)
    # and the recovered course tracks the common source
    for m in ("mean_flip", "pca_flip"):
        assert abs(np.corrcoef(combine_sources(X, m), common)[0, 1]) > 0.99


def test_flip_modes_are_sign_anchored_to_the_plain_mean():
    """Without an anchor the sign is arbitrary, and draws would cancel when summed."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        X = rng.standard_normal((4, 60))
        plain = X.mean(axis=0)
        for m in ("mean_flip", "pca_flip"):
            assert float(combine_sources(X, m) @ plain) >= 0


def test_single_member_and_bad_input():
    X = np.arange(10.0).reshape(1, 10)
    for m in VALID_COMBINE_MODES:
        np.testing.assert_allclose(combine_sources(X, m), X[0])
    with pytest.raises(ValueError, match="2-D"):
        combine_sources(np.zeros(5))
    with pytest.raises(ValueError, match="Unknown combine mode"):
        combine_sources(np.zeros((3, 4)), "median")


def _toy_pool(seed=0, n_pool=90, n_ch=16):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_ch, n_pool))
    pos = rng.standard_normal((n_pool, 3)) * 10
    labels = [f"roi{i % 6}" for i in range(n_pool)]
    return G, pos, labels


def test_parcel_rows_and_operator_accept_combine():
    G, pos, labels = _toy_pool()
    idx = np.sort(np.random.default_rng(4).choice(G.shape[1], 40, replace=False))
    rows = {m: _parcel_rows(G, idx, labels, 1.0 / 9.0, m) for m in VALID_COMBINE_MODES}
    assert set(rows["mean"]) == set(rows["pca_flip"])
    # the default argument keeps the historical result
    base = _parcel_rows(G, idx, labels, 1.0 / 9.0)
    for p in base:
        np.testing.assert_allclose(base[p], rows["mean"][p])

    ops = {m: build_roi_operator(G, pos, labels, n_sources=30, k=5, combine=m)[0]
           for m in VALID_COMBINE_MODES}
    default_op = build_roi_operator(G, pos, labels, n_sources=30, k=5)[0]
    np.testing.assert_allclose(default_op, ops["mean"])
    # flipping changes the operator, but not beyond recognition
    assert not np.allclose(ops["mean"], ops["pca_flip"])
    for m in ("mean_flip", "pca_flip"):
        cos = [float(ops[m][i] @ ops["mean"][i] /
                     (np.linalg.norm(ops[m][i]) * np.linalg.norm(ops["mean"][i])))
               for i in range(ops["mean"].shape[0])]
        assert min(cos) > 0, f"{m} produced a negated parcel row"
