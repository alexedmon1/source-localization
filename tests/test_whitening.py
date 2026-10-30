"""Tests for the whitener candidates and their held-out selection (validation/whitening.py, Phase 2 decision).

Synthetic noise with a known covariance stands in for the recordings: each "recording" draws noise
from one shared shape with its own scale, as the whitener assumes. They pin that the plain sample
covariance amplifies held-out noise when samples are few (the Phase 2 blocker) and shrinkage does
not; that every consistent estimator whitens held-out noise to unit power when data are ample; that
the held-out log-likelihood picks the estimator family of the true covariance (correlated ->
not diagonal; per-electrode independent -> diagonal); and the selection rule itself.
"""
import numpy as np
import pytest

from source_localization.validation.whitening import (NoiseWhitener, average_reference_basis, choose_whitener, estimate_covariance,
                          gaussian_loglik, heldout_scores)

N = 30


def correlated_L(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, 10))
    return np.linalg.cholesky(A @ A.T / 10 + 0.3 * np.eye(N))


def independent_L(seed=0):
    return np.diag(np.random.default_rng(seed).uniform(0.3, 3.0, N))


def recordings(rng, L, n_rec, cols):
    return [rng.uniform(0.5, 2.0) * L @ rng.standard_normal((N, cols)) for _ in range(n_rec)]


def test_sample_covariance_amplifies_heldout_noise_and_shrinkage_does_not():
    rng = np.random.default_rng(1)
    L = correlated_L()
    train, test = recordings(rng, L, 32, 1), recordings(rng, L, 400, 1)
    assert heldout_scores(train, test, "sample")["power"] > 5
    for est in ("ledoit_wolf", "oas"):
        assert 0.5 < heldout_scores(train, test, est)["power"] < 2


def test_with_ample_data_heldout_power_is_one():
    rng = np.random.default_rng(2)
    L = correlated_L()
    train, test = recordings(rng, L, 64, 20), recordings(rng, L, 400, 1)
    for est in ("sample", "ledoit_wolf", "oas"):
        assert heldout_scores(train, test, est)["power"] == pytest.approx(1.0, abs=0.15)


def test_loglik_rejects_diagonal_for_correlated_noise():
    rng = np.random.default_rng(3)
    L = correlated_L()
    train, test = recordings(rng, L, 64, 4), recordings(rng, L, 200, 1)
    ll = {e: heldout_scores(train, test, e)["loglik"] for e in ("sample", "ledoit_wolf", "oas", "diagonal")}
    assert max(ll, key=ll.get) != "diagonal"
    assert ll["diagonal"] < min(ll["ledoit_wolf"], ll["oas"]) - 3


def test_loglik_picks_diagonal_for_independent_electrodes():
    rng = np.random.default_rng(4)
    L = independent_L()
    train, test = recordings(rng, L, 32, 4), recordings(rng, L, 400, 1)
    ll = {e: heldout_scores(train, test, e)["loglik"] for e in ("sample", "ledoit_wolf", "oas", "diagonal")}
    assert max(ll, key=ll.get) == "diagonal"


def test_diagonal_is_independent_electrodes_then_referenced():
    rng = np.random.default_rng(5)
    d = rng.uniform(0.5, 2.0, N)
    B = average_reference_basis(N)
    X = B.T @ (np.sqrt(d)[:, None] * rng.standard_normal((N, 200000)))
    C = estimate_covariance(X, "diagonal", B)
    # E[(B X)_e^2] = (H D H)_ee with H the centering matrix: the referenced per-electrode variance
    H = np.eye(N) - 1.0 / N
    d_ref = np.diag(H @ np.diag(d) @ H)
    assert np.allclose(C, B.T @ np.diag(np.mean((B @ X) ** 2, axis=1)) @ B)        # the definition
    assert np.allclose(C, B.T @ np.diag(d_ref) @ B, atol=0.03)                    # and its expectation
    assert np.allclose(C, C.T)


def test_default_fit_is_phase1_exactly():
    rng = np.random.default_rng(6)
    noise = recordings(rng, correlated_L(), 40, 2)
    B = average_reference_basis(N)
    Nz = [B.T @ z for z in noise]
    s = [np.sqrt(np.sum(z ** 2) / z.size) for z in Nz]
    C = sum((z / si) @ (z / si).T for z, si in zip(Nz, s)) / sum(z.shape[1] for z in Nz)
    wh = NoiseWhitener.fit(noise)
    assert np.array_equal(wh.covariance, C)
    assert np.allclose(wh.scales, s)


def test_shape_samples_set_the_shape_but_not_the_scale():
    rng = np.random.default_rng(7)
    L = correlated_L()
    noise = recordings(rng, L, 10, 1)
    extra = [100.0 * z for z in recordings(rng, L, 10, 50)]           # different units: normalised away
    wh = NoiseWhitener.fit(noise, estimator="oas", shape_samples=extra)
    B = average_reference_basis(N)
    assert np.allclose(wh.scales, [np.sqrt(np.mean((B.T @ z) ** 2)) for z in noise])
    assert np.trace(wh.covariance) == pytest.approx(N - 1, rel=0.05)


def test_gaussian_loglik_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(8)
    A = rng.standard_normal((5, 5))
    C = A @ A.T + np.eye(5)
    V = rng.standard_normal((5, 7))
    assert gaussian_loglik(C, V) == pytest.approx(np.mean(multivariate_normal(np.zeros(5), C).logpdf(V.T)))


def test_choose_whitener_rule():
    cands = {"i": {"median_loglik": -10, "median_power": 2.0},
             "ii": {"median_loglik": -20, "median_power": 1.08},
             "iii": {"median_loglik": -15, "median_power": 0.86},
             "iv": {"median_loglik": -30, "median_power": 1.0}}
    assert choose_whitener(cands) == "iii"                        # i is best by likelihood but fails power
    assert choose_whitener({"i": cands["i"]}) is None
    assert choose_whitener(cands, power_tolerance=0.1) == "ii"
