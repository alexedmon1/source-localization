"""Tests for the head-model-ensemble posterior and its map calibration (validation/ensemble_posterior.py).

Synthetic leadfields, as in test_posterior.py. These pin:
- a one-model ensemble is DipolePosterior's marginal posterior;
- the identity calibration changes nothing;
- tempering and blurring widen the map without moving mass out of it;
- ``mass_above`` scores credible-region membership the same way as ``credible_mask``;
- coverage is nominal when the model is exact;
- with a head-model mismatch the raw map under-covers and the calibration recovers coverage.
The calibrated constants themselves (VOLUMETRIC, SURFACE) were validated on the MEA30 geometry in bayes-sl, not here.
"""
import numpy as np
import pytest

from source_localization.validation.ensemble_posterior import (SURFACE, UNCALIBRATED, VOLUMETRIC, EnsemblePosterior,
                                                               MirrorPairPosterior, mirror_pairs, model_probabilities,
                                                               MapCalibration, blur_kernel, coverage, mass_above)
from source_localization.validation.posterior import DipolePosterior


def _leadfield(pos, sensors):
    """(P, m, 3) dipole-ish 1/r^2 leadfield."""
    d = sensors[None] - pos[:, None]
    r = np.linalg.norm(d, axis=2, keepdims=True)
    return d / r ** 3


@pytest.fixture
def geom():
    rng = np.random.default_rng(0)
    g = np.arange(0, 5, 0.5)
    pos = np.array(np.meshgrid(g, g, np.arange(0, 3, 0.5), indexing='ij')).reshape(3, -1).T
    sensors = rng.uniform(-4, 9, size=(16, 3))
    sensors[:, 2] += 8.0
    return pos, sensors


def _simulate(G_truth, rng, snr, cols=4):
    """Patterns from free-orientation dipoles with leadfields G_truth (n, m, 3 x sources), plus unit white noise."""
    n, m, d = G_truth.shape
    mom = rng.standard_normal((n, d, cols))
    s = np.einsum('nmk,nkc->nmc', G_truth, mom)
    s *= np.sqrt(snr * m / (np.sum(s ** 2, axis=(1, 2), keepdims=True) / cols))
    return s + rng.standard_normal(s.shape)


def test_one_model_ensemble_is_the_dipole_marginal_posterior(geom):
    pos, sensors = geom
    G = _leadfield(pos, sensors)
    ens = EnsemblePosterior(G[None], pos, 0.5)
    dp = DipolePosterior(G.transpose(1, 0, 2).reshape(len(sensors), -1), pos, 0.5)
    y = _simulate(G[[40]], np.random.default_rng(1), snr=2.0)[0]
    tau2 = max(np.sum(y ** 2) / y.shape[1] - len(sensors), 1e-3 * len(sensors)) / ens.gain2
    expect = dp.posterior(y, noise_std=1.0, moment_std=np.sqrt(tau2))
    assert np.allclose(ens.posterior(y), expect, atol=1e-10)


def test_identity_calibration_changes_nothing(geom):
    pos, sensors = geom
    ens = EnsemblePosterior(_leadfield(pos, sensors)[None], pos, 0.5)
    Y = _simulate(_leadfield(pos[[3, 50]], sensors), np.random.default_rng(2), snr=1.0)
    lp = ens.log_posterior(Y)
    assert np.allclose(ens.posterior(Y, UNCALIBRATED), np.exp(lp))
    assert np.allclose(np.exp(lp).sum(1), 1)


def test_blur_kernel_is_row_stochastic_and_conserves_mass(geom):
    pos, _ = geom
    K = blur_kernel(pos, 0.5)
    assert np.allclose(np.asarray(K.sum(axis=1)).ravel(), 1)
    lp = np.log(np.random.default_rng(3).dirichlet(np.ones(len(pos)) * 0.1))
    p = MapCalibration(1.0, 0.5).apply(lp, pos)
    assert p.sum() == pytest.approx(1.0)


def test_tempering_and_blur_widen_the_map_and_keep_its_peak(geom):
    pos, sensors = geom
    ens = EnsemblePosterior(_leadfield(pos, sensors)[None], pos, 0.5)
    y = _simulate(_leadfield(pos[[77]], sensors), np.random.default_rng(4), snr=3.0)[0]
    lp = ens.log_posterior(y)

    def hpd90(p):
        return int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.9) + 1)
    raw = MapCalibration().apply(lp, pos)
    hot = MapCalibration(2.0).apply(lp, pos)
    blurred = MapCalibration(2.0, 0.5).apply(lp, pos)
    assert hot.argmax() == raw.argmax()
    assert hpd90(raw) <= hpd90(hot) <= hpd90(blurred)


def test_mass_above_agrees_with_credible_mask(geom):
    pos, sensors = geom
    ens = EnsemblePosterior(_leadfield(pos, sensors)[None], pos, 0.5)
    p = ens.posterior(_simulate(_leadfield(pos[[120]], sensors), np.random.default_rng(5), snr=0.5)[0])
    for idx in (0, 120, 300, int(p.argmax())):
        for a in (0.5, 0.9):
            assert (mass_above(p, idx) < a) == bool(DipolePosterior.credible_mask(p, a)[idx])


def _coverage_run(ens, G_truth_fn, pos, n, rng, cal, snr=2.0):
    idx = rng.integers(0, len(pos), n)
    Y = _simulate(G_truth_fn(pos[idx]), rng, snr)
    return coverage(mass_above(ens.posterior(Y, cal), idx), levels=(0.5, 0.9))


def test_coverage_is_nominal_when_the_model_is_exact(geom):
    """Data drawn from the model itself (uniform position, moment ~ N(0, tau^2 I), unit noise), tau known."""
    pos, sensors = geom
    G = _leadfield(pos, sensors)
    ens = EnsemblePosterior(G[None], pos, 0.5)
    rng = np.random.default_rng(6)
    n, cols = 400, 4
    tau2 = 2.0 / ens.gain2                         # about 2 x noise per channel at the median gain
    idx = rng.integers(0, len(pos), n)
    Y = np.einsum('nmk,nkc->nmc', G[idx], rng.normal(0, np.sqrt(tau2), (n, 3, cols))) \
        + rng.standard_normal((n, len(sensors), cols))
    cov = coverage(mass_above(ens.posterior(Y, moment_var=tau2), idx), levels=(0.5, 0.9))
    assert cov[0.5] == pytest.approx(0.5, abs=0.07)
    assert cov[0.9] == pytest.approx(0.9, abs=0.05)


def test_head_model_mismatch_under_covers_and_calibration_recovers(geom):
    """Ensemble of 8 shifted sensor arrays; truths from a fresh shift. The raw map is overconfident; widening helps."""
    pos, sensors = geom
    rng = np.random.default_rng(7)
    shift = lambda: sensors + rng.normal(0, 0.3, 3)   # noqa: E731  (a rigid registration error)
    ens = EnsemblePosterior(np.stack([_leadfield(pos, sensors)] + [_leadfield(pos, shift()) for _ in range(7)]),
                            pos, 0.5)
    fresh = shift()
    raw = _coverage_run(ens, lambda p: _leadfield(p, fresh), pos, 300, np.random.default_rng(8), UNCALIBRATED, snr=20.0)
    wide = _coverage_run(ens, lambda p: _leadfield(p, fresh), pos, 300, np.random.default_rng(8),
                         MapCalibration(3.0, 0.5), snr=20.0)
    assert raw[0.9] < 0.8
    assert wide[0.9] > raw[0.9] + 0.1


def test_validated_constant_is_recorded():
    assert VOLUMETRIC == MapCalibration(temperature=1.5, blur_mm=0.25)
    assert SURFACE == MapCalibration(temperature=2.5, blur_mm=0.0)


@pytest.fixture
def sym():
    """A grid symmetric about x = 0 under a symmetric sensor array, so mirror pairs are exact."""
    rng = np.random.default_rng(10)
    gx = np.arange(-2.25, 2.3, 0.5)
    pos = np.array(np.meshgrid(gx, np.arange(0, 4, 0.5), np.arange(0, 2, 0.5), indexing='ij')).reshape(3, -1).T
    half = rng.uniform([0.3, -3, 6], [5, 7, 9], size=(8, 3))
    sensors = np.concatenate([half, half * np.array([-1, 1, 1])])
    return pos, sensors


def test_mirror_pairs_are_mirror_images(sym):
    pos, _ = sym
    pairs = mirror_pairs(pos, 0.0, 0.3)
    assert np.all(pos[pairs[:, 0], 0] > 0.3)
    assert np.allclose(pos[pairs[:, 1]], pos[pairs[:, 0]] * np.array([-1, 1, 1]))


def test_set_evidence_prefers_the_true_source_set(sym):
    pos, sensors = sym
    G = _leadfield(pos, sensors)[None]
    one = EnsemblePosterior(G, pos, 0.5)
    two = MirrorPairPosterior(G, pos, 0.5)
    rng = np.random.default_rng(11)
    n = 100
    far = np.flatnonzero(np.abs(pos[two.configs[:, 0], 0]) > 1.0)   # pairs near the midline look like one source
    i, j = two.configs[rng.choice(far, n)].T
    Y_pair = _simulate(np.concatenate([_leadfield(pos[i], sensors), _leadfield(pos[j], sensors)], axis=2), rng,
                       snr=50.0)
    side = np.flatnonzero(np.abs(pos[:, 0]) > 1.0)
    Y_one = _simulate(_leadfield(pos[rng.choice(side, n)], sensors), rng, snr=50.0)
    p_pair = model_probabilities({'one': one.log_evidence(Y_pair), 'pair': two.log_evidence(Y_pair)})['pair']
    p_one = model_probabilities({'one': one.log_evidence(Y_one), 'pair': two.log_evidence(Y_one)})['one']
    assert np.mean(p_pair > 0.5) > 0.85
    assert np.mean(p_one > 0.5) > 0.9


def test_pair_position_map_puts_mass_on_both_members(sym):
    pos, sensors = sym
    two = MirrorPairPosterior(_leadfield(pos, sensors)[None], pos, 0.5)
    p = two.posterior(_simulate(np.concatenate([_leadfield(pos[[two.configs[5, 0]]], sensors),
                                                _leadfield(pos[[two.configs[5, 1]]], sensors)], axis=2),
                                np.random.default_rng(12), snr=5.0)[0])
    assert p.sum() == pytest.approx(1.0)
    assert two.position_map(p).sum() == pytest.approx(2.0)
