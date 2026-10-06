"""Tests for the parcel-subspace ROI operator (source_space/parcel_subspace.py) and the shared
pool helper (source_space/pool.py).

The operator's arithmetic is pinned on toy leadfields with known structure. Its localization
against Monte Carlo and ROI-based was measured in simulation in surface-evoked
(analysis/diag_mc_combination.py; troubleshooting log section 24), and the deployed step is
checked against that construction on a real run (analysis/diag_subspace_deployed_check.py).
"""
import numpy as np
import pytest

from source_localization.source_space import parcel_subspace as ps
from source_localization.source_space.pool import is_pool_sampling, pool_config


def toy_pool(rng, n_ch=12, per=8):
    """Three parcels, each a different 2-D subspace of channel space, plus noise columns."""
    basis = np.linalg.qr(rng.normal(size=(n_ch, n_ch)))[0]
    G, labels = [], []
    for k, name in enumerate(("A", "B", "C")):
        sub = basis[:, 2 * k:2 * k + 2]
        G.append(sub @ rng.normal(size=(2, per)))
        labels += [name] * per
    G.append(rng.normal(size=(n_ch, 3)) * 0.01)
    labels += [None, "", None]
    return np.hstack(G), labels, basis


def test_each_parcel_wins_its_own_patterns():
    rng = np.random.default_rng(0)
    G, labels, basis = toy_pool(rng)
    for method in ps.SUBSPACE_METHODS:
        op, parcels, comps, _ = ps.build_subspace_operator(G, labels, n_patterns=2, lambda2=1e-3,
                                                           method=method)
        assert parcels == ["A", "B", "C"]
        for k, p in enumerate(parcels):
            for col in np.flatnonzero(np.array(labels, dtype=object) == p):
                y = op @ G[:, col]
                power = [float((y[comps[q]] ** 2).sum()) for q in parcels]
                assert int(np.argmax(power)) == k, (method, p)


def test_patterns_capped_at_rank_and_energy_reported():
    rng = np.random.default_rng(1)
    G, labels, _ = toy_pool(rng)
    _, parcels, comps, rep = ps.build_subspace_operator(G, labels, n_patterns=5, lambda2=1e-3)
    for p in parcels:
        assert rep[p]["n_patterns"] == 2 == len(comps[p])        # rank 2 per toy parcel
        assert rep[p]["energy_captured"] == pytest.approx(1.0)
        assert rep[p]["n_pool"] == 8
    _, _, comps1, rep1 = ps.build_subspace_operator(G, labels, n_patterns=1, lambda2=1e-3)
    assert all(len(c) == 1 for c in comps1.values())
    assert all(0.5 <= rep1[p]["energy_captured"] < 1.0 for p in rep1)


def test_dominant_pattern_sign_is_anchored():
    rng = np.random.default_rng(2)
    Gp = rng.normal(size=(10, 6)) + 2.0                            # a clear mean pattern
    pat, _, _ = ps.parcel_patterns(Gp, 3)
    assert pat[:, 0] @ Gp.sum(axis=1) > 0
    pat_neg, _, _ = ps.parcel_patterns(-Gp, 3)
    assert pat_neg[:, 0] @ (-Gp).sum(axis=1) > 0


def test_patterns_scaled_to_typical_source_gain():
    Gp = np.tile(np.array([[3.0], [4.0]]), (1, 9))                # nine identical columns, norm 5
    pat, energy, _ = ps.parcel_patterns(Gp, 3)
    assert pat.shape == (2, 1)
    assert np.linalg.norm(pat[:, 0]) == pytest.approx(5.0)
    assert energy == pytest.approx(1.0)


def test_sloreta_is_mne_standardized_by_resolution():
    """sLORETA rows are the MNE rows divided by sqrt(diag(W G)), the convention
    realizations._parcel_rows and the fixed-grid inverse use."""
    rng = np.random.default_rng(3)
    G, labels, _ = toy_pool(rng)
    mne_op, parcels, _, _ = ps.build_subspace_operator(G, labels, n_patterns=2, lambda2=0.1, method="MNE")
    sl_op, _, _, _ = ps.build_subspace_operator(G, labels, n_patterns=2, lambda2=0.1, method="sLORETA")
    stacked = np.hstack([ps.parcel_patterns(G[:, [i for i, l in enumerate(labels) if l == p]], 2)[0]
                         for p in parcels])
    res = np.einsum("ij,ji->i", mne_op, stacked)
    np.testing.assert_allclose(sl_op, mne_op / np.sqrt(res)[:, None])


def test_rejects_bad_input():
    G = np.eye(4)
    with pytest.raises(ValueError):
        ps.build_subspace_operator(G, ["A", "B"])
    with pytest.raises(ValueError):
        ps.build_subspace_operator(G, ["A", "A", "B", "B"], method="eLORETA")
    with pytest.raises(ValueError):
        ps.build_subspace_operator(G, ["A", "A", "B", "B"], n_patterns=0)


def _cfg(sampling, **sections):
    return {"source_space": {"source_sampling": sampling, **sections}}


def test_pool_sampling_modes():
    assert not is_pool_sampling(_cfg("fixed"))
    assert not is_pool_sampling({"source_space": {}})
    assert is_pool_sampling(_cfg("monte_carlo"))
    assert is_pool_sampling(_cfg("parcel_subspace"))


def test_pool_config_falls_back_to_monte_carlo():
    mc = {"pool_spacing_mm": 0.2, "n_sources": 48}
    assert pool_config(_cfg("monte_carlo", monte_carlo=mc)) == mc
    got = pool_config(_cfg("parcel_subspace", monte_carlo=mc, parcel_subspace={"n_patterns": 3}))
    assert got["pool_spacing_mm"] == 0.2 and got["n_patterns"] == 3
    got = pool_config(_cfg("parcel_subspace", monte_carlo=mc, parcel_subspace={"pool_spacing_mm": 0.3}))
    assert got["pool_spacing_mm"] == 0.3
    assert pool_config(_cfg("fixed", monte_carlo=mc)) == {}
