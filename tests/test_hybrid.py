"""The hybrid source space: anatomical surface plus a volume grid for deep structures.

What has to hold for a hybrid result to be read as "the surface arm, plus somewhere
for deep signal to go":

* the volume part carries exactly the categories the surface leaves out, never one
  the surface already represents;
* surface sources enter the Monte Carlo pool along their normal -- the same column a
  surface-only run uses -- and volume sources along their dominant direction, the
  Cartesian arm's collapse;
* stratified draws give the surface part exactly the draws a surface-only pool would
  get, with the volume sources added on top.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from source_localization.source_space import hybrid, realizations
from source_localization.source_space.realizations import (
    build_roi_operator, farthest_point_sample)

ALLEN26 = {
    "inputs": {"roi_mapping": "data/atlas/allen/roi_mapping_allen26.json"},
    "source_space": {"surface": {"categories": ["cortical", "cerebellum", "olfactory"]}},
}


def _cfg(**hyb):
    import copy
    c = copy.deepcopy(ALLEN26)
    if hyb:
        c["source_space"]["hybrid"] = hyb
    return c


class TestVolumeCategories:
    def test_default_is_everything_the_surface_leaves_out(self):
        cats, ids = hybrid.volume_category_ids(_cfg())
        assert set(cats) == {"brainstem", "hippocampal", "hypothalamic", "subcortical", "thalamic"}
        # Brainstem_Tectum 21, hippocampus 4/5/14/15, hypothalamus 24,
        # amygdala + basal ganglia 1/3/11/13, thalamus 26
        assert ids == sorted([21, 4, 5, 14, 15, 24, 1, 3, 11, 13, 26])

    def test_a_category_cannot_be_in_both_parts(self):
        with pytest.raises(ValueError, match="represented once"):
            hybrid.volume_category_ids(_cfg(volume_categories=["thalamic", "cerebellum"]))

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="not in the atlas"):
            hybrid.volume_category_ids(_cfg(volume_categories=["midbrain"]))


def test_fixed_sampling_is_refused():
    cfg = _cfg()
    cfg["source_space"]["source_sampling"] = "fixed"
    with pytest.raises(ValueError, match="monte_carlo only"):
        hybrid.create_source_space(cfg, {})


def _fake_forward(rng, n_surf=4, n_vol=3, n_ch=6):
    G3 = rng.standard_normal((n_ch, 3 * (n_surf + n_vol)))
    nn = rng.standard_normal((n_surf, 3))
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    src = [
        {"type": "surf", "nuse": n_surf, "nn": np.vstack([nn, np.ones((2, 3))]),
         "vertno": np.arange(n_surf)},            # two unused vertices, never read
        {"type": "vol", "nuse": n_vol, "nn": np.zeros((n_vol, 3)), "vertno": np.arange(n_vol)},
    ]
    fwd = {"sol": {"data": G3}, "src": src,
           "source_ori": mne.io.constants.FIFF.FIFFV_MNE_FREE_ORI,
           "source_nn": np.tile(np.eye(3), (n_surf + n_vol, 1))}
    return fwd, G3, nn


class TestPoolColumns:
    def test_surface_takes_the_normal_and_volume_the_dominant_direction(self):
        rng = np.random.default_rng(0)
        fwd, G3, nn = _fake_forward(rng)
        G, is_surf = hybrid.pool_columns(fwd, "fixed")
        np.testing.assert_array_equal(is_surf, [True] * 4 + [False] * 3)
        for i in range(4):
            np.testing.assert_allclose(G[:, i], G3[:, 3 * i:3 * i + 3] @ nn[i])
        for i in range(4, 7):
            u, s, _ = np.linalg.svd(G3[:, 3 * i:3 * i + 3], full_matrices=False)
            np.testing.assert_allclose(G[:, i], u[:, 0] * s[0])

    def test_free_collapses_everything(self):
        rng = np.random.default_rng(1)
        fwd, G3, _ = _fake_forward(rng)
        G, _ = hybrid.pool_columns(fwd, "free")
        for i in range(7):
            u, s, _ = np.linalg.svd(G3[:, 3 * i:3 * i + 3], full_matrices=False)
            np.testing.assert_allclose(G[:, i], u[:, 0] * s[0])

    def test_normals_come_from_the_source_space_not_source_nn(self):
        """A free forward's source_nn is the x/y/z axes; using it would be wrong."""
        rng = np.random.default_rng(2)
        fwd, G3, nn = _fake_forward(rng)
        G, _ = hybrid.pool_columns(fwd, "fixed")
        assert not np.allclose(G[:, 0], G3[:, 0])

    def test_fixed_forward_is_refused(self):
        rng = np.random.default_rng(3)
        fwd, _, _ = _fake_forward(rng)
        fwd["source_ori"] = mne.io.constants.FIFF.FIFFV_MNE_FIXED_ORI
        with pytest.raises(ValueError, match="free-orientation"):
            hybrid.pool_columns(fwd, "fixed")


def _pool(rng, n_a=300, n_b=200):
    pos = np.vstack([rng.uniform(-5, 5, (n_a, 3)), rng.uniform(-2, 2, (n_b, 3)) + [0, 0, -6]])
    G = rng.standard_normal((12, n_a + n_b))
    labels = (["A1", "A2"] * n_a)[:n_a] + (["B"] * n_b)
    return pos, G, labels


class TestStratifiedDraws:
    def _record(self, monkeypatch):
        seen = []
        orig = realizations._parcel_rows

        def spy(G_pool, idx, labels, lambda2, combine=realizations.DEFAULT_COMBINE_MODE):
            seen.append(np.array(idx))
            return orig(G_pool, idx, labels, lambda2, combine)
        monkeypatch.setattr(realizations, "_parcel_rows", spy)
        return seen

    def test_each_stratum_draws_what_it_would_draw_alone(self, monkeypatch):
        rng = np.random.default_rng(4)
        pos, G, labels = _pool(rng)
        a, b = np.arange(300), np.arange(300, 500)
        seen = self._record(monkeypatch)
        build_roi_operator(G, pos, labels, k=5, seed=11, strata=[(a, 20), (b, 7)])
        for j, idx in enumerate(seen):
            np.testing.assert_array_equal(idx[idx < 300], farthest_point_sample(pos[a], 20, 11 + j))
            np.testing.assert_array_equal(idx[idx >= 300] - 300, farthest_point_sample(pos[b], 7, 11 + j))

    def test_one_stratum_over_the_whole_pool_is_the_old_behaviour(self):
        rng = np.random.default_rng(5)
        pos, G, labels = _pool(rng)
        op0, p0, _ = build_roi_operator(G, pos, labels, n_sources=30, k=6, seed=3)
        op1, p1, _ = build_roi_operator(G, pos, labels, k=6, seed=3,
                                        strata=[(np.arange(500), 30)])
        assert p0 == p1
        np.testing.assert_array_equal(op0, op1)

    def test_overlapping_strata_raise(self):
        rng = np.random.default_rng(6)
        pos, G, labels = _pool(rng)
        with pytest.raises(ValueError, match="overlap"):
            build_roi_operator(G, pos, labels, k=2, strata=[(np.arange(300), 5), (np.arange(250, 500), 5)])
