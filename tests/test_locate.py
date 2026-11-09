"""The location package's foundations: head-model ensembles, whitening, regions (field-method plan, phase F1).

Ported from probability-atlas; these tests pin what must not change. The ensemble's draws are fixed by its catalogued
seed, so the first and last draws and a fingerprint of all 33 are asserted (computed 2026-11-09, identical to
probability-atlas's ``draw_ensemble``). The leadfields' equality with probability-atlas's cached ensemble (relative
2e-16) needs a Sentinel pipeline run and runs only when ``SL_LOCATE_RUN`` and ``SL_LOCATE_PA_ENSEMBLE`` point at
one and at the cache file.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from source_localization import methods as M
from source_localization.locate import (ENSEMBLE_PA_PHASE2, ENSEMBLES, HeadModel, HeadModelPrior, NoiseWhitener,
                                        RegionMap, average_reference_basis, credible_set, draw_ensemble)
from source_localization.locate import whitening as W


class TestEnsemble:
    def test_pa_phase2_draws_are_pinned(self):
        models = ENSEMBLE_PA_PHASE2.models()
        assert len(models) == 33 and models[0].is_nominal
        arr = np.array([[*m.shift_mm, m.skull_factor] for m in models])
        np.testing.assert_array_equal(arr[1], [-0.5300112088374551, -0.18714803694409557, -0.592710680269924,
                                               1.0654280503082947])
        np.testing.assert_array_equal(arr[32], [0.2396699986668224, 0.4586441205127128, 0.21584243809581935,
                                                1.3168975828414375])
        assert hashlib.sha256(arr.tobytes()).hexdigest()[:16] == '14d5731b4520cbe4'

    def test_spec_is_catalogued_and_reports_itself(self):
        for key, spec in ENSEMBLES.items():
            assert key == spec.id and key in M.LOCATION_PRIORS
            prov = spec.provenance()
            assert prov['id'] == key and prov['version'] == M.LOCATION_PRIORS[key].version
            assert prov['seed'] == list(spec.seed) and prov['n_draws'] == spec.n_draws

    def test_draw_order_matches_the_prior(self):
        rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
        prior = HeadModelPrior(0.3, (0.5, 2.0))
        a = draw_ensemble(prior, 3, rng1, include_nominal=False)
        z = rng2.normal(0.0, 1.0, 3) * 0.3
        assert a[0].shift_mm == tuple(z)

    def test_validation_uses_the_same_classes(self):
        from source_localization.validation import head_models
        assert head_models.HeadModel is HeadModel and head_models.HeadModelPrior is HeadModelPrior

    def test_old_ensemble_posterior_path_still_imports(self):
        from source_localization.locate import ensemble_posterior as new
        from source_localization.validation import ensemble_posterior as old
        assert old.EnsemblePosterior is new.EnsemblePosterior and old.VOLUMETRIC is new.VOLUMETRIC


class TestWhitening:
    def test_reference_basis_is_orthonormal_and_drops_the_mean(self):
        B = average_reference_basis(30)
        np.testing.assert_allclose(B.T @ B, np.eye(29), atol=1e-12)
        np.testing.assert_allclose(np.ones(30) @ B, 0, atol=1e-12)

    @pytest.mark.parametrize('est', W.ESTIMATORS)
    def test_whitened_noise_has_unit_variance(self, est):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(30, 30))
        cov = A @ A.T / 30 + np.eye(30)
        L = np.linalg.cholesky(cov)
        scales = rng.uniform(0.5, 3, 200)
        noise = [s * (L @ rng.normal(size=(30, 4))) for s in scales]
        wh = NoiseWhitener.fit(noise, estimator=est)
        z = np.concatenate([wh.whiten(n, r) for r, n in enumerate(noise)], axis=1)
        assert wh.n_whitened == 29
        assert np.mean(z ** 2) == pytest.approx(1.0, rel=0.05 if est != 'diagonal' else 0.5)

    def test_every_estimator_is_catalogued(self):
        assert {f'whiten.{e}' for e in W.ESTIMATORS} == set(M.WHITENERS)

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError):
            W.estimate_covariance(np.zeros((3, 4)), 'nope')


class TestRegions:
    def test_aggregation_conserves_mass_and_merges(self):
        rng = np.random.default_rng(2)
        lab = rng.integers(0, 4, 100)
        post = rng.dirichlet(np.ones(100))
        R = RegionMap(lab, ['a', 'b', 'c', 'd'])
        agg = R.aggregate(post)
        assert agg.sum() == pytest.approx(1.0)
        merged = R.merge({'ab': ['a', 'b']})
        assert merged.names == ['ab', 'c', 'd']
        assert merged.aggregate(post)[0] == pytest.approx(agg[0] + agg[1])

    def test_credible_set_is_smallest_reaching_the_level(self):
        idx = credible_set(np.array([0.5, 0.3, 0.15, 0.05]), 0.8)
        assert list(idx) == [0, 1]


@pytest.mark.skipif(not (os.environ.get('SL_LOCATE_RUN') and os.environ.get('SL_LOCATE_PA_ENSEMBLE')),
                    reason='needs a Sentinel pipeline run and probability-atlas\'s cached ensemble')
def test_leadfields_equal_probability_atlas(tmp_path):
    from source_localization.locate.space import LocationSpace
    sp = LocationSpace(os.environ['SL_LOCATE_RUN'], cache_dir=tmp_path)
    G_pa = np.load(os.environ['SL_LOCATE_PA_ENSEMBLE'])[:, :len(sp.positions_mm)]
    assert np.abs(sp.G_ens - G_pa).max() <= 1e-12 * np.abs(G_pa).max()
