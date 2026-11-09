"""Locating fields (field-method plan, phase F4): calibration domains, refusal outside them, the field locator.

Synthetic: a small grid with random leadfields stands in for a head. The reproduction of surface-evoked's location
report of the frozen Sentinel chirp fields (localize_fields_bayes.json: grand mean, 40 mouse-days, 200-replicate
bootstrap; no differences at its stored precision, 2026-11-09) is bayes-sl ``scripts/f4_check_localize_fields.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from source_localization import methods as M
from source_localization.locate import ENSEMBLE_PA_PHASE2, UNCALIBRATED, VOLUMETRIC
from source_localization.locate.ensemble_posterior import SURFACE, MapCalibration
from source_localization.locate.field_model import FieldLocator, hpd_size, summarize
from source_localization.locate.space import CalibrationDomainError, LocationSpace, array_reference
from source_localization.locate.whitening import average_reference_basis

REF = array_reference('mea30_sentinel')


def _space(rng, channels=None, electrodes=None, ensemble=ENSEMBLE_PA_PHASE2, spacing=0.5, kind='volumetric', K=3):
    xs = np.arange(-2, 2.01, 0.5)
    pos = np.array([[x, y, 0.0] for x in xs for y in (-1.0, 0.0, 1.0)])
    sp = object.__new__(LocationSpace)
    sp.channel_names = list(channels or REF['channels'])
    sp.electrode_pos_mm = np.asarray(electrodes if electrodes is not None else REF['electrodes_mm'], float)
    sp.ensemble, sp.spacing_mm, sp.positions_mm = ensemble, spacing, pos
    sp.labels = {'kind': kind, 'regions': None}
    n = len(sp.channel_names)
    base = rng.normal(size=(len(pos), n, 3))
    sp.G_ens = np.stack([base + 0.05 * rng.normal(size=base.shape) for _ in range(K)])
    sp.G_ens -= sp.G_ens.mean(axis=2, keepdims=True)
    sp.B_ref = average_reference_basis(n)
    sp.provenance = lambda: {'space': 'synthetic'}
    return sp


class TestDomains:
    def test_catalogued_calibrations_carry_their_domain(self):
        for cal in (VOLUMETRIC, SURFACE):
            assert cal.id in M.MAP_CALIBRATIONS and cal.domain.array == 'mea30_sentinel'
            assert cal.domain.ensemble == ENSEMBLE_PA_PHASE2.id and cal.domain.spacing_mm == 0.5
        assert UNCALIBRATED.domain is None and UNCALIBRATED.id == 'calibration.none'

    def test_the_calibration_array_is_mea30(self):
        assert len(REF['channels']) == 30 and np.shape(REF['electrodes_mm']) == (30, 3)

    def test_matching_space_passes(self):
        _space(np.random.default_rng(0)).check_calibration(VOLUMETRIC)

    @pytest.mark.parametrize('change', ['channels', 'electrodes', 'ensemble', 'spacing', 'positions'])
    def test_space_outside_the_domain_is_refused(self, change):
        rng = np.random.default_rng(1)
        kw = {'channels': [f'C{i}' for i in range(30)]} if change == 'channels' else \
             {'electrodes': np.asarray(REF['electrodes_mm']) + [0.3, 0, 0]} if change == 'electrodes' else \
             {'ensemble': type(ENSEMBLE_PA_PHASE2)('prior.other', ENSEMBLE_PA_PHASE2.prior, 8, (1,))} \
             if change == 'ensemble' else {'spacing': 0.25} if change == 'spacing' else {'kind': 'surface'}
        sp = _space(rng, **kw)
        with pytest.raises(CalibrationDomainError):
            sp.check_calibration(VOLUMETRIC)
        with pytest.raises(CalibrationDomainError):
            FieldLocator(sp)

    def test_uncalibrated_or_adhoc_maps_only_when_asked(self):
        sp = _space(np.random.default_rng(2))
        with pytest.raises(CalibrationDomainError):
            FieldLocator(sp, calibration=UNCALIBRATED)
        with pytest.raises(CalibrationDomainError):
            FieldLocator(sp, calibration=MapCalibration(1.2, 0.1))
        FieldLocator(sp, calibration=UNCALIBRATED, allow_uncalibrated=True)


class TestLocator:
    def test_locates_a_planted_single_source_and_a_planted_pair(self):
        rng = np.random.default_rng(3)
        sp = _space(rng)
        noise = [rng.normal(size=(30, 4)) for _ in range(20)]
        loc = FieldLocator(sp).fit(noise)
        G0 = sp.G_ens[0]
        src = 4                                                    # off-midline position, x = -1.5
        y = G0[src] @ np.array([3.0, -2.0, 1.0]) * 2
        pts, ppair = loc.locate(loc.whiten(y, scale=1.0))
        assert pts == [src] and ppair < 0.5
        right = [i for i, p in enumerate(sp.positions_mm) if p[0] == 1.0 and p[1] == 0.0][0]
        left = [i for i, p in enumerate(sp.positions_mm) if p[0] == -1.0 and p[1] == 0.0][0]
        y2 = (G0[right] @ np.array([2.0, 1.0, -1.0]) + G0[left] @ np.array([-1.0, 2.0, 1.5])) * 3
        pts2, ppair2 = loc.locate(loc.whiten(y2, scale=1.0))
        assert ppair2 >= 0.5 and pts2 == [right, left]

    def test_summary_and_provenance(self):
        rng = np.random.default_rng(4)
        sp = _space(rng)
        loc = FieldLocator(sp).fit([rng.normal(size=(30, 4)) for _ in range(20)])
        y = loc.whiten(sp.G_ens[0][10] @ np.array([1.0, 1.0, 0.0]) * 4, scale=1.0)
        labels = (sp.positions_mm[:, 0] > 0).astype(int)
        s = summarize(loc, y, labels, ['L', 'R'], region_groups={'right': ['R']})
        assert s['single']['peak_mm'] == sp.positions_mm[10].round(2).tolist()
        assert 0 <= s['pair']['p_pair'] <= 1 and set(s['single']['region_mass']) <= {'L', 'R'}
        prov = loc.provenance()
        assert prov['calibration']['id'] == 'calibration.volumetric'
        assert prov['models']['whiten.oas'] == M.WHITENERS['whiten.oas'].version
        assert prov['models']['locate.mirror_pair'] == M.LOCATION_POSTERIORS['locate.mirror_pair'].version

    def test_hpd_size(self):
        assert hpd_size(np.array([0.5, 0.3, 0.15, 0.05]), 0.8).tolist() == [2]


class TestFieldModel:
    def _recs(self, rng, sp, kind, src, n=12):
        G0 = sp.G_ens[0]
        u = G0[src] @ np.array([1.0, 0.5, -0.5])
        recs = []
        for _ in range(n):
            a = rng.uniform(1, 2)
            if kind == 'ongoing':
                C = 10 * a * np.outer(u, u) + np.eye(30)
                recs.append({'csd': C, 'csd_q': np.stack([C + 0.1 * np.diag(rng.uniform(size=30)) for _ in range(4)])})
            elif kind == 'steady':
                z = a * np.exp(1j * rng.uniform(0, 6.3)) * u + 0.05 * (rng.normal(size=30) + 1j * rng.normal(size=30))
                recs.append({'z': z, 'z_flip': 0.05 * (rng.normal(size=(4, 30)) + 1j * rng.normal(size=(4, 30)))})
            else:
                t = np.sin(np.linspace(0, 6, 50))
                recs.append({'xbar': a * np.outer(u, t) + 0.05 * rng.normal(size=(30, 50)),
                             'x_flip': 0.05 * rng.normal(size=(4, 30, 50))})
        return recs

    @pytest.mark.parametrize('kind', ['ongoing', 'steady', 'evoked'])
    def test_map_concentrates_on_the_planted_source(self, kind):
        from source_localization.locate.field_model import FieldModel, Units, inner_draws
        rng = np.random.default_rng(10)
        sp = _space(rng)
        recs = self._recs(rng, sp, kind, src=4)
        units = Units(np.arange(len(recs)))
        fm = FieldModel(sp, kind)
        mass, share = fm.map(recs, units, inner_draws(np.ones(units.U), 20, rng), fm.setup(recs, units))
        assert mass.sum() == pytest.approx(1.0) and int(mass.argmax()) == 4 and 0 <= share <= 1

    def test_units_weight_records_within_a_unit(self):
        from source_localization.locate.field_model import Units
        u = Units(['a', 'b', 'b'])
        np.testing.assert_allclose(u.rec_weights(np.array([2.0, 1.0])), [2.0, 0.5, 0.5])

    def test_field_model_refuses_outside_the_domain(self):
        from source_localization.locate.field_model import FieldModel
        with pytest.raises(CalibrationDomainError):
            FieldModel(_space(np.random.default_rng(11), spacing=1.0), 'ongoing')

    def test_model_filter_reproduces_a_mirrored_pair_field(self):
        from source_localization.locate.field_model import MIRROR, model_filter_pair
        rng = np.random.default_rng(12)
        sp = _space(rng)
        G = sp.G_ens - sp.G_ens.mean(2, keepdims=True)
        ir = [i for i, p in enumerate(sp.positions_mm) if p[0] == 1.0 and p[1] == 0.0][0]
        il = [i for i, p in enumerate(sp.positions_mm) if p[0] == -1.0 and p[1] == 0.0][0]
        m = np.array([1.0, -0.5, 0.8])
        target = G[0, ir] @ m + G[0, il] @ (MIRROR @ m)
        target /= np.linalg.norm(target)
        out = model_filter_pair(sp, [1.0, 0.0, 0.0], target)
        assert (out['right_index'], out['left_index']) == (ir, il)
        assert out['filter'] @ target > 0.99 and abs(np.linalg.norm(out['filter']) - 1) < 1e-12
