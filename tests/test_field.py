"""Field definitions and field templates (field-method plan, phase F2).

Synthetic checks that each catalogued definition recovers a planted field, that sign rules fix signs, and that the
template file validates, freezes without overwriting and round-trips with its hash. The reproduction of surface-
evoked's frozen Sentinel fields from their defining data (exact to 1e-16, 2026-11-09) needs that study's data; when
``SL_FIELD_TEMPLATE`` points at ``sentinel_fields_v1.json``, the file is loaded and checked here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from source_localization.field import (FieldTemplate, TemplateError, apply_sign_rule, csd_leading_pattern, freeze,
                                       load, phase_aligned_pattern, phase_free_pattern, response_svd, scale_units,
                                       unit_index)

CH = [f'E{i}' for i in range(1, 9)]


def _field(rng, n=8):
    v = rng.normal(size=n)
    v -= v.mean()
    return v / np.linalg.norm(v)


def test_unit_index_groups_by_key():
    keys, idx = unit_index([('m2', 'D1'), ('m1', 'D1'), ('m2', 'D1')])
    assert keys == [('m1', 'D1'), ('m2', 'D1')] and idx == [[1], [0, 2]]


def test_response_svd_recovers_the_planted_field():
    rng = np.random.default_rng(0)
    u = _field(rng)
    t = np.linspace(0, 1, 60)
    resp = np.array([a * np.outer(u, np.sin(6 * t)) + 0.01 * rng.normal(size=(8, 60))
                     for a in rng.uniform(0.5, 3, 20)])
    span = t > 0.05
    Y, scales = scale_units(resp, span)
    np.testing.assert_allclose(np.linalg.norm(Y[:, :, span].reshape(20, -1), axis=1), 1)
    maps, s = response_svd(Y.mean(0), span, k=2)
    assert abs(maps[:, 0] @ u) > 0.999 and s[0] > 10 * s[1]


def test_phase_aligned_and_phase_free_recover_a_steady_state_field():
    rng = np.random.default_rng(1)
    u = _field(rng)
    y = np.array([rng.uniform(0.5, 2) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * u
                  + 0.02 * (rng.normal(size=8) + 1j * rng.normal(size=8)) for _ in range(30)])
    units = [[i, i + 1] for i in range(0, 30, 2)]
    D, U, _ = phase_aligned_pattern(y, units, init=u + 0.3 * _field(rng), n_iter=3)
    assert abs(D @ u) > 0.999 and U.shape == (15, 8)
    assert abs(phase_free_pattern(y) @ u) > 0.999


def test_csd_leading_recovers_the_dominant_field():
    rng = np.random.default_rng(2)
    u = _field(rng)
    csd = np.array([a * np.outer(u, u) + 0.05 * np.eye(8) for a in rng.uniform(1, 3, 10)])
    assert abs(csd_leading_pattern(csd) @ u) > 0.999


def test_sign_rule_flips_and_refuses_a_zero_sum():
    v = np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(apply_sign_rule(v, CH, {'electrodes': ['E2'], 'sign': 1}), -v)
    np.testing.assert_array_equal(apply_sign_rule(v, CH, {'electrodes': ['E1'], 'sign': -1}), -v)
    with pytest.raises(ValueError):
        apply_sign_rule(v, CH, {'electrodes': ['E1', 'E2'], 'sign': 1})


def _template(rng, **kw):
    u = apply_sign_rule(_field(rng), CH, {'electrodes': ['E1'], 'sign': 1})
    base = dict(name='toy', version=1, channels=CH, electrodes_mm=np.zeros((8, 3)), reference='average',
                maps={'A': u}, sign_rules={'A': {'electrodes': ['E1'], 'sign': 1}},
                definitions={'A': {'id': 'field.response_svd', 'version': 1, 'span_s': [0.005, 0.3]}})
    base.update(kw)
    return FieldTemplate(**base)


class TestTemplate:
    def test_freeze_round_trips_and_never_overwrites(self, tmp_path):
        t = freeze(_template(np.random.default_rng(3)), tmp_path / 'toy_v1.json')
        assert not t.legacy and len(t.sha256) == 64
        assert t.provenance()['definitions'] == {'A': {'id': 'field.response_svd', 'version': 1}}
        assert json.loads((tmp_path / 'toy_v1.json').read_text())['schema'] == 'source_localization.field_template/1'
        with pytest.raises(FileExistsError):
            freeze(_template(np.random.default_rng(3)), tmp_path / 'toy_v1.json')

    def test_freeze_needs_definitions_and_sign_rules(self, tmp_path):
        with pytest.raises(TemplateError):
            freeze(_template(np.random.default_rng(4), definitions={}), tmp_path / 'a.json')
        with pytest.raises(TemplateError):
            freeze(_template(np.random.default_rng(4), sign_rules={}), tmp_path / 'b.json')

    @pytest.mark.parametrize('bad', ['norm', 'reference', 'sign', 'channel', 'length'])
    def test_validation_catches(self, bad):
        rng = np.random.default_rng(5)
        t = _template(rng)
        if bad == 'norm':
            t.maps['A'] = 2 * t.maps['A']
        elif bad == 'reference':
            t.maps['A'] = t.maps['A'] + 0.1
            t.maps['A'] /= np.linalg.norm(t.maps['A'])
        elif bad == 'sign':
            t.maps['A'] = -t.maps['A']
        elif bad == 'channel':
            t.sign_rules['A'] = {'electrodes': ['E99'], 'sign': 1}
        else:
            t.maps['A'] = t.maps['A'][:7]
        with pytest.raises(TemplateError):
            t.validate()

    def test_pre_schema_templates_load_with_sign_rules_from_text(self, tmp_path):
        rng = np.random.default_rng(6)
        u = apply_sign_rule(_field(rng), CH, {'electrodes': ['E3', 'E5'], 'sign': 1})
        doc = {'name': 'old', 'version': 1, 'channels': CH, 'electrodes_mm': np.zeros((8, 3)).tolist(),
               'reference': 'average', 'maps': {'D': u.tolist()},
               'sign_conventions': {'D': 'posterior midline (E3 + E5) positive (the ASSR sign)'}}
        (tmp_path / 'old.json').write_text(json.dumps(doc))
        t = load(tmp_path / 'old.json')
        assert t.legacy and t.sign_rules == {'D': {'electrodes': ['E3', 'E5'], 'sign': 1}}
        assert t.provenance()['schema'] is None


@pytest.mark.skipif(not os.environ.get('SL_FIELD_TEMPLATE'), reason='needs surface-evoked sentinel_fields_v1.json')
def test_sentinel_v1_loads_and_validates_unchanged():
    t = load(os.environ['SL_FIELD_TEMPLATE'])
    assert t.legacy and t.name == 'sentinel_fields' and t.version == 1
    assert list(t.maps) == ['chirp_D', 'chirp_U2', 'assr40_D', 'assr80_D'] and len(t.channels) == 30
    assert t.sign_rules['chirp_U2']['electrodes'] == ['E1', 'E24', 'E27', 'E30']
