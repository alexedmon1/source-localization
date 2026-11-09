"""The field mode's project step (field-method plan, phase F3), against the field-projection design's test list.

Synthetic: the two catalogued projections, the residual, channel matching by name, reference invariance, the output
layout source-analytics reads, the run record, the template hash check, the pipeline's mode switch and the CLI.
The reproduction of surface-evoked's virtual channels (all 192 held-out recordings, relative 6e-16, 2026-11-09) is
bayes-sl ``scripts/f3_check_projection.py``; one recording is compared here when ``SL_FIELD_TEMPLATE``,
``SL_FIELD_RECORDING`` and ``SL_FIELD_REFERENCE`` are set.
"""
from __future__ import annotations

import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat

from source_localization import methods as M
from source_localization.config import Config
from source_localization.field import FieldTemplate, apply_sign_rule, freeze, load
from source_localization.field.project import pick_channels, project, write_run

CH = [f'E{i}' for i in range(1, 9)]


def _unit(rng):
    v = rng.normal(size=8)
    v -= v.mean()
    return v / np.linalg.norm(v)


@pytest.fixture
def template(tmp_path):
    rng = np.random.default_rng(0)
    a, b = _unit(rng), _unit(rng)
    maps = {'A': apply_sign_rule(a, CH, {'electrodes': ['E1'], 'sign': 1}),
            'B': apply_sign_rule(b, CH, {'electrodes': ['E1'], 'sign': 1})}
    t = FieldTemplate(name='toy', version=1, channels=CH, electrodes_mm=np.zeros((8, 3)), reference='average',
                      maps=maps, sign_rules={k: {'electrodes': ['E1'], 'sign': 1} for k in maps},
                      definitions={k: {'id': 'field.response_svd', 'version': 1} for k in maps})
    return freeze(t, tmp_path / 'toy_v1.json')


def _epochs(rng, n=6, t=40):
    return rng.normal(size=(n, 8, t))


def test_inner_projection_is_the_inner_product_and_reference_invariant(template):
    rng = np.random.default_rng(1)
    X = _epochs(rng)
    r = project(X, template)
    np.testing.assert_allclose(r['series']['A'], np.einsum('c,ect->et', template.maps['A'], X), atol=1e-12)
    shifted = project(X + rng.normal(size=(6, 1, 40)), template)          # a common offset across channels
    np.testing.assert_allclose(shifted['series']['B'], r['series']['B'], atol=1e-12)
    np.testing.assert_allclose(shifted['residual_average'], r['residual_average'], atol=1e-12)


def test_joint_projection_is_least_squares_and_equals_inner_for_orthogonal_fields(template):
    rng = np.random.default_rng(2)
    X = _epochs(rng)
    Xr = X - X.mean(1, keepdims=True)
    Mm = template.matrix
    r = project(X, template, method='joint')
    want = np.einsum('ck,ect->ket', np.linalg.pinv(Mm), Xr)
    np.testing.assert_allclose(np.stack([r['series']['A'], r['series']['B']]), want, atol=1e-10)
    orth = FieldTemplate(**{**template.__dict__, 'maps': {'A': template.maps['A']}, 'sign_rules': {},
                            'definitions': {}})
    np.testing.assert_allclose(project(X, orth, method='joint')['series']['A'],
                               project(X, orth, method='inner')['series']['A'], atol=1e-12)


def test_residual_is_orthogonal_to_the_fields_and_completes_the_average(template):
    rng = np.random.default_rng(3)
    X = _epochs(rng)
    r = project(X, template, method='joint')
    np.testing.assert_allclose(template.matrix @ r['residual_average'], 0, atol=1e-10)
    xbar = (X - X.mean(1, keepdims=True)).mean(0)
    recon = template.matrix.T @ np.stack([r['series'][k].mean(0) for k in ('A', 'B')]) + r['residual_average']
    np.testing.assert_allclose(recon, xbar, atol=1e-10)


def test_channels_are_matched_by_name_and_a_missing_one_raises(template):
    rng = np.random.default_rng(4)
    X = _epochs(rng)
    order = rng.permutation(8)
    shuffled = X[:, order]
    names = [CH[i] for i in order]
    np.testing.assert_array_equal(pick_channels(shuffled, names, CH), X)
    with pytest.raises(ValueError, match='missing'):
        pick_channels(shuffled[:, 1:], names[1:], CH)


def test_unknown_map_or_method_raises(template):
    X = _epochs(np.random.default_rng(5))
    with pytest.raises(ValueError):
        project(X, template, maps=['Z'])
    with pytest.raises(ValueError):
        project(X, template, method='nope')


def test_output_layout_and_record(template, tmp_path):
    rng = np.random.default_rng(6)
    X = _epochs(rng, n=5, t=30)
    r = project(X, template, maps=['A'])
    d = write_run(tmp_path / 'rec1', r, template, sfreq=250.0, tmin=-0.5, times=np.arange(30) / 250 - 0.5)
    series = pickle.load(open(d / 'step6_roi_timeseries_signed.pkl', 'rb'))
    assert list(series) == ['A'] and series['A'].shape == (5 * 30,)
    np.testing.assert_allclose(series['A'].reshape(5, 30), r['series']['A'])
    assert float(loadmat(str(d / 'roi_timeseries_signed.set'))['srate'].squeeze()) == 250.0
    res = np.load(d / 'field_residual_average.npz')
    assert res['residual'].shape == (8, 30) and list(res['channels']) == CH
    rec = json.loads((d / 'field_run.json').read_text())
    assert rec['readout_mode'] == 'field' and rec['template']['sha256'] == template.sha256
    assert rec['projection'] == {'id': 'field.project_inner',
                                 'version': M.FIELD_PROJECTIONS['field.project_inner'].version}
    assert rec['n_epochs'] == 5 and rec['n_times'] == 30 and rec['residual'] == 'trial average'


class TestModes:
    def test_modes_are_declared_with_source_the_default(self):
        assert M.default_of(M.PIPELINE_MODES) == 'source' and 'field' in M.PIPELINE_MODES

    def test_source_mode_still_needs_a_head_model(self):
        with pytest.raises(ValueError, match='bem'):
            Config({'pipeline': {'name': 'x'}, 'source_space': {}, 'inverse': {}})

    def test_unknown_mode_and_field_without_template_raise(self):
        with pytest.raises(ValueError, match='mode'):
            Config({'pipeline': {'mode': 'nope'}})
        with pytest.raises(ValueError, match='template'):
            Config({'pipeline': {'mode': 'field'}, 'field': {}})

    def test_field_step_refuses_a_template_with_another_hash(self, template, tmp_path):
        from source_localization.steps import field_project
        cfg = Config({'pipeline': {'mode': 'field'},
                      'field': {'template': str(template.path), 'template_sha256': '0' * 64},
                      'inputs': {'eeg_file': 'x.set'}, 'outputs': {'dir': str(tmp_path)}})
        with pytest.raises(ValueError, match='sha256'):
            field_project.run(cfg, {})


def test_pipeline_and_cli_run_the_field_mode(template, tmp_path, monkeypatch):
    import mne
    rng = np.random.default_rng(7)
    info = mne.create_info(CH[::-1], 250.0, 'eeg')                       # file order differs from the template's
    ep = mne.EpochsArray(rng.normal(size=(4, 8, 25)), info, tmin=-0.5, verbose=False)
    monkeypatch.setattr(mne, 'read_epochs_eeglab', lambda *a, **k: ep)
    src = tmp_path / 'rec.set'
    src.write_bytes(b'stand-in')
    from source_localization.pipeline import Pipeline
    cfg = Config({'pipeline': {'name': 'toy', 'mode': 'field'}, 'field': {'template': str(template.path)}})
    Pipeline(cfg).run(eeg_file=str(src), output_dir=str(tmp_path / 'pipe'))
    rec = json.loads((tmp_path / 'pipe' / 'data' / 'field_run.json').read_text())
    assert rec['n_epochs'] == 4 and rec['source']['path'] == str(src)
    from source_localization.field import cli
    args = SimpleNamespace(field_command='project', template=str(template.path), template_sha256=None, eeg=[str(src)],
                           roster=None, out=str(tmp_path / 'cli'), maps=None, residual_of=None, method='inner',
                           overwrite=False)
    assert cli.run(args) == 0
    a = pickle.load(open(tmp_path / 'cli' / 'rec' / 'data' / 'step6_roi_timeseries_signed.pkl', 'rb'))
    b = pickle.load(open(tmp_path / 'pipe' / 'data' / 'step6_roi_timeseries_signed.pkl', 'rb'))
    np.testing.assert_array_equal(a['A'], b['A'])
    X = pick_channels(ep.get_data(), ep.ch_names, CH)
    np.testing.assert_allclose(a['A'].reshape(4, 25), np.einsum('c,ect->et', template.maps['A'], X), atol=1e-12)


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('SL_FIELD_TEMPLATE', 'SL_FIELD_RECORDING',
                                                        'SL_FIELD_REFERENCE')),
                    reason='needs a Sentinel recording and surface-evoked\'s projection of it')
def test_reproduces_surface_evoked_projection(tmp_path):
    from source_localization.field.project import project_file
    d = project_file(os.environ['SL_FIELD_RECORDING'], load(os.environ['SL_FIELD_TEMPLATE']), tmp_path / 'r')
    got = pickle.load(open(d / 'step6_roi_timeseries_signed.pkl', 'rb'))
    want = pickle.load(open(os.environ['SL_FIELD_REFERENCE'], 'rb'))
    for k in want:
        assert np.abs(got[k] - want[k]).max() <= 1e-12 * np.abs(want[k]).max()
