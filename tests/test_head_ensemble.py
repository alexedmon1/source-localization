"""The MEA30 ensemble's draws, frame offsets and the export file's round trip. Exact values pin the reference draws: MS2's
leadfields and every field-model result rest on them."""
import json

import numpy as np

from source_localization.validation.head_ensemble import MEA30_ENSEMBLE, EXPORT_VERSION, _ontology, load_head_ensemble
from source_localization.validation.head_models import HeadModel, HeadModelPrior, draw_ensemble


def test_ensemble_sizes_and_nominal_first():
    models, truth = MEA30_ENSEMBLE.models(), MEA30_ENSEMBLE.truth_models()
    assert len(models) == 33 and len(truth) == 8
    assert models[0].is_nominal and not any(m.is_nominal for m in truth)
    key = lambda m: (m.shift_mm, m.skull_factor)
    assert not {key(m) for m in models} & {key(m) for m in truth}


def test_draws_are_reproducible_and_follow_the_stream_order():
    a, b = MEA30_ENSEMBLE.models(), MEA30_ENSEMBLE.models()
    assert a == b
    rng = np.random.default_rng([20261005, 1, 0])
    shift = rng.normal(0.0, 1.0, 3) * 0.3
    factor = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    assert a[1] == HeadModel(tuple(shift), factor)


def test_offset_moves_every_model_and_only_the_shift():
    base, off = MEA30_ENSEMBLE.models(), MEA30_ENSEMBLE.models((0.0, 0.25, 0.0))
    for m, n in zip(base, off):
        assert np.allclose(np.subtract(n.shift_mm, m.shift_mm), (0.0, 0.25, 0.0)) and n.skull_factor == m.skull_factor


def test_draw_ensemble_without_nominal():
    assert len(draw_ensemble(HeadModelPrior(), 3, np.random.default_rng(0), include_nominal=False)) == 3


def test_ontology_holds_the_colliculi():
    o = _ontology()
    assert o[4]["acronym"] == "IC" and o[302]["acronym"] == "SCs" and o[294]["acronym"] == "SCm"


def test_load_refuses_a_newer_export(tmp_path):
    p = tmp_path / "e.npz"
    np.savez(p, positions_mm=np.zeros((2, 3)), meta_json=np.array(json.dumps(dict(export_version=EXPORT_VERSION + 1))))
    try:
        load_head_ensemble(p)
    except ValueError as e:
        assert "export version" in str(e)
    else:
        raise AssertionError("a newer export version must be refused")


def test_load_round_trip(tmp_path):
    p = tmp_path / "e.npz"
    np.savez(p, positions_mm=np.ones((2, 3)), meta_json=np.array(json.dumps(dict(export_version=EXPORT_VERSION, atlas="allen26"))))
    d = load_head_ensemble(p)
    assert d["meta"]["atlas"] == "allen26" and d["positions_mm"].shape == (2, 3)
