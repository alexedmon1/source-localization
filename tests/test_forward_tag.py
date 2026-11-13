"""Saved forward models carry the solver that made them, and readers refuse ones that do not (bem/solution.py).

Forward models saved before the BEM-order fix look exactly like current ones. These pin:
- the forward step tags every forward, in the object and in a sidecar written even without intermediates, with the
  surface-BEM tag for surface BEMs and the analytic tag for the analytic sphere;
- ``load_forward`` refuses an untagged or unknown-tagged pickle; ``run_forward_is_current`` reads the sidecar, else
  the pickle;
- a batch run neither skips nor overwrites a subject whose outputs have no current tag; it reports it as failed;
- head-ensemble exports before version 2 are refused;
- a saved operator whose signature has no current solver is refused, unless the cache path asks not to check.
"""
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

mne = pytest.importorskip("mne")
from source_localization.bem import solution as S  # noqa: E402


def _surface_inputs(tmp_path):
    from source_localization.bem.ellipsoid import create_ellipsoid_surfaces_from_axes
    axes = [np.full(3, 5.75)]
    for t in (0.5, 0.3):
        axes.append(axes[-1] * (1 + t / axes[-1].mean()))
    surfs = create_ellipsoid_surfaces_from_axes(np.zeros(3), axes, np.eye(3), [0.33, 0.0042, 0.33], n_vertices=162)
    names = [f"E{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    d = rng.normal(size=(8, 3))
    d[:, 2] = np.abs(d[:, 2]) + 1
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, d * axes[-1][0] / 1000)),
                                                   coord_frame="head"))
    pos = rng.normal(size=(5, 3)) * 1.0
    src = mne.setup_volume_source_space(pos=dict(rr=pos / 1000, nn=np.tile([0, 0, 1.0], (5, 1))), verbose=False)
    config = {"pipeline": {"bem_type": "ellipsoid", "source_type": "test"},
              "forward": {"eeg": True, "meg": False, "mindist": 0.0, "average_reference": True},
              "outputs": {"dir": str(tmp_path), "save_intermediate": False}}
    prev = {"info": info, "src": src, "bem": {"surfs": surfs, "is_sphere": False}, "source_coords_mm": pos}
    return config, prev


def test_forward_step_tags_the_forward_and_writes_the_sidecar(tmp_path):
    from source_localization.steps import forward_solution
    config, prev = _surface_inputs(tmp_path)
    out = forward_solution.run(config, prev)
    assert out["fwd"][S.TAG_KEY] == S.SOLVER_TAG
    side = json.loads((tmp_path / "data" / S.TAG_FILE).read_text())
    assert side["bem_solver"] == S.SOLVER_TAG and side["bem_type"] == "ellipsoid"
    assert S.run_forward_is_current(tmp_path / "data")
    with open(tmp_path / "fwd.pkl", "wb") as f:
        pickle.dump(out["fwd"], f)
    assert S.load_forward(tmp_path / "fwd.pkl")[S.TAG_KEY] == S.SOLVER_TAG


def test_untagged_and_unknown_forwards_are_refused(tmp_path):
    for name, fwd in (("untagged", {"sol": {"data": np.zeros((2, 3))}}),
                      ("unknown", {"sol": {"data": np.zeros((2, 3))}, S.TAG_KEY: "something else"})):
        d = tmp_path / name / "data"
        d.mkdir(parents=True)
        with open(d / "step4_forward.pkl", "wb") as f:
            pickle.dump(fwd, f)
        with pytest.raises(S.StaleForwardError, match="b5de19f"):
            S.load_forward(d / "step4_forward.pkl")
        assert not S.run_forward_is_current(d)
    assert not S.run_forward_is_current(tmp_path / "missing")
    with pytest.raises(ValueError):
        S.tag_forward({}, "not a tag")


def test_sidecar_decides_without_the_pickle(tmp_path):
    S.write_forward_tag(tmp_path, S.ANALYTIC_TAG, "sphere")
    assert S.run_forward_is_current(tmp_path)
    (tmp_path / S.TAG_FILE).write_text(json.dumps({"bem_solver": "old"}))
    assert not S.run_forward_is_current(tmp_path)


def test_batch_reports_stale_outputs_instead_of_skipping(tmp_path):
    from source_localization.study.batch import process_subject
    out = tmp_path / "sub-01"
    (out / "roi_timeseries").mkdir(parents=True)
    (out / "roi_timeseries" / "roi_timeseries_signed.set").write_text("old")
    (out / "pipeline" / "data").mkdir(parents=True)
    with open(out / "pipeline" / "data" / "step4_forward.pkl", "wb") as f:
        pickle.dump({"sol": {"data": np.zeros((2, 3))}}, f)
    cfg = SimpleNamespace(get_subject_output_dir=lambda s: out)
    r = process_subject(cfg, SimpleNamespace(subject_id="01"), skip_existing=True)
    assert not r.success and "BEM-solver tag" in r.error_message and "--force" in r.error_message
    assert (out / "roi_timeseries" / "roi_timeseries_signed.set").read_text() == "old"

    S.write_forward_tag(out / "pipeline" / "data", S.SOLVER_TAG, "ellipsoid")
    r = process_subject(cfg, SimpleNamespace(subject_id="01"), skip_existing=True)
    assert r.success and r.metadata == {"skipped": True}


def test_head_ensemble_exports_before_version_2_are_refused(tmp_path):
    from source_localization.validation.head_ensemble import EXPORT_VERSION, load_head_ensemble
    assert EXPORT_VERSION >= 2
    for v, ok in ((1, False), (EXPORT_VERSION, True)):
        p = tmp_path / f"v{v}.npz"
        np.savez(p, leadfields=np.zeros((1, 2, 3, 3)), meta_json=np.array(json.dumps({"export_version": v})))
        if ok:
            assert load_head_ensemble(p)["meta"]["export_version"] == v
        else:
            with pytest.raises(S.StaleForwardError, match="Re-export"):
                load_head_ensemble(p)


def test_saved_operators_from_the_old_solver_are_refused(tmp_path):
    from source_localization.validation.posterior import DipolePosterior
    op = DipolePosterior(np.ones((4, 6)), np.zeros((2, 3)), 0.5)
    for name, prov, refused in (("old", {"signature": {"spacing_mm": 0.5}}, True),
                                ("new", {"signature": {"spacing_mm": 0.5, "bem_solver": S.SOLVER_TAG}}, False),
                                ("other", {}, False)):
        op.provenance = prov
        p = op.save(tmp_path / f"{name}.npz")
        if refused:
            with pytest.raises(S.StaleForwardError):
                DipolePosterior.load(p)
            assert DipolePosterior.load(p, check=False).provenance == prov
        else:
            assert DipolePosterior.load(p).provenance == prov


def test_analytic_sphere_forwards_get_the_analytic_tag(tmp_path):
    from source_localization.steps import forward_solution
    config, prev = _surface_inputs(tmp_path)
    config["pipeline"]["bem_type"] = "sphere"
    prev["bem"] = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.0066, relative_radii=[0.87, 0.92, 1.0],
                                        sigmas=[0.33, 0.0042, 0.33], verbose=False)
    out = forward_solution.run(config, prev)
    assert out["fwd"][S.TAG_KEY] == S.ANALYTIC_TAG
    assert json.loads((tmp_path / "data" / S.TAG_FILE).read_text())["bem_solver"] == S.ANALYTIC_TAG
