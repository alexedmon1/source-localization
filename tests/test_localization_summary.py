"""The study's localization summary (study/results_spec.py) and the QC plots it lists.

A small fake study tree: three recordings, one with a session, one that failed,
two localized by Monte Carlo with a collinear pair of parcels. The conformance
tests need the optional ``results`` extra (Lumière) and skip without it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from source_localization.study import results_spec as rs
from source_localization.study.config import StudyConfig, SubjectInfo
from source_localization.study.qc import generate_qc_plots


def _recording(root: Path, output_id: str, *, n_sources: int = 160, collinear: bool = True) -> None:
    pipe = root / "derivatives" / output_id / "pipeline"
    (pipe / "data").mkdir(parents=True)
    (pipe / "figures").mkdir()
    (pipe / "electrode_registration").mkdir()
    (root / "derivatives" / output_id / "roi_timeseries").mkdir()
    (pipe / "pipeline_report.html").write_text("<html></html>")
    (pipe / "electrode_registration" / "electrode_registration_validation.png").write_bytes(b"png")
    for name in ("step1_electrodes.png", "step3_source_space.png"):
        (pipe / "figures" / name).write_bytes(b"png")
    resolved = {"source_localization_version": "0.7.0", "config": {
        "pipeline": {"bem_type": "ellipsoid", "source_type": "shell"},
        "source_space": {"source_sampling": "monte_carlo",
                         "monte_carlo": {"n_sources": n_sources, "n_draws": 100, "seed": 1}},
        "inverse": {"method": "sLORETA", "snr": 3.0},
        "outputs": {"output_variants": "signed"},
        "provenance": {"preset": "shell_ellipsoid", "atlas": "allen26"}}}
    (pipe / "data" / "config_resolved.yaml").write_text(yaml.safe_dump(resolved))
    per_parcel = {
        "Motor_L": {"gain": 1.4, "n_realizations": 100, "coverage": 1.0, "max_collinearity": 0.8,
                    "collinear_with": None},
        "Basal_Ganglia_L": {"gain": 0.5, "n_realizations": 100, "coverage": 0.9, "max_collinearity": 0.995,
                            "collinear_with": "Basal_Ganglia_R" if collinear else None},
        "Basal_Ganglia_R": {"gain": 0.6, "n_realizations": 40, "coverage": 0.4, "max_collinearity": 0.995,
                            "collinear_with": "Basal_Ganglia_L" if collinear else None},
    }
    report = {"n_sources": n_sources, "n_draws": 100, "seed": 1, "pool_sources": 8000,
              "parcels": list(per_parcel), "per_parcel": per_parcel}
    (pipe / "data" / "monte_carlo_report.json").write_text(json.dumps(report))


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "localization"
    _recording(root, "sub-801")
    _recording(root, "sub-802_ses-01", n_sources=200)          # a different recipe
    (root / "derivatives" / "processing_log.json").write_text(json.dumps({"subjects": [
        {"subject_id": "801", "success": True, "processing_time_sec": 30.0},
        {"subject_id": "803", "success": False, "error_message": "no channels"}]}))
    subjects = [SubjectInfo("801", tmp_path / "a.set", group="KO"),
                SubjectInfo("802", tmp_path / "b.set", group="WT", session="01"),
                SubjectInfo("803", tmp_path / "c.set", group="WT")]
    return StudyConfig(name="Demo study", root_dir=root, subjects=subjects)


def _rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_the_summary_conforms_and_names_every_recording(study):
    pytest.importorskip("lumiere")
    from lumiere import check

    qc = study.root_dir / "qc"
    qc.mkdir()
    rep = rs.write(study, qc)
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    (again,) = check(qc)
    assert again.ok

    a = json.loads((qc / "analysis.json").read_text())
    assert a["id"] == "eeg/other/localization/localization"
    assert (a["role"], a["domain"], a["modality"], a["atlas"]) == ("diagnostic", "localization", "eeg",
                                                                   {"name": "allen26"})
    assert a["effect"]["measure"] is None and a["design"] == {"n": 2, "groups": {"KO": 1, "WT": 1}}

    rec = {r["subject_id"]: r for r in _rows(qc / "recordings.csv")}
    # joined on fields, not on folder names: the session is its own column
    assert (rec["802"]["session"], rec["802"]["folder"]) == ("01", "../derivatives/sub-802_ses-01")
    assert rec["801"]["report"] == "../derivatives/sub-801/pipeline/pipeline_report.html"
    assert rec["801"]["figures"].count("; ") == 1
    assert (rec["801"]["status"], rec["803"]["status"], rec["803"]["error"]) == ("localized", "failed",
                                                                                 "no channels")
    assert (rec["801"]["mc_n_collinear"], rec["801"]["mc_n_undersampled"]) == ("2", "1")

    par = {r["parcel"]: r for r in _rows(qc / "parcel_coverage.csv")}
    assert par["Motor_L"]["reliable"] == "true" and par["Basal_Ganglia_R"]["reliable"] == "false"
    assert (par["Basal_Ganglia_L"]["collinear_with"], par["Basal_Ganglia_L"]["collinear_with_n"]) == (
        "Basal_Ganglia_R", "2")

    caveats = " ".join(a["caveats"])
    assert "not all localized with the same recipe: they differ in monte_carlo" in caveats
    assert "1 failed and 0 have no outputs" in caveats
    assert "2 of 3 parcels are near-collinear" in caveats

    prov = json.loads((qc / "provenance.json").read_text())
    assert prov["settings"]["recipe_consistent"] is False
    assert prov["generated_by"][0]["Name"] == "source-localization"


def test_qc_figures_are_listed(study):
    pytest.importorskip("lumiere")
    qc = study.root_dir / "qc"
    (qc / "figures").mkdir(parents=True)
    fig = qc / "figures" / "02_forward_condition.png"
    fig.write_bytes(b"png")
    rep = rs.write(study, qc, plot_paths=[fig, study.root_dir / "elsewhere.png"])
    assert rep.ok, rep.errors
    figures = json.loads((qc / "analysis.json").read_text())["figures"]
    assert figures == [{"path": "figures/02_forward_condition.png", "kind": "diagnostic",
                        "drawn_by": "analysis", "caption": rs.FIGURE_CAPTIONS["forward_condition"],
                        "shows": "recordings.csv"}]


def test_without_the_extra_nothing_is_written(study, monkeypatch):
    monkeypatch.setattr(rs, "lumiere", None)
    qc = study.root_dir / "qc"
    qc.mkdir()
    assert rs.write(study, qc) is None
    assert not any(qc.iterdir())


def test_the_summary_never_raises(study, monkeypatch):
    pytest.importorskip("lumiere")

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(rs, "collect", boom)
    assert rs.write(study, study.root_dir / "qc") is None


def test_qc_plots_skip_metrics_a_monte_carlo_run_does_not_have(tmp_path):
    """Every stc_* is None under Monte Carlo sampling; plotting used to crash on it."""
    df = pd.DataFrame({"subject_id": ["1", "2", "3"], "group": ["KO", "WT", "WT"],
                       "forward_condition_number": [40.0, 41.0, 43.0],
                       "stc_amp_mean": [None, None, None], "stc_amp_std": [None, None, None],
                       "stc_n_times": [None, None, None], "roi_amp_mean": [2e-10, 3e-10, 2.5e-10],
                       "roi_amp_std": [1e-10, 1e-10, 1e-10], "sources_per_roi_mean": [None, None, None],
                       "processing_time_sec": [30.0, 28.0, 31.0], "outlier_flag": [False, False, False]})
    paths = generate_qc_plots(df, {}, tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["01_source_amplitude.png", "02_forward_condition.png", "05_processing_time.png",
                     "06_zscore_heatmap.png"]
    assert all(p.stat().st_size > 0 for p in paths)
