"""Planted-network validation (validation.networks).

Unit tests run anywhere. The acceptance test reproduces cells of the probability-atlas Phase 9 simulation (FORGE
vehicle resting EEG, MEA30, hybrid allen26 run) at the same seeds and checks the port gives the same numbers; it is
skipped where those files are absent.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from source_localization.validation.networks import signals as sg
from source_localization.validation.networks.harness import NetworkSpec, NetworkValidation, _resolve_callable
from source_localization.validation.networks.scoring import auc, resolvable_snr, resolvable_table


# ------------------------------------------------------------------ unit tests
def test_lagged_is_a_shift_and_delayed_pair_is_fractional():
    tc = sg.plant_timecourses("lagged", np.random.default_rng(1), 3, 1000, 500.0, (6, 8), lag_ms=20)
    assert np.allclose(tc[1][:, 10:], tc[0][:, :-10])                     # B(t) = A(t - 10 samples)
    a, b = sg.delayed_pair(np.random.default_rng(2), 4, 1000, 500.0, (15, 25), lag_ms=5.0)
    f = np.fft.rfftfreq(1000, 1 / 500.0)
    sel = (f >= 15) & (f <= 25)
    cs = (np.fft.rfft(a, axis=-1) * np.conj(np.fft.rfft(b, axis=-1)))[:, sel].sum(0)
    assert np.median(np.angle(cs) / (2 * np.pi * f[sel])) == pytest.approx(0.005, abs=0.0005)


def test_scale_to_snr_hits_the_target_at_the_most_sensitive_electrode():
    rng = np.random.default_rng(0)
    g = rng.normal(size=8)
    tc = sg.band_carrier(rng, 5, 1000, 500.0, (15, 25))
    bg = rng.normal(size=(5, 8, 1000))
    bp = sg.band_power(bg, 500.0, (15, 25))
    k = sg.scale_to_snr(g, tc, bp, 5.0, 500.0, (15, 25))
    planted = sg.avg_ref(k * g)[None, :, None] * tc[:, None, :]
    e = int(np.argmax(np.abs(sg.avg_ref(g))))
    ratio = sg.band_power(planted, 500.0, (15, 25))[e] / bp[e]
    assert 10 * np.log10(ratio) == pytest.approx(5.0, abs=0.05)


def test_resolvable_rule_needs_every_higher_snr():
    top = pd.Series({-5.0: 0.9, 0.0: 0.5, 5.0: 0.8})
    false = pd.Series({-5.0: 0.0, 0.0: 0.0, 5.0: 0.0})
    assert resolvable_snr(top, false) == 5.0                        # -5 dB passes but 0 dB does not
    assert resolvable_snr(pd.Series({5.0: 0.8}), pd.Series({5.0: 0.2})) is None   # false edges too high
    assert auc([2, 3], [0, 1]) == 1.0 and auc([0], [0]) == 0.5


def test_resolvable_table_from_rows():
    rows = []
    for snr in (0.0, 5.0):
        for kind, hit in (("lagged", True), ("single", False)):
            for t in range(10):
                rows.append(dict(pair=0, pair_class="c", node_a="A", node_b="B", readout="R", metric="m",
                                 band="beta", kind=kind, snr=snr, truth=t, top_hit=hit, false_frac=0.0,
                                 attributable=True, z_target=1.0))
    tab = resolvable_table(pd.DataFrame(rows))
    assert tab.loc[tab.kind == "lagged", "resolvable_from_db"].item() == 0.0


def test_metric_callable_from_a_file(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def conn(ts, sfreq, bands):\n    return {}, sorted(ts)\n")
    fn = _resolve_callable(f"{f}:conn")
    assert fn({"b": 1, "a": 2}, 1.0, {})[1] == ["a", "b"]
    assert _resolve_callable("numpy:mean") is np.mean
    with pytest.raises(ValueError):
        _resolve_callable("no_colon")


# ------------------------------------------------------------------ acceptance: reproduce Phase 9
PA = Path("/mnt/e/research/EEG/probability-atlas")
FORGE = Path("/mnt/e/research/EEG/FORGE")
SA_CONN = Path.home() / "sandbox/source-analytics/src/source_analytics/spectral/connectivity.py"
PHASE9_PAIRS = [("dorsal_far", "Frontal_Anterior", "Retrosplenial_L"),
                ("dorsal_far", "Frontal_Anterior", "Visual_Parietal_R"),
                ("dorsal_near", "Motor_L", "Somatosensory_L"), ("dorsal_near", "Motor_R", "Somatosensory_R"),
                ("homologous", "Motor_L", "Motor_R"), ("homologous", "Somatosensory_L", "Somatosensory_R"),
                ("lateral", "Lateral_Cortex_R", "Motor_R"), ("lateral", "Auditory_R", "Somatosensory_R"),
                ("cortex_deep", "Frontal_Anterior", "Deep"), ("cortex_deep", "Retrosplenial_L", "Deep"),
                ("cortex_deep", "Motor_L", "Deep"), ("within_deep", "Deep", "Deep")]
NEEDED = [PA / "scratch/phase9/geometry_run", PA / "scratch/phase9/sims.csv.gz", FORGE / "subject_roster.csv",
          SA_CONN, PA / "reports/2026-10-05_phase2_regions.json"]


def phase9_spec() -> NetworkSpec:
    import json
    roster = list(csv.DictReader(open(FORGE / "subject_roster.csv")))
    groups = []
    for g in ("KO ICV", "WT ICV"):
        rows = sorted((r for r in roster if r["group"] == g and r["mouse_id"] != "803"), key=lambda r: r["subject_id"])
        groups.append({"group": g, "files": [str(Path(r["eeg_dir"]) / r["eeg_filename"]) for r in rows]})
    regions = json.load(open(PA / "reports/2026-10-05_phase2_regions.json"))["amended"]["labellings"]["surface"]["regions"]
    merge = {p: name for name, members in regions.items() for p in members}
    return NetworkSpec(
        pipeline_dir=str(PA / "scratch/phase9/geometry_run"), backgrounds=groups,
        pairs=[{"class": c, "a": a, "b": b} for c, a, b in PHASE9_PAIRS],
        metrics={"callable": f"{SA_CONN}:compute_connectivity_matrix",
                 "names": ["coherence", "imag_coherence", "pli", "wpli", "dwpli", "aec", "partial_corr"],
                 "abs": ["partial_corr"]},
        readouts=[{"name": "R-el", "type": "electrodes"},
                  {"name": "R-parcel", "type": "nodes", "collapse_volume": True},
                  {"name": "R-region", "type": "nodes", "collapse_volume": True, "merge_map": merge}],
        workers=1)


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="probability-atlas Phase 9 / FORGE files absent")
def test_reproduces_probability_atlas_phase9_cells():
    nv = NetworkValidation(phase9_spec(), verbose=False)
    nv.setup()
    nv.control()
    cells = [(0, "beta", 5.0, "lagged", 0), (8, "beta", 5.0, "single", 3), (11, "theta", -5.0, "envelope", 7)]
    new = nv.simulate(cells)
    ref = pd.read_csv(PA / "scratch/phase9/sims.csv.gz")
    for p, band, snr, kind, i in cells:
        a = new[(new.pair == p) & (new.band == band) & (new.snr == snr) & (new.kind == kind) & (new.truth == i)]
        b = ref[(ref.pair == p) & (ref.band == band) & (ref.snr == snr) & (ref.kind == kind) & (ref.truth == i)]
        m = a.merge(b, on=["readout", "metric"], suffixes=("", "_ref"))
        assert len(m) == 21                                              # 3 readouts x 7 metrics
        assert (m.top_pair == m.top_pair_ref).all()
        assert np.allclose(m.z_target, m.z_target_ref, rtol=1e-6, atol=1e-8, equal_nan=True)
        assert np.allclose(m.false_frac, m.false_frac_ref, rtol=0, atol=1e-12, equal_nan=True)


PHASE10_AXES = {0: "AP", 1: "AP", 2: "AP", 3: "AP", 4: "LR", 5: "LR", 8: "CD", 9: "CD", 10: "CD"}
SA_SPECTRAL = SA_CONN.parent


def _phase10_axes(truth_nodes):
    surface = [n for n in truth_nodes if n != "Deep"]
    anterior = {"Olfactory_Bulb", "Frontal_Anterior", "Motor_L", "Motor_R"}
    return {"AP": {"anterior": [n for n in surface if n in anterior],
                   "posterior": [n for n in surface if n not in anterior]},
            "LR": {"left": [n for n in surface if "_L" in n and "_R" not in n],
                   "right": [n for n in surface if "_R" in n and "_L" not in n]},
            "CD": {"Deep": ["Deep"], "cortex": surface}}


@pytest.mark.skipif(not (all(p.exists() for p in NEEDED) and (PA / "scratch/phase10/cells.csv").exists()),
                    reason="probability-atlas Phase 10 / FORGE files absent")
def test_reproduces_probability_atlas_phase10_cells():
    from source_localization.validation.networks import DirectedNetworkValidation
    spec = phase9_spec()
    for p, ax in PHASE10_AXES.items():
        spec.pairs[p]["axis"] = ax
    nv = NetworkValidation(spec, verbose=False)
    nv.setup()
    truth_nodes = sorted({x for x in nv.state["lab_truth"] if x is not None})
    dspec = {"seed": 20261008, "snrs_db": [-5.0, 0.0, 5.0], "weak_db": 10.0, "gc_order": 15,
             "lag_ranges": {"short": [2.0, 5.0],
                            "long": {"theta": [5.0, 25.0], "beta": [5.0, 16.0], "low_gamma": [5.0, 0.4 / 45.0 * 1000]}},
             "measures": {
                 "dpli": {"callable": f"{SA_SPECTRAL}/connectivity.py:compute_connectivity_matrix", "key": "dpli",
                          "net": "antisym", "kwargs": {"metrics": ["dpli"]}},
                 "dtf": {"callable": f"{SA_SPECTRAL}/directed.py:compute_dtf", "key": "dtf", "net": "antisym"},
                 "te1": {"callable": f"{SA_SPECTRAL}/transfer_entropy.py:compute_transfer_entropy", "key": "net_te",
                         "net": "net", "kwargs": {"lag": 1}},
                 "te5": {"callable": f"{SA_SPECTRAL}/transfer_entropy.py:compute_transfer_entropy", "key": "net_te",
                         "net": "net", "kwargs": {"lag": 5}}},
             "builtin": ["psi", "gc", "trgc"], "axes": _phase10_axes(truth_nodes)}
    dv = DirectedNetworkValidation(nv, dspec)
    all_cells = dv.cells()
    ref_cells = pd.read_csv(PA / "scratch/phase10/cells.csv")
    assert len(all_cells) == len(ref_cells)
    pick = [c for c in all_cells if c[0] in (3, 160, 2017, 30000)]
    meta, arr, nodes = dv.simulate(pick)
    for r in arr:
        ref = np.load(PA / f"scratch/phase10/net_{r}.npy", mmap_mode="r")
        for n, c in enumerate(pick):
            row = ref_cells[ref_cells.k == c[0]].iloc[0]
            assert (row.pair, row.band, row.cond, row.truth) == (c[1] if c[1] >= 0 else -1, c[2], c[4], c[5])
            assert np.allclose(arr[r][n], ref[c[0]], rtol=1e-4, atol=1e-6), (r, c)
