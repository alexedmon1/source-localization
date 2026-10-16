"""The leadfield and the data must be on the same reference (X57).

MNE's EEG forward is not average-referenced, and the EEG step added an
average-reference projector without applying it. Every inverse here is
hand-rolled on ``fwd['sol']['data']`` and ``epochs.get_data()``, so the operator
was built from an unreferenced leadfield and applied to data in whatever
reference the file carried. With both referenced, a source estimate cannot
depend on the recording reference: adding any signal common to every channel
changes nothing.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest
import yaml
from pathlib import Path

from source_localization.config import Config
from source_localization.steps import eeg_data
from source_localization.steps.forward_solution import average_reference_leadfield
from source_localization.steps.inverse_solution import apply_inverse_custom_sLORETA

N_CH = 12
PRESET = Path(__file__).resolve().parents[1] / "src/source_localization/config/presets/roi_based_ellipsoid.yaml"


@pytest.fixture(scope="module")
def fwd():
    rng = np.random.default_rng(0)
    names = [f"E{i + 1}" for i in range(N_CH)]
    v = rng.normal(size=(N_CH, 3))
    v[:, 2] = np.abs(v[:, 2]) + 0.5
    pos = 0.009 * v / np.linalg.norm(v, axis=1, keepdims=True)
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, pos)), coord_frame="head"))
    sphere = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.01, verbose=False)
    src = mne.setup_volume_source_space(pos=2.0, sphere=(0, 0, 0, 0.007), exclude=1.0, verbose=False)
    return mne.make_forward_solution(info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False)


def test_mne_forward_is_not_average_referenced(fwd):
    G = fwd["sol"]["data"]
    assert np.median(np.abs(G.mean(0)) / np.sqrt((G ** 2).mean(0))) > 0.1


def test_referencing_zeroes_the_channel_mean_of_both_gains_and_is_idempotent(fwd):
    f = fwd.copy()
    average_reference_leadfield(f)
    assert np.abs(f["sol"]["data"].mean(0)).max() < 1e-12 * np.abs(f["sol"]["data"]).max()
    assert np.abs(f["_orig_sol"].mean(0)).max() < 1e-12 * np.abs(f["_orig_sol"]).max()
    once = f["sol"]["data"].copy()
    average_reference_leadfield(f)
    np.testing.assert_allclose(f["sol"]["data"], once, atol=1e-15 * np.abs(once).max())


def _estimate(f, X):
    return apply_inverse_custom_sLORETA(f, X, snr=3.0, verbose=False)[0]


def test_estimate_is_independent_of_the_recording_reference_only_when_both_are_referenced(fwd):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(N_CH, 50))
    common = rng.normal(size=(1, 50)) * 3          # a reference change adds the same signal to every channel
    P = np.eye(N_CH) - 1.0 / N_CH

    ref = fwd.copy()
    average_reference_leadfield(ref)
    a, b = _estimate(ref, P @ X), _estimate(ref, P @ (X + common))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.abs(a).max())

    raw = fwd.copy()                                 # the pre-0.7.0 behaviour: unreferenced G, unprojected data
    a, b = _estimate(raw, X), _estimate(raw, X + common)
    assert np.abs(a - b).max() > 1e-3 * np.abs(a).max()


def test_config_records_the_default_and_respects_an_explicit_false():
    cfg = Config.from_preset("roi_based_ellipsoid")
    assert "average_reference" not in yaml.safe_load(PRESET.read_text())["forward"]
    assert cfg["forward"]["average_reference"] is True        # so config_resolved.yaml records it
    d = yaml.safe_load(PRESET.read_text())
    d["forward"]["average_reference"] = False
    assert Config(d)["forward"]["average_reference"] is False


@pytest.mark.parametrize("on", [True, False])
def test_eeg_step_applies_the_projector_only_when_on(fwd, monkeypatch, tmp_path, on):
    names = fwd["info"]["ch_names"]
    rng = np.random.default_rng(2)
    epochs = mne.EpochsArray(rng.normal(size=(3, N_CH, 20)) + 5.0, mne.create_info(names, 250.0, "eeg"),
                             verbose=False)
    monkeypatch.setattr(mne.io, "read_epochs_eeglab", lambda *a, **k: epochs.copy())
    (tmp_path / "x.set").touch()
    out = eeg_data.run({"inputs": {"eeg_file": str(tmp_path / "x.set")}, "forward": {"average_reference": on}},
                       {"info": fwd["info"]})
    mean = np.abs(out["epochs"].get_data().mean(axis=1)).max()
    assert (mean < 1e-12) if on else (mean > 1.0)
