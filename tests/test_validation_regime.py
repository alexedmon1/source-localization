"""The validation regime (realistic vs legacy), truth head models, recorded backgrounds, the noise-only summary.

The first group is the back-compatibility gate: the ``legacy`` regime must reproduce the pre-0.6.0
``simulate_dipole`` output **bit for bit**, because the published benchmark (NIMG-26-1224, Tables 2-3) was made
with it. The rest pin the realistic regime's pieces.
"""
import logging

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from source_localization.validation.head_models import HeadModel, HeadModelPrior, TruthForward
from source_localization.validation.noise import RecordedBackground, generate_noise
from source_localization.validation.regime import (LEGACY, LEGACY_WARNING, REALISTIC, describe,
                                                   noise_only_summary, resolve_regime)
from source_localization.validation.simulation import DipoleSimulator


def _toy_simulator(n_sources=27, n_channels=16, seed=0):
    rng = np.random.RandomState(seed)
    grid = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], float)[:n_sources] * 2.0
    fwd = {'sol': {'data': rng.randn(n_channels, n_sources * 3)}, 'source_rr': grid / 1000.0}
    info = mne.create_info([f"E{i + 1}" for i in range(n_channels)], sfreq=500.0, ch_types="eeg")
    for ch, p in zip(info['chs'], rng.randn(n_channels, 3) * 0.01):
        ch['loc'][:3] = p
    return DipoleSimulator(fwd, info, source_space=None, verbose=False)


@pytest.fixture(scope="module")
def sim():
    return _toy_simulator()


# ------------------------------------------------------------------ legacy: bit-identical
def test_legacy_path_is_the_pre_060_formula(sim):
    """No truth_gain, no background: the output equals leadfield @ moment + seeded generated noise scaled to the
    SNR, computed here independently of simulate_dipole."""
    pos, ori, amp, n_t, snr, seed = np.array([2.0, 2.0, 2.0]), np.array([0.0, 1.0, 0.0]), 50.0, 50, 10.0, 7
    out, meta = sim.simulate_dipole(position_mm=pos, orientation=ori, amplitude_nAm=amp, duration_s=0.1,
                                    sfreq=500.0, snr_db=snr, noise_seed=seed)
    idx = int(np.argmin(np.linalg.norm(sim.source_positions - pos, axis=1)))
    moment = np.zeros((sim.n_dipoles * 3, n_t))
    for i in range(3):
        moment[idx * 3 + i, :] = ori[i] * amp * 1e-9
    clean = sim.leadfield @ moment
    noise = generate_noise(sim.n_channels, n_t, np.random.RandomState(seed), noise_type='white',
                           electrode_positions_mm=sim.electrode_positions_mm)
    scale = np.sqrt(np.mean(clean ** 2) / (10 ** (snr / 10) * np.mean(noise ** 2)))
    assert np.array_equal(out, clean + scale * noise)
    assert meta['signal_source'] == 'simulation forward at snapped source'
    assert meta['noise_source'] == 'generated'


# ------------------------------------------------------------------ realistic: overrides
def test_truth_gain_replaces_the_simulation_forward(sim):
    g = np.random.RandomState(3).randn(sim.n_channels, 3)
    ori = np.array([0.0, 0.0, 1.0])
    out, meta = sim.simulate_dipole(position_mm=[1.0, 1.0, 1.0], orientation=ori, amplitude_nAm=50.0,
                                    duration_s=0.1, sfreq=500.0, snr_db=300.0, noise_seed=1, truth_gain=g)
    expected = (g @ ori)[:, None] * np.full(50, 50e-9)[None, :]
    assert np.allclose(out, expected, rtol=1e-12, atol=1e-20)
    assert meta['signal_source'] == 'truth_gain at requested position'
    with pytest.raises(ValueError, match="truth_gain must be"):
        sim.simulate_dipole(position_mm=[1, 1, 1], duration_s=0.1, truth_gain=np.zeros((3, 3)))


def test_background_replaces_generated_noise(sim):
    bg = np.random.RandomState(5).randn(sim.n_channels, 50)
    ori = np.array([1.0, 0.0, 0.0])
    kw = dict(position_mm=[2.0, 0.0, 2.0], orientation=ori, amplitude_nAm=50.0, duration_s=0.1, sfreq=500.0,
              snr_db=0.0, noise_seed=2)
    clean, _ = sim.simulate_dipole(snr_db=300.0, **{k: v for k, v in kw.items() if k != 'snr_db'})
    out, meta = sim.simulate_dipole(background=bg, **kw)
    resid = out - clean
    k = (resid * bg).sum() / (bg * bg).sum()
    assert np.allclose(resid, k * bg, rtol=1e-6, atol=1e-20)          # the noise is the background, rescaled
    assert np.isclose(np.mean(clean ** 2) / np.mean(resid ** 2), 1.0, rtol=1e-6)   # at 0 dB
    assert meta['noise_source'] == 'recorded background' and meta['noise_type'] == 'recorded'


# ------------------------------------------------------------------ regime resolution and annotation
def test_default_regime_is_realistic():
    r = resolve_regime({})
    assert r.name == REALISTIC and r.perturbed_truths and r.noise_only_control and not r.is_legacy
    assert r.head_model_prior == {"shift_sd_mm": 0.3, "skull_factor_range": (0.5, 2.0)}


def test_legacy_regime_is_explicit_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        r = resolve_regime({"regime": "legacy", "background": {"files": "x"}})
    assert r.is_legacy and not r.perturbed_truths and not r.noise_only_control
    assert LEGACY_WARNING in caplog.text and "ignores ['background']" in caplog.text
    d = describe(r)
    assert d["name"] == LEGACY and "inverse crime" in d["description"] and "not comparable" in d["description"]
    assert d["defined_in"] == "source_localization.validation.regime"


def test_override_beats_config_and_unknown_raises():
    assert resolve_regime({"regime": "legacy"}, override="realistic").name == REALISTIC
    with pytest.raises(ValueError, match="regime must be one of"):
        resolve_regime({"regime": "old"})


# ------------------------------------------------------------------ noise-only summary
def test_noise_only_summary_flags_concentration():
    reach = list(range(1, 11))
    spread = noise_only_summary(np.tile(reach, 20), reach)
    assert not spread["flagged"] and np.isclose(spread["normalized_entropy"], 1.0)
    stuck = noise_only_summary([3] * 150 + list(range(1, 11)) * 5, reach)
    assert stuck["flagged"] and stuck["top_roi"] == 3 and stuck["ratio_to_uniform"] > 3


# ------------------------------------------------------------------ head models
def test_prior_draws_are_reproducible_and_bounded():
    prior = HeadModelPrior(0.3, (0.5, 2.0))
    a, b = prior.draw_many(50, seed=1), prior.draw_many(50, seed=1)
    assert a == b
    f = np.array([m.skull_factor for m in a])
    assert f.min() >= 0.5 and f.max() <= 2.0
    assert HeadModelPrior((0.5, 0.5, 0.0)).draw(np.random.default_rng(0)).shift_mm[2] == 0.0
    with pytest.raises(ValueError):
        HeadModel(skull_factor=0.0)


def _sphere_setup():
    """16 electrodes on a 7 mm sphere's upper cap, an analytical 3-layer sphere, positions inside."""
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi / 3, 16)
    ph = rng.uniform(0, 2 * np.pi, 16)
    pos = 0.007 * np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    names = [f"E{i + 1}" for i in range(16)]
    info = mne.create_info(names, 500.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(dict(zip(names, pos)), coord_frame="head"))
    bem = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.007, relative_radii=(0.87, 0.92, 1.0),
                                sigmas=(0.33, 0.0042, 0.33), verbose=False)
    positions_mm = np.array([[0.0, 0.0, 3.0], [1.5, -1.0, 2.0], [-2.0, 1.0, 1.0]])
    return info, bem, positions_mm


def test_truth_forward_nominal_matches_mne_and_perturbations_change_it():
    info, bem, P = _sphere_setup()
    tf = TruthForward(info, bem)
    g0, kept = tf.gains(HeadModel(), P)
    assert kept.all() and g0.shape == (3, 16, 3)
    src = mne.setup_volume_source_space(pos={"rr": P / 1000.0, "nn": np.tile([0, 0, 1.0], (3, 1))}, verbose=False)
    ref = mne.make_forward_solution(info, trans=None, src=src, bem=bem, eeg=True, meg=False, mindist=0.0,
                                    verbose=False)["sol"]["data"].reshape(16, -1, 3).transpose(1, 0, 2)
    assert np.allclose(g0, ref, rtol=1e-6)                       # rebuilt sphere at factor 1 = the original
    g_skull, _ = tf.gains(HeadModel(skull_factor=2.0), P)
    g_shift, _ = tf.gains(HeadModel(shift_mm=(0.3, 0.0, 0.0)), P)
    rel = lambda a: np.linalg.norm(a - g0) / np.linalg.norm(g0)
    assert rel(g_skull) > 0.05 and rel(g_shift) > 0.01
    G, keep = tf.gains_for_models([HeadModel(), HeadModel(skull_factor=2.0)], P)
    assert G.shape == (2, 3, 16, 3) and keep.all() and np.allclose(G[0], g0)


def test_truth_forward_flags_positions_outside_the_head():
    info, bem, P = _sphere_setup()
    g, kept = TruthForward(info, bem).gains(HeadModel(), np.vstack([P, [[0.0, 0.0, 20.0]]]))
    assert kept[:3].all() and not kept[3] and np.isnan(g[3]).all()


# ------------------------------------------------------------------ recorded background
def test_recorded_background_draws_unit_power_windows():
    bg = RecordedBackground.__new__(RecordedBackground)
    rng0 = np.random.default_rng(0)
    bg._data = [rng0.normal(size=(5, 4, 1000)) * 3.0, rng0.normal(size=(2, 4, 1000))]
    bg.files, bg.ch_names, bg.sfreq = ["a", "b"], ["E1", "E2", "E3", "E4"], 500.0
    assert bg.n_segments == 7
    for rng in (np.random.default_rng(1), np.random.RandomState(1)):
        x = bg.draw(500, rng)
        assert x.shape == (4, 500) and np.isclose(np.mean(x ** 2), 1.0)
    with pytest.raises(ValueError, match="simulation needs"):
        bg.draw(2000, np.random.default_rng(0))


# ------------------------------------------------------------------ runner output directories
def test_legacy_runs_get_their_own_output_directory(tmp_path):
    """Legacy results must never land in (or overwrite) a realistic run's directory."""
    from source_localization.validation.regime import regime_name
    from source_localization.validation.runner import ValidationRunner
    cfg = tmp_path / "c.yaml"
    cfg.write_text("pipeline: {name: x}\nvalidation: {regime: legacy}\n")
    legacy = ValidationRunner(cfg, output_dir=tmp_path / "results" / "c", verbose=False)
    realistic = ValidationRunner(cfg, output_dir=tmp_path / "results" / "c", verbose=False, regime="realistic")
    assert legacy.output_dir.name == "c_legacy" and realistic.output_dir.name == "c"
    assert regime_name({}) == "realistic" and regime_name({"regime": "LEGACY"}) == "legacy"
    with pytest.raises(ValueError):
        regime_name({}, override="old")


# ------------------------------------------------------------------ RobustnessTest under the regime
def _sphere_forward():
    info, bem, _ = _sphere_setup()
    rr = np.array([[x, y, z] for x in (-1.5, 0.0, 1.5) for y in (-1.5, 0.0, 1.5) for z in (1.0, 2.5)]) / 1000.0
    src = mne.setup_volume_source_space(pos={"rr": rr, "nn": np.tile([0, 0, 1.0], (len(rr), 1))}, verbose=False)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=bem, eeg=True, meg=False, mindist=0.0,
                                    verbose=False)
    return fwd, fwd['src'], info, bem


def test_robustness_realistic_needs_the_bem_and_legacy_is_explicit():
    from source_localization.validation.robustness import RobustnessTest
    fwd, src, info, bem = _sphere_forward()
    with pytest.raises(ValueError, match="needs the simulation BEM"):
        RobustnessTest(fwd, src, info, verbose=False)
    legacy = RobustnessTest(fwd, src, info, verbose=False, regime="legacy")
    assert legacy.validation_regime["name"] == LEGACY and legacy._realistic is None
    eeg, meta = legacy._sim1((5, 0, 0), position_mm=legacy.source_pos_mm[0], amplitude_nAm=50.0, snr_db=10.0,
                             duration_s=0.5, sfreq=256.0, noise_seed=1)
    assert meta['signal_source'] == 'simulation forward at snapped source'


def test_robustness_realistic_uses_truth_head_models_and_runs_the_noise_only_control():
    from source_localization.validation.robustness import RobustnessTest
    fwd, src, info, bem = _sphere_forward()
    t = RobustnessTest(fwd, src, info, verbose=False, bem=bem,
                       regime_options={"truth_head_model": {"n_models": 3}, "noise_only_control": {"n_trials": 20}})
    assert t.validation_regime["name"] == REALISTIC
    eeg, meta = t._sim1((5, 0, 0), position_mm=t.source_pos_mm[0], amplitude_nAm=50.0, snr_db=10.0,
                        duration_s=0.5, sfreq=256.0, noise_seed=1)
    assert meta['signal_source'] == 'truth_gain at requested position' and eeg.shape == (16, 128)
    a, _ = t._sim1((5, 0, 0), position_mm=t.source_pos_mm[0], amplitude_nAm=50.0, snr_db=10.0, duration_s=0.5,
                   sfreq=256.0, noise_seed=1)
    assert np.array_equal(eeg, a)                                        # reproducible from the key
    eeg2, meta2 = t._sim2((1, 0, 1, 0), position1_mm=t.source_pos_mm[0], position2_mm=t.source_pos_mm[5],
                          snr_db=10.0, duration_s=0.5, sfreq=256.0, noise_seed=2)
    assert meta2['signal_source'] == 'truth_gain at requested position'
    noc = t.run_noise_only_control()
    assert noc['n_trials'] == 20 and 'ratio_to_uniform' in noc and noc['noise_source'] == 'generated (white)'
    assert len(t._realistic.meta()['models']) == 3


def test_background_resamples_to_the_callers_rate():
    bg = RecordedBackground.__new__(RecordedBackground)
    bg._data = [np.random.default_rng(0).normal(size=(3, 4, 1000))]
    bg.files, bg.ch_names, bg.sfreq = ["a"], ["E1", "E2", "E3", "E4"], 500.0
    x = bg.draw(128, np.random.default_rng(1), sfreq=256.0)             # 0.5 s at 256 Hz from 500 Hz data
    assert x.shape == (4, 128) and np.isclose(np.mean(x ** 2), 1.0)


def test_posterior_noise_only_control_stays_diffuse():
    from source_localization.validation.posterior import DipolePosterior
    fwd, src, info, _ = _sphere_forward()
    used = fwd['src'][0]['rr'][fwd['src'][0]['inuse'].astype(bool)] * 1000.0
    dp = DipolePosterior(fwd['sol']['data'], used, 1.5)
    rng = np.random.default_rng(0)
    noc = dp.noise_only_control(snr_db=10.0, n_trials=30, rng=rng)
    assert noc['n_trials'] == 30 and noc['noise_std'] > 0
    planted = []
    for _ in range(30):
        data, sig = dp.simulate(used[4], np.array([0, 0, 1.0]), 30.0, rng)
        planted.append(dp.credible_radius_mm(dp.posterior(data, sig, moment_std=1.0), 0.9))
    assert noc['credible_radius_mm']['0.9']['median'] > np.median(planted)
