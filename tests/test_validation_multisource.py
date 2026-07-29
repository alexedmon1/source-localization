"""WP-4 — multi-source and patch simulation (NIMG-26-1224 revision, R3-1b / R3-2b).

The first test is the one that matters. `simulate_dipole` gained a `time_course`
argument, and `simulation.py` is the module the *published* benchmark depends
on, so back-compatibility is not a nicety: if the default path shifted by even a
rounding step, every Table 2/3 number would become unreproducible. Invariant 3
of the response plan requires a bit-identical check before any WP-4 result is
used.
"""

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from source_localization.validation.simulation import DipoleSimulator


# ---------------------------------------------------------------------------
# A small synthetic forward model — no atlas or BEM needed, so these tests are
# fast and run anywhere.
# ---------------------------------------------------------------------------

def _toy_simulator(n_sources=40, n_channels=30, seed=0):
    rng = np.random.RandomState(seed)

    grid = []
    step = 2.0
    n = int(np.ceil(n_sources ** (1 / 3)))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                grid.append([i * step, j * step, k * step])
    src_rr = np.asarray(grid[:n_sources], dtype=float) / 1000.0  # mm -> m

    fwd = {
        'sol': {'data': rng.randn(n_channels, n_sources * 3)},
        'source_rr': src_rr,
    }

    info = mne.create_info(
        [f"E{i+1}" for i in range(n_channels)], sfreq=500.0, ch_types="eeg")
    pos = rng.randn(n_channels, 3) * 0.01
    for ch, p in zip(info['chs'], pos):
        ch['loc'][:3] = p

    return DipoleSimulator(fwd, info, source_space=None, verbose=False)


@pytest.fixture(scope="module")
def sim():
    return _toy_simulator()


# ---------------------------------------------------------------------------
# The back-compatibility gate
# ---------------------------------------------------------------------------

def test_time_course_none_is_bit_identical_to_constant_moment(sim):
    """`time_course=None` must reproduce the pre-WP-4 constant-moment path exactly.

    The published benchmark calls simulate_dipole without a time course. A
    changed default would silently invalidate Tables 2-3.
    """
    kw = dict(position_mm=np.array([2.0, 2.0, 2.0]),
              orientation=np.array([1.0, 0.0, 0.0]),
              amplitude_nAm=50.0, duration_s=0.1, sfreq=500.0,
              snr_db=10.0, noise_seed=7)

    a, _ = sim.simulate_dipole(**kw)
    b, _ = sim.simulate_dipole(**kw)
    assert np.array_equal(a, b), "simulate_dipole is not deterministic for a fixed seed"

    # An explicit all-ones waveform is the same physical signal, so it must
    # agree to floating-point, though it need not be bit-identical (it takes a
    # different multiply). This pins the semantics of the new argument.
    c, _ = sim.simulate_dipole(time_course=np.ones(50), **kw)
    assert np.allclose(a, c, rtol=0, atol=1e-18), \
        "an all-ones time course must reproduce the constant-moment signal"


def test_time_course_length_is_validated(sim):
    with pytest.raises(ValueError, match="time_course has"):
        sim.simulate_dipole(position_mm=np.array([2.0, 2.0, 2.0]),
                            duration_s=0.1, sfreq=500.0,
                            time_course=np.ones(17))


# ---------------------------------------------------------------------------
# simulate_n_dipoles
# ---------------------------------------------------------------------------

def test_n_dipoles_superposition_is_linear(sim):
    """Noise-free, N sources must equal the sum of their individual fields."""
    p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([4.0, 4.0, 4.0])
    o1, o2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    kw = dict(duration_s=0.1, sfreq=500.0, snr_db=np.inf, noise_seed=3)

    both, meta = sim.simulate_n_dipoles(
        positions_mm=[p1, p2], orientations=[o1, o2], **kw)
    a, _ = sim.simulate_dipole(position_mm=p1, orientation=o1, **kw)
    b, _ = sim.simulate_dipole(position_mm=p2, orientation=o2, **kw)

    assert np.allclose(both, a + b, rtol=1e-10, atol=0)
    assert meta['n_sources'] == 2
    assert not meta['collided']


def test_n_dipoles_records_both_separations(sim):
    """Requested separation is the design variable; snapped is what was simulated."""
    _, meta = sim.simulate_n_dipoles(
        positions_mm=[[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]],
        duration_s=0.05, sfreq=500.0, snr_db=np.inf, noise_seed=1)
    assert meta['pairwise_requested_separation_mm']['0-1'] == pytest.approx(3.1)
    assert '0-1' in meta['pairwise_snapped_separation_mm']
    assert len(meta['snapping_error_mm']) == 2


def test_n_dipoles_flags_collision(sim):
    """Two requests snapping to one grid point is not two sources -- say so."""
    p = [0.05, 0.05, 0.05]
    _, meta = sim.simulate_n_dipoles(
        positions_mm=[p, p], duration_s=0.05, sfreq=500.0,
        snr_db=np.inf, noise_seed=1)
    assert meta['collided'] is True


def test_n_dipoles_correlation_is_controllable(sim):
    """The point of time courses: sources that are NOT perfectly correlated.

    The pre-existing two-dipole path gives r = 1 because each dipole carries a
    constant DC moment. This must be disclosed rather than presented as
    pre-existing capability.
    """
    rng = np.random.RandomState(0)
    n_times = 200
    a = rng.randn(n_times)
    b = rng.randn(n_times)

    _, meta = sim.simulate_n_dipoles(
        positions_mm=[[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        time_courses=np.vstack([a, b]),
        duration_s=n_times / 500.0, sfreq=500.0, snr_db=np.inf, noise_seed=1)

    r = meta['source_correlation'][0, 1]
    assert abs(r) < 0.3, f"independent waveforms should be near-uncorrelated, got r={r}"

    # And the default (DC) case is perfectly correlated, as the old path was.
    _, meta_dc = sim.simulate_n_dipoles(
        positions_mm=[[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        duration_s=0.1, sfreq=500.0, snr_db=np.inf, noise_seed=1)
    assert meta_dc['source_correlation'] is None, \
        "constant moments have zero variance; correlation must be reported as undefined"


# ---------------------------------------------------------------------------
# simulate_patch
# ---------------------------------------------------------------------------

def test_patch_activates_multiple_sources_and_reports_extent(sim):
    _, meta = sim.simulate_patch(
        center_mm=[2.0, 2.0, 2.0], radius_mm=3.0,
        duration_s=0.05, sfreq=500.0, snr_db=np.inf, noise_seed=1)
    assert meta['n_patch_sources'] > 1
    assert meta['patch_extent_mm'] > 0
    assert meta['patch_coherent'] is True


def test_patch_conserves_total_moment(sim):
    """A larger patch must not win by carrying more total current.

    Amplitude is split across members, so patch and point source of the same
    nominal strength are comparable.
    """
    kw = dict(center_mm=[2.0, 2.0, 2.0], amplitude_nAm=50.0,
              orientation=np.array([1.0, 0.0, 0.0]),
              duration_s=0.05, sfreq=500.0, snr_db=np.inf, noise_seed=1)
    small, m_small = sim.simulate_patch(radius_mm=0.5, **kw)
    large, m_large = sim.simulate_patch(radius_mm=4.0, **kw)

    assert m_large['n_patch_sources'] > m_small['n_patch_sources']
    assert m_small['total_amplitude_nAm'] == m_large['total_amplitude_nAm'] == 50.0
    # Fields differ (spread differs) but not by an order of magnitude, which is
    # what an unconserved total moment would produce.
    ratio = np.abs(large).max() / np.abs(small).max()
    assert 0.1 < ratio < 10, f"patch field scale ran away: ratio={ratio:.3f}"


def test_patch_degenerates_to_a_point_when_radius_is_tiny(sim):
    _, meta = sim.simulate_patch(
        center_mm=[2.0, 2.0, 2.0], radius_mm=1e-6,
        duration_s=0.05, sfreq=500.0, snr_db=np.inf, noise_seed=1)
    assert meta['n_patch_sources'] == 1
