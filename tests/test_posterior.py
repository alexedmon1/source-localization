"""Tests for the calibrated source-location posterior (validation/posterior.py).

Building a real leadfield needs a BEM, so these use a small synthetic one. They
pin the properties the "orbital" claims rest on: the density is a proper
normalized distribution, credible regions are the smallest sets holding the
stated mass, the density concentrates on the truth as noise falls, and the
regions actually cover the truth at roughly the nominal rate.
"""
import numpy as np
import pytest

from source_localization.validation.posterior import DipolePosterior


@pytest.fixture
def toy():
    """12 sensors around a 5x5x3 grid; gain falls off with distance."""
    rng = np.random.default_rng(0)
    g = np.arange(5.0)
    pos = np.array(np.meshgrid(g, g, np.arange(3.0), indexing='ij')).reshape(3, -1).T
    sensors = rng.uniform(-4, 8, size=(12, 3))
    sensors[:, 2] += 8.0                       # an array above the volume

    cols = []
    for p in pos:
        d = sensors - p
        r = np.linalg.norm(d, axis=1, keepdims=True)
        cols.append(d / r ** 3)                # dipole-ish 1/r^2 falloff
    G = np.concatenate(cols, axis=1)
    return DipolePosterior(G, pos, spacing_mm=1.0)


def test_posterior_is_a_normalized_distribution(toy):
    rng = np.random.default_rng(1)
    data, sigma = toy.simulate(toy.positions_mm[20], np.array([1.0, 0, 0]),
                               snr_db=10.0, rng=rng)
    p = toy.posterior(data, sigma)
    assert p.shape == (toy.n_positions,)
    assert np.all(p >= 0)
    assert p.sum() == pytest.approx(1.0)


def test_credible_mask_is_the_smallest_set_with_that_mass(toy):
    rng = np.random.default_rng(2)
    data, sigma = toy.simulate(toy.positions_mm[30], np.array([0, 1.0, 0]),
                               snr_db=10.0, rng=rng)
    p = toy.posterior(data, sigma)
    mask = toy.credible_mask(p, 0.9)
    assert p[mask].sum() >= 0.9
    # Dropping the weakest member must break the guarantee — i.e. it is minimal.
    kept = np.where(mask)[0]
    weakest = kept[np.argmin(p[kept])]
    reduced = mask.copy()
    reduced[weakest] = False
    assert p[reduced].sum() < 0.9


def test_regions_shrink_as_noise_falls(toy):
    truth = toy.positions_mm[37]
    sizes = []
    for snr in (0.0, 10.0, 30.0):
        rng = np.random.default_rng(3)
        data, sigma = toy.simulate(truth, np.array([0, 0, 1.0]), snr_db=snr, rng=rng)
        p = toy.posterior(data, sigma)
        sizes.append(toy.credible_volume_mm3(p, 0.9))
    assert sizes[0] > sizes[1] > sizes[2]


def test_posterior_concentrates_on_the_truth(toy):
    idx = 37
    rng = np.random.default_rng(4)
    data, sigma = toy.simulate(toy.positions_mm[idx], np.array([0, 0, 1.0]),
                               snr_db=30.0, rng=rng)
    p = toy.posterior(data, sigma)
    best = int(np.argmax(p))
    assert np.linalg.norm(toy.positions_mm[best] - toy.positions_mm[idx]) <= 1.0


def test_marginal_differs_from_profile_and_stays_normalized(toy):
    rng = np.random.default_rng(5)
    data, sigma = toy.simulate(toy.positions_mm[20], np.array([1.0, 0, 0]),
                               snr_db=0.0, rng=rng)
    prof = toy.posterior(data, sigma)
    marg = toy.posterior(data, sigma, moment_std=1.0)
    assert marg.sum() == pytest.approx(1.0)
    # The Occam term genuinely reweights positions; if these matched, the
    # marginal likelihood would not be doing anything.
    assert not np.allclose(prof, marg)


def test_coverage_is_near_nominal(toy):
    """The claim the whole module rests on, checked on the toy model."""
    r = toy.coverage_test(levels=(0.9,), snr_db=10.0, n_trials=120, seed=7,
                          moment_std=1.0)
    assert 0.75 <= r['coverage'][0] <= 1.0


# ---------------------------------------------------------------------------
# Persistence
#
# Rebuilding the operator costs a BEM solution plus a forward solve, which is
# pure waste when only a figure is changing. These pin that a round-trip is
# lossless and that the cache cannot silently serve a stale operator.
# ---------------------------------------------------------------------------

def test_save_load_round_trip_is_lossless(toy, tmp_path):
    toy.provenance = {'signature': {'spacing_mm': 1.0}}
    toy.electrode_pos_mm = np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0]])
    toy.channel_names = ['E1', 'E2']
    toy.bem_surfaces_mm = [np.ones((4, 3)), 2 * np.ones((4, 3))]
    toy.bem_conductivities = [0.33, 0.0042]

    back = DipolePosterior.load(toy.save(tmp_path / 'op.npz'))
    assert np.array_equal(back.leadfield, toy.leadfield)
    assert np.array_equal(back.positions_mm, toy.positions_mm)
    assert back.spacing_mm == toy.spacing_mm
    assert back.channel_names == ['E1', 'E2']
    assert np.array_equal(back.electrode_pos_mm, toy.electrode_pos_mm)
    assert len(back.bem_surfaces_mm) == 2
    assert back.provenance == toy.provenance


def test_reloaded_operator_gives_identical_posteriors(toy, tmp_path):
    """A cached operator must produce the same numbers, not merely load."""
    toy.provenance = {}
    back = DipolePosterior.load(toy.save(tmp_path / 'op.npz'))
    rng = np.random.default_rng(0)
    data, sigma = toy.simulate(toy.positions_mm[20], np.array([1.0, 0, 0]),
                               snr_db=10.0, rng=rng)
    assert np.allclose(toy.posterior(data, sigma, moment_std=1.0),
                       back.posterior(data, sigma, moment_std=1.0))


def test_cache_signature_tracks_its_inputs(tmp_path):
    """Spacing alone is not enough — a changed BEM must invalidate the cache."""
    bem_dir = tmp_path / 'bem_cache'
    bem_dir.mkdir()
    bem = bem_dir / 'ellipsoid_3layer.pkl'
    bem.write_bytes(b'x' * 100)

    sig = DipolePosterior._cache_signature(tmp_path, 1.0, 0.93)
    assert sig == DipolePosterior._cache_signature(tmp_path, 1.0, 0.93)
    assert sig != DipolePosterior._cache_signature(tmp_path, 0.5, 0.93)
    assert sig != DipolePosterior._cache_signature(tmp_path, 1.0, 0.90)

    bem.write_bytes(b'x' * 200)          # head model rebuilt
    assert sig != DipolePosterior._cache_signature(tmp_path, 1.0, 0.93)
