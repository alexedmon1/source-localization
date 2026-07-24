"""Tests for ROI-level certainty (validation/roi_certainty.py).

The parcel-assignment and matrix machinery is pinned here without the heavy
pipeline: ParcelMap uses the bundled allen32 atlas, and the confusion/
reliability logic is exercised with synthetic attributors so the arithmetic is
checked independently of any estimator.
"""
import numpy as np
import pytest

from source_localization.validation.posterior import DipolePosterior
from source_localization.validation import roi_certainty as rc


@pytest.fixture(scope='module')
def toy_grid():
    """A coarse regular grid spanning the mouse brain box, with a fake montage."""
    g = np.meshgrid(np.linspace(-4, 4, 9), np.linspace(-6, 6, 13),
                    np.linspace(-2.5, 2.5, 6), indexing='ij')
    pos = np.stack([x.ravel() for x in g], axis=1)
    electrodes = np.array([[0, 0, 6.0], [3, 3, 5.0], [-3, -3, 5.0]])
    return pos, electrodes


@pytest.fixture(scope='module')
def pmap(toy_grid):
    pos, electrodes = toy_grid
    return rc.ParcelMap(pos, atlas='allen32', electrode_pos_mm=electrodes)


def test_parcelmap_has_all_allen32_parcels(pmap):
    assert pmap.n_parcels == 32
    assert len(pmap.parcel_names) == 32
    assert np.all(np.diff(pmap.parcel_ids) > 0)          # sorted, unique


def test_inside_is_a_subset_of_labeled(pmap):
    # Every "inside" position must be force-assigned to some parcel, and inside
    # is a strict subset (unlabeled positions exist on a whole-box grid).
    assert pmap.inside.sum() > 0
    assert pmap.inside.sum() < pmap.n_positions
    assert np.all(pmap.labels[pmap.inside] > 0)


def test_aggregate_conserves_mass(pmap):
    rng = np.random.default_rng(0)
    post = rng.random(pmap.n_positions)
    post /= post.sum()
    parcel_probs = pmap.aggregate(post)
    assert parcel_probs.shape == (pmap.n_parcels,)
    assert parcel_probs.sum() == pytest.approx(1.0)


def test_aggregate_charges_each_cell_to_its_parcel(pmap):
    # A posterior concentrated on one grid cell must put all mass on that cell's
    # parcel and nowhere else.
    post = np.zeros(pmap.n_positions)
    post[100] = 1.0
    parcel_probs = pmap.aggregate(post)
    k = np.searchsorted(pmap.parcel_ids, pmap.labels[100])
    assert parcel_probs[k] == pytest.approx(1.0)
    assert parcel_probs.sum() == pytest.approx(1.0)


def test_sample_truth_is_balanced_and_inside(pmap):
    rng = np.random.default_rng(1)
    truth = rc.sample_truth_positions(pmap, n_per_parcel=3, rng=rng)
    # Balanced: every represented parcel gets exactly n_per_parcel trials.
    _, counts = np.unique(truth['parcel'], return_counts=True)
    assert set(counts.tolist()) == {3}
    # Every truth index is strictly inside a label.
    assert np.all(pmap.inside[truth['idx']])
    # depth aligns with the map.
    assert np.allclose(truth['depth_mm'], pmap.depth_mm[truth['idx']])


def test_confusion_normalizations():
    """recall row-normalizes, precision column-normalizes, on a known matrix."""
    counts = np.array([[8, 2], [1, 9]])
    res = rc.ConfusionResult(counts, ['A', 'B'], 'test', int(counts.sum()))

    assert res.recall == pytest.approx(np.array([[0.8, 0.2], [0.1, 0.9]]))
    # columns sum: attr A = 8+1=9, attr B = 2+9=11
    assert res.precision == pytest.approx(
        np.array([[8 / 9, 2 / 11], [1 / 9, 9 / 11]]))
    assert res.overall_accuracy == pytest.approx(17 / 20)
    # empty rows/cols must not divide by zero.
    empty = rc.ConfusionResult(np.zeros((2, 2), int), ['A', 'B'], 'test', 0)
    assert np.all(empty.recall == 0) and np.all(empty.precision == 0)


def test_perfect_and_adversarial_attributors(pmap):
    """The confusion loop reproduces a known-truth attributor exactly."""
    dp = _toy_posterior(pmap)
    rng = np.random.default_rng(2)
    truth = rc.sample_truth_positions(pmap, n_per_parcel=2, rng=rng)

    # An oracle that returns the true parcel gives an identity confusion matrix.
    oracle = lambda gi, data, ns, r: int(
        np.searchsorted(pmap.parcel_ids, pmap.labels[gi]))
    pooled, _ = rc.confusion_matrices(
        oracle, dp, pmap, truth, 10.0, 'oracle', np.random.default_rng(3))
    assert pooled.overall_accuracy == pytest.approx(1.0)
    assert np.trace(pooled.counts) == pooled.counts.sum()

    # A constant attributor concentrates every attribution in one column.
    const = lambda gi, data, ns, r: 0
    pooled_c, _ = rc.confusion_matrices(
        const, dp, pmap, truth, 10.0, 'const', np.random.default_rng(3))
    assert np.all(pooled_c.counts[:, 1:] == 0)
    assert pooled_c.counts[:, 0].sum() == len(truth['idx'])


def test_confusion_depth_bands_partition_the_trials(pmap):
    dp = _toy_posterior(pmap)
    rng = np.random.default_rng(4)
    truth = rc.sample_truth_positions(pmap, n_per_parcel=2, rng=rng)
    bands = (('shallow', 0.0, 4.0), ('deep', 4.0, 99.0))
    oracle = lambda gi, data, ns, r: int(
        np.searchsorted(pmap.parcel_ids, pmap.labels[gi]))
    pooled, banded = rc.confusion_matrices(
        oracle, dp, pmap, truth, 10.0, 'oracle', np.random.default_rng(5), bands)
    # Bands are a partition of the (finite-depth) trials.
    assert sum(r.n_trials for r in banded.values()) == pooled.n_trials


def test_reliability_of_a_calibrated_toy(pmap):
    dp = _toy_posterior(pmap)
    rng = np.random.default_rng(6)
    truth = rc.sample_truth_positions(pmap, n_per_parcel=4, rng=rng)
    rel = rc.reliability(dp, pmap, truth, snr_db=10.0,
                         rng=np.random.default_rng(7), n_bins=5)
    # Multiclass calibration bins every (trial, parcel) pair, not just the true one.
    assert rel['count'].sum() == len(truth['idx']) * pmap.n_parcels
    # Where there are enough pairs, predicted and empirical should track within
    # a loose tolerance — this is a smoke check on calibration, not a proof.
    good = rel['count'] >= 8
    if good.any():
        assert np.nanmax(np.abs(
            rel['mean_predicted'][good] - rel['empirical'][good])) < 0.5


def _toy_posterior(pmap):
    """A DipolePosterior on the same grid, with a smooth distance-decay leadfield."""
    pos = pmap.positions_mm
    electrodes = np.array([[0, 0, 6.0], [3, 3, 5.0], [-3, -3, 5.0],
                           [3, -3, 5.0], [-3, 3, 5.0]])
    cols = []
    for p in pos:
        d = electrodes - p
        r = np.linalg.norm(d, axis=1, keepdims=True)
        cols.append(d / r ** 3)
    G = np.concatenate(cols, axis=1)
    dp = DipolePosterior(G, pos, spacing_mm=1.0)
    return dp
