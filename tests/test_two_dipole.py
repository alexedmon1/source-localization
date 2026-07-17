"""Tests for the two-dipole robustness sweep's core logic.

The end-to-end sweep needs a real forward model, but its two novel pieces —
matching two recovered peaks to two true positions, and selecting grid pairs at
a target separation — are pure geometry and testable without one. These pin the
behaviour that the sweep's numbers depend on: the 2x2 assignment must be
order-invariant, peak suppression must return two *distinct* sources, and pair
selection must honour its separation tolerance.
"""
import numpy as np
import pytest

from source_localization.validation.robustness import RobustnessTest


class _StubTest:
    """RobustnessTest with a hand-placed source grid and no MNE dependencies.

    Borrows the real geometry methods so we test the shipping code, not a copy.
    """
    _source_activity_norm = RobustnessTest._source_activity_norm
    _find_two_peaks_and_errors = RobustnessTest._find_two_peaks_and_errors
    _select_dipole_pairs = RobustnessTest._select_dipole_pairs
    _select_test_positions = RobustnessTest._select_test_positions

    def __init__(self, source_pos_mm):
        self.source_pos_mm = np.asarray(source_pos_mm, dtype=float)
        self.n_sources = self.source_pos_mm.shape[0]
        # depth = distance from grid centroid, for _select_test_positions
        centroid = self.source_pos_mm.mean(axis=0)
        self.source_depths = np.linalg.norm(self.source_pos_mm - centroid, axis=1)


@pytest.fixture
def grid():
    """A 1mm line of sources along x, from 0 to 10mm."""
    pos = np.zeros((11, 3))
    pos[:, 0] = np.arange(11)
    return _StubTest(pos)


def _activity_at(n_sources, active_indices, n_orient=1, n_times=4):
    """Build source activity with unit magnitude only at given source indices."""
    data = np.zeros((n_sources * n_orient, n_times))
    for idx in active_indices:
        data[idx * n_orient:(idx + 1) * n_orient, :] = 1.0
    return data


def test_two_peaks_recovered_exactly(grid):
    """Two clean, separated peaks -> zero error at both."""
    activity = _activity_at(grid.n_sources, [2, 8])
    activity[2] = 3.0  # peak 1 strongest
    activity[8] = 2.0
    true = np.array([grid.source_pos_mm[2], grid.source_pos_mm[8]])
    e1, e2 = grid._find_two_peaks_and_errors(activity, true, suppression_mm=1.5)
    assert e1 == pytest.approx(0.0)
    assert e2 == pytest.approx(0.0)


def test_assignment_is_order_invariant(grid):
    """Swapping the true-position order must not change the recovered errors."""
    activity = _activity_at(grid.n_sources, [2, 8])
    activity[2] = 3.0
    activity[8] = 2.0
    a = np.array([grid.source_pos_mm[2], grid.source_pos_mm[8]])
    b = a[::-1]
    e_a = grid._find_two_peaks_and_errors(activity, a, suppression_mm=1.5)
    e_b = grid._find_two_peaks_and_errors(activity, b, suppression_mm=1.5)
    # Same set of errors regardless of which true position is listed first.
    assert sorted(e_a) == pytest.approx(sorted(e_b))


def test_suppression_prevents_double_counting_one_blob(grid):
    """Adjacent hot sources are one blob; the 2nd peak must come from elsewhere.

    Sources 4 and 5 are the two strongest and adjacent. Without suppression both
    peaks land on the same blob; with suppression the 2nd peak is forced away,
    so it can be matched to the genuinely separate true source at index 0.
    """
    activity = _activity_at(grid.n_sources, [0, 4, 5])
    activity[4] = 5.0   # strongest
    activity[5] = 4.9   # adjacent, nearly as strong — the trap
    activity[0] = 3.0   # the real second source, far away
    true = np.array([grid.source_pos_mm[4], grid.source_pos_mm[0]])

    e_suppressed = grid._find_two_peaks_and_errors(activity, true, suppression_mm=2.0)
    # Peak1 = source 4 (err 0 to true[0]); peak2 forced past the blob to source 0.
    assert max(e_suppressed) < 1e-9

    e_none = grid._find_two_peaks_and_errors(activity, true, suppression_mm=0.0)
    # Without suppression the 2nd peak is source 5, ~4mm from the true source 0.
    assert max(e_none) > 3.0


def test_suppression_fallback_when_grid_too_small():
    """If suppression masks every source, still return two distinct peaks."""
    stub = _StubTest([[0, 0, 0], [1, 0, 0]])
    activity = _activity_at(2, [0, 1])
    activity[0] = 2.0
    true = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    # suppression_mm huge -> masks everything; fallback picks global 2nd-best.
    e1, e2 = stub._find_two_peaks_and_errors(activity, true, suppression_mm=100.0)
    assert e1 == pytest.approx(0.0)
    assert e2 == pytest.approx(0.0)


def test_free_orientation_activity_is_collapsed(grid):
    """3-orientation source activity must reduce to one magnitude per source."""
    activity = _activity_at(grid.n_sources, [2, 8], n_orient=3)
    activity[2 * 3:3 * 3] = 3.0
    activity[8 * 3:9 * 3] = 2.0
    norm = grid._source_activity_norm(activity)
    assert norm.shape == (grid.n_sources,)
    assert int(np.argmax(norm)) == 2


def test_select_dipole_pairs_hits_target_separation(grid):
    """Selected pairs must sit within tolerance of the requested separation."""
    pairs = grid._select_dipole_pairs([3.0, 5.0], n_pairs=3, tolerance_mm=0.5)
    for sep, plist in pairs.items():
        assert plist, f"no pair found for separation {sep}"
        for idx1, idx2 in plist:
            actual = np.linalg.norm(grid.source_pos_mm[idx1]
                                    - grid.source_pos_mm[idx2])
            assert abs(actual - sep) <= 0.5


def test_select_dipole_pairs_skips_unreachable_separation(grid):
    """A separation with no grid pair within tolerance yields an empty list."""
    # Grid spans 10mm; 50mm is unreachable.
    pairs = grid._select_dipole_pairs([50.0], n_pairs=3, tolerance_mm=1.0)
    assert pairs[50.0] == []


def test_select_dipole_pairs_never_pairs_source_with_itself(grid):
    pairs = grid._select_dipole_pairs([0.0], n_pairs=3, tolerance_mm=0.5)
    for idx1, idx2 in pairs[0.0]:
        assert idx1 != idx2
