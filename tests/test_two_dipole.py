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
    _select_depth_matched_pairs = RobustnessTest._select_depth_matched_pairs
    _select_axis_pairs = RobustnessTest._select_axis_pairs
    _select_test_positions = RobustnessTest._select_test_positions

    def __init__(self, source_pos_mm, source_depths=None):
        self.source_pos_mm = np.asarray(source_pos_mm, dtype=float)
        self.n_sources = self.source_pos_mm.shape[0]
        if source_depths is not None:
            self.source_depths = np.asarray(source_depths, dtype=float)
        else:
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


# ---- depth-matched pair selection ----------------------------------------

@pytest.fixture
def depth_grid():
    """A 2D sheet of sources with an explicit depth gradient along x.

    Sources on a 21-point x-line, each also assigned a depth equal to its x
    coordinate. So a pair 4mm apart in x is also 4mm apart in depth — unless we
    build a grid where separation and depth can differ. Here we use a y-offset
    lattice so same-depth pairs exist at a given separation.
    """
    xs = np.arange(0, 11)          # 0..10 mm, the depth axis
    ys = np.array([0.0, 4.0])      # two rows -> same-depth partners 4mm apart in y
    pos, depth = [], []
    for x in xs:
        for y in ys:
            pos.append([x, y, 0.0])
            depth.append(float(x))  # depth depends only on x
    return _StubTest(np.array(pos), source_depths=np.array(depth))


def test_depth_matched_pairs_respect_both_tolerances(depth_grid):
    """Selected pairs must match target separation AND sit in the depth bin."""
    bins = [('shallow', 0, 30), ('deep', 70, 100)]
    pairs = depth_grid._select_depth_matched_pairs(
        [4.0], bins, n_pairs=3,
        separation_tolerance_mm=0.5, depth_tolerance_mm=0.5
    )
    d = depth_grid.source_depths
    for (label, sep), plist in pairs.items():
        lo, hi = np.percentile(d, dict(shallow=(0, 30), deep=(70, 100))[label])
        for idx1, idx2 in plist:
            actual = np.linalg.norm(
                depth_grid.source_pos_mm[idx1] - depth_grid.source_pos_mm[idx2])
            assert abs(actual - sep) <= 0.5
            # both sources within the depth bin and matched to each other
            assert lo - 1e-9 <= d[idx1] <= hi + 1e-9
            assert abs(d[idx1] - d[idx2]) <= 0.5


def test_depth_matched_pairs_separate_shallow_from_deep(depth_grid):
    """Shallow-bin pairs must actually be shallower than deep-bin pairs."""
    bins = [('shallow', 0, 25), ('deep', 75, 100)]
    pairs = depth_grid._select_depth_matched_pairs(
        [4.0], bins, n_pairs=5,
        separation_tolerance_mm=0.5, depth_tolerance_mm=0.5
    )
    d = depth_grid.source_depths
    shallow_depths = [0.5 * (d[i] + d[j]) for i, j in pairs[('shallow', 4.0)]]
    deep_depths = [0.5 * (d[i] + d[j]) for i, j in pairs[('deep', 4.0)]]
    assert shallow_depths and deep_depths
    assert max(shallow_depths) < min(deep_depths)


def test_depth_matched_pairs_empty_cell_when_unsatisfiable(depth_grid):
    """A separation with no same-depth partner yields an empty cell, not a crash."""
    bins = [('shallow', 0, 30)]
    # 50mm separation is unreachable on this grid.
    pairs = depth_grid._select_depth_matched_pairs(
        [50.0], bins, n_pairs=3,
        separation_tolerance_mm=0.5, depth_tolerance_mm=0.5
    )
    assert pairs[('shallow', 50.0)] == []


# ---- axis-oriented pair selection ----------------------------------------

@pytest.fixture
def cube_grid():
    """A 3D lattice so pairs can be oriented along any axis."""
    xs = np.arange(0, 11, 2.0)
    pos = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=float)
    return _StubTest(pos)


def test_axis_pairs_are_oriented_along_requested_axis(cube_grid):
    """A-P (y) pairs must displace mainly in y; L-R (x) pairs mainly in x."""
    axes = [('AP', (0, 1, 0)), ('LR', (1, 0, 0)), ('DV', (0, 0, 1))]
    pairs = cube_grid._select_axis_pairs(
        [4.0], axes, n_pairs=8,
        separation_tolerance_mm=0.5, angular_tolerance_deg=15.0)
    axis_col = {'AP': 1, 'LR': 0, 'DV': 2}
    for (label, sep), plist in pairs.items():
        assert plist, f"no pairs for {label}"
        col = axis_col[label]
        for i, j in plist:
            disp = np.abs(cube_grid.source_pos_mm[i] - cube_grid.source_pos_mm[j])
            # displacement concentrated on the axis's own coordinate
            assert disp[col] == pytest.approx(np.linalg.norm(disp), abs=0.6)
            assert np.linalg.norm(disp) == pytest.approx(sep, abs=0.5)


def test_axis_pairs_none_direction_is_isotropic():
    """A None direction accepts pairs in any orientation.

    Uses an irregular (jittered) grid: on a perfectly axis-aligned lattice the
    deterministic nearest-separation tiebreak would always land on one axis,
    which is a fixture artifact, not the behavior on the real source grid.
    """
    xs = np.arange(0, 11, 2.0)
    base = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=float)
    jitter = (np.sin(np.arange(base.size)).reshape(base.shape)) * 0.4
    stub = _StubTest(base + jitter)
    pairs = stub._select_axis_pairs(
        [4.0], [('any', None)], n_pairs=20, separation_tolerance_mm=0.8)
    plist = pairs[('any', 4.0)]
    assert len(plist) >= 5
    disps = np.array([np.abs(stub.source_pos_mm[i] - stub.source_pos_mm[j])
                      for i, j in plist])
    dominant_axis = disps.argmax(axis=1)
    assert len(set(dominant_axis.tolist())) > 1  # orientations genuinely mix


def test_axis_pairs_angular_tolerance_excludes_off_axis(cube_grid):
    """A tight angular tolerance along y must reject diagonal displacements."""
    pairs = cube_grid._select_axis_pairs(
        [4.0], [('AP', (0, 1, 0))], n_pairs=20,
        separation_tolerance_mm=0.5, angular_tolerance_deg=10.0)
    for i, j in pairs[('AP', 4.0)]:
        disp = cube_grid.source_pos_mm[j] - cube_grid.source_pos_mm[i]
        cos_ang = abs(disp[1]) / np.linalg.norm(disp)
        assert cos_ang >= np.cos(np.deg2rad(10.0)) - 1e-9


# ---- resolvability detector ----------------------------------------------

_StubResolve = RobustnessTest  # methods borrowed below


@pytest.fixture
def line_grid():
    """Fine 1D source line (0..10 mm at 0.25 mm) for controlled activity maps."""
    xs = np.arange(0, 10.001, 0.25)
    pos = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    stub = _StubTest(pos)
    # borrow the resolvability methods onto the stub instance's class chain
    for name in ('_get_source_kdtree', '_median_grid_spacing', '_local_maxima',
                 '_segment_trough', '_detect_two_lobes', '_resolve_two_sources'):
        setattr(_StubTest, name, getattr(RobustnessTest, name))
    return stub


def _gaussian_activity(stub, centers, sigma=0.5, amps=None):
    """Sum-of-Gaussians magnitude map over the stub's source line, as (n,1)."""
    x = stub.source_pos_mm[:, 0]
    amps = amps or [1.0] * len(centers)
    val = np.zeros_like(x)
    for c, a in zip(centers, amps):
        val += a * np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    return val[:, None]


def test_resolved_two_clear_peaks(line_grid):
    """Two narrow, well-separated peaks with a deep dip -> resolved."""
    activity = _gaussian_activity(line_grid, [2.0, 6.0], sigma=0.5)
    true = np.array([[2.0, 0, 0], [6.0, 0, 0]])
    res = line_grid._resolve_two_sources(activity, true, saddle_ratio=0.8)
    assert res['resolved'] is True
    assert res['n_candidate_peaks'] >= 1
    assert res['distinct_sources'] is True
    assert res['saddle_ratio_observed'] < 0.8


def test_not_resolved_single_blob(line_grid):
    """One source (single peak, monotonic decay) -> no second peak -> not resolved."""
    activity = _gaussian_activity(line_grid, [5.0], sigma=1.0)
    true = np.array([[4.0, 0, 0], [6.0, 0, 0]])
    res = line_grid._resolve_two_sources(activity, true)
    assert res['resolved'] is False
    assert res['n_candidate_peaks'] == 0


def test_prominence_gate_rejects_small_second_peak(line_grid):
    """A tiny second bump (noise-like) must NOT count as a second source.

    This is the guard against the false-positive floor: a single dipole plus a
    small ripple was mislabeled as two sources ~40-50% of the time until the
    second peak was required to be a substantial fraction of the first.
    """
    # Big peak at 3, tiny bump (20% height) at 7.
    activity = _gaussian_activity(line_grid, [3.0, 7.0], sigma=0.4, amps=[1.0, 0.2])
    true = np.array([[3.0, 0, 0], [7.0, 0, 0]])
    strict = line_grid._resolve_two_sources(activity, true, prominence_frac=0.5)
    lenient = line_grid._resolve_two_sources(activity, true, prominence_frac=0.1)
    assert strict['n_candidate_peaks'] == 0   # 0.2 < 0.5 -> no valid second peak
    assert strict['resolved'] is False
    assert lenient['resolved'] is True        # 0.2 > 0.1 -> accepted, and resolves


def test_not_resolved_when_dip_too_shallow(line_grid):
    """Two broad, overlapping peaks with only a shallow dip -> merged, not resolved."""
    activity = _gaussian_activity(line_grid, [2.0, 6.0], sigma=1.7)
    true = np.array([[2.0, 0, 0], [6.0, 0, 0]])
    res = line_grid._resolve_two_sources(activity, true, saddle_ratio=0.8)
    # A saddle may or may not be detected, but the dip is too shallow to pass.
    assert res['saddle_ratio_observed'] > 0.8
    assert res['resolved'] is False


def test_correspondence_is_parameter_free_not_a_localization_gate(line_grid):
    """Peaks offset from the true sources still resolve, as long as each peak is
    nearest a DIFFERENT true source. This is the fix for the detector conflating
    resolvability with localization accuracy: a fixed mm tolerance tighter than
    the localization error would wrongly reject genuine two-lobe detections."""
    # Peaks at 2 and 6; true sources shifted ~1mm off (3 and 7). Each peak is
    # still nearest a distinct true source, so the pair is resolved.
    activity = _gaussian_activity(line_grid, [2.0, 6.0], sigma=0.5)
    true = np.array([[3.0, 0, 0], [7.0, 0, 0]])
    res = line_grid._resolve_two_sources(activity, true, saddle_ratio=0.8)
    assert res['distinct_sources'] is True
    assert res['resolved'] is True
    assert res['match_max_mm'] > 0.5  # peaks are NOT accurately localized...
    # ...yet the pair is still (correctly) counted as resolved.


def test_not_resolved_when_both_peaks_nearest_same_source(line_grid):
    """Two peaks that both fall nearest ONE true source -> not two distinct lobes."""
    activity = _gaussian_activity(line_grid, [2.0, 6.0], sigma=0.5)
    true = np.array([[8.5, 0, 0], [9.5, 0, 0]])  # both peaks are left of both sources
    res = line_grid._resolve_two_sources(activity, true, saddle_ratio=0.8)
    assert res['saddle_ok'] is True         # the dip itself is fine
    assert res['distinct_sources'] is False  # but both peaks map to the same source
    assert res['resolved'] is False


def test_max_match_cap_rejects_wildly_off_peaks(line_grid):
    """The optional sanity cap rejects distinct-but-far peaks when enabled."""
    activity = _gaussian_activity(line_grid, [1.0, 6.0], sigma=0.4)
    true = np.array([[1.5, 0, 0], [10.0, 0, 0]])  # source2 at 10; peak2 (~6) is 4mm off
    # Distinct-nearest passes (peak1->1.5, peak2->10, since 6 is closer to 10
    # than to 1.5), but peak2 is ~4mm from its assigned source.
    uncapped = line_grid._resolve_two_sources(activity, true, saddle_ratio=0.8)
    capped = line_grid._resolve_two_sources(activity, true, saddle_ratio=0.8,
                                            max_match_mm=3.0)
    assert uncapped['distinct_sources'] is True
    assert uncapped['resolved'] is True
    assert uncapped['match_max_mm'] > 3.0
    assert capped['resolved'] is False  # cap catches the far peak


def test_saddle_ratio_threshold_is_respected(line_grid):
    """A lenient saddle_ratio should resolve a pair that a strict one rejects."""
    activity = _gaussian_activity(line_grid, [3.0, 6.0], sigma=1.1)
    true = np.array([[3.0, 0, 0], [6.0, 0, 0]])
    observed = line_grid._resolve_two_sources(
        activity, true, saddle_ratio=0.99
    )['saddle_ratio_observed']
    lenient = line_grid._resolve_two_sources(
        activity, true, saddle_ratio=observed + 0.02)
    strict = line_grid._resolve_two_sources(
        activity, true, saddle_ratio=observed - 0.02)
    assert lenient['resolved'] is True
    assert strict['resolved'] is False


def test_detect_two_lobes_is_correspondence_free(line_grid):
    """The null-scoring event: two-lobe detection needs no true positions.

    Both the one-source null and two-source signal are scored by this identical
    event, which is what makes the excess-over-null comparison apples-to-apples.
    """
    two = _gaussian_activity(line_grid, [2.0, 6.0], sigma=0.5)
    one = _gaussian_activity(line_grid, [5.0], sigma=1.0)
    assert line_grid._detect_two_lobes(two)['detected'] is True
    assert line_grid._detect_two_lobes(one)['detected'] is False
    # A tiny second bump must not trip it (prominence gate).
    ripple = _gaussian_activity(line_grid, [3.0, 7.0], sigma=0.4, amps=[1.0, 0.2])
    assert line_grid._detect_two_lobes(ripple, prominence_frac=0.5)['detected'] is False


def test_resolve_builds_on_detect(line_grid):
    """resolved must imply detected: correspondence only ever removes, never adds."""
    activity = _gaussian_activity(line_grid, [2.0, 6.0], sigma=0.5)
    true = np.array([[2.0, 0, 0], [6.0, 0, 0]])
    lobes = line_grid._detect_two_lobes(activity)
    res = line_grid._resolve_two_sources(activity, true)
    assert lobes['detected'] is True
    assert res['resolved'] is True
    # Same peaks surfaced by both.
    assert res['peak2_mm'] == lobes['peak2_mm']


def test_local_maxima_and_spacing(line_grid):
    """Sanity-check the grid helpers the detector relies on."""
    assert line_grid._median_grid_spacing() == pytest.approx(0.25, abs=1e-6)
    activity = _gaussian_activity(line_grid, [2.0, 6.0], sigma=0.5)[:, 0]
    maxima = line_grid._local_maxima(activity, radius_mm=0.4)
    peak_x = sorted(line_grid.source_pos_mm[maxima, 0])
    assert any(abs(px - 2.0) < 0.3 for px in peak_x)
    assert any(abs(px - 6.0) < 0.3 for px in peak_x)
