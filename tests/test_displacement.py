"""Tests for peak displacement (validation/displacement.py).

The arithmetic is pinned on hand-computable geometry. The reproduction of the
published MS1 table, from MS1's own confusion matrix and source space, lives in
surface-evoked (analysis/displacement_ms1_reproduce.py) because its inputs are
study data, not package data.
"""
import numpy as np
import pandas as pd
import pytest

from source_localization.validation import displacement as dp


@pytest.fixture
def geometry():
    # Three parcels on the x axis, centroids at 0, 3 and 4 mm; two points each.
    coords = np.array([[-1, 0, 0], [1, 0, 0],
                       [3, -1, 0], [3, 1, 0],
                       [4, 0, -2], [4, 0, 2]], float)
    return dp.parcel_geometry(coords, ['A', 'A', 'B', 'B', 'C', 'C'])


def test_geometry(geometry):
    assert list(geometry.index) == ['A', 'B', 'C']
    np.testing.assert_allclose(geometry.cx, [0, 3, 4])
    np.testing.assert_allclose(geometry.radius_mm, [1, 1, 2])
    assert list(geometry.n_points) == [2, 2, 2]


def test_unassigned_points_are_ignored():
    g = dp.parcel_geometry(np.zeros((3, 3)), ['A', None, ''])
    assert list(g.index) == ['A']


def test_perfect_attribution_has_zero_displacement(geometry):
    conf = pd.DataFrame(np.eye(3), index=list('ABC'), columns=list('ABC'))
    t = dp.displacement_table(conf, geometry).set_index('roi')
    np.testing.assert_allclose(t.displacement_mm, 0)
    np.testing.assert_allclose(t.recall, 1)
    np.testing.assert_allclose(t.information_gain, 1)
    assert t.displacement_given_error_mm.isna().all()


def test_hand_computed_row(geometry):
    # A: 50% right, 30% to B (3 mm), 20% to C (4 mm).
    conf = pd.DataFrame([[0.5, 0.3, 0.2]], index=['A'], columns=list('ABC'))
    r = dp.displacement_table(conf, geometry).iloc[0]
    assert r.displacement_mm == pytest.approx(0.3 * 3 + 0.2 * 4)
    assert r.displacement_given_error_mm == pytest.approx((0.3 * 3 + 0.2 * 4) / 0.5)
    assert r.null_displacement_mm == pytest.approx((0 + 3 + 4) / 3)
    assert r.information_gain == pytest.approx(1 - 1.7 / (7 / 3))
    assert r.displacement_in_radii == pytest.approx(1.7)
    assert r.recall == pytest.approx(0.5)


def test_counts_and_probabilities_agree(geometry):
    p = pd.DataFrame([[0.5, 0.3, 0.2]], index=['A'], columns=list('ABC'))
    t1 = dp.displacement_table(p, geometry)
    t2 = dp.displacement_table(p * 40, geometry)
    pd.testing.assert_frame_equal(t1, t2)


def test_columns_without_centroid_are_skipped(geometry):
    # MS1's matrices carry an '<outside>' column; it is dropped and the row
    # renormalised over the parcels that have a centroid.
    conf = pd.DataFrame([[0.4, 0.4, 0.0, 0.2]], index=['A'],
                        columns=['A', 'B', 'C', '<outside>'])
    r = dp.displacement_table(conf, geometry).iloc[0]
    assert r.displacement_mm == pytest.approx(0.4 * 3 / 0.8)
    assert r.recall == pytest.approx(0.4)


def test_operator_attributor_picks_largest_normalised_output():
    # Parcel P reads channel 0, Q reads channel 1 with ten times the gain.
    W = np.array([[1.0, 0.0], [0.0, 10.0]])
    names = ['Q', 'X', 'P']
    att = dp.operator_attributor(W, ['P', 'Q'], names)
    rng = np.random.default_rng(0)
    assert att(0, np.array([2.0, 1.0]), 0.0, rng) == names.index('P')
    raw = dp.operator_attributor(W, ['P', 'Q'], names, normalize=False)
    assert raw(0, np.array([2.0, 1.0]), 0.0, rng) == names.index('Q')
    # time series: power summed over samples
    assert att(0, np.array([[2.0, 0.0], [0.0, 1.5]]), 0.0, rng) == names.index('P')


def test_operator_attributor_rejects_unknown_parcel():
    with pytest.raises(ValueError):
        dp.operator_attributor(np.eye(2), ['P', 'Z'], ['P', 'Q'])
